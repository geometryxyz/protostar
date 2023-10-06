use core::num;
use std::{collections::BTreeSet, ops::Range};

use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;

use crate::{
    arithmetic::{log2_ceil, parallelize},
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        circuit::FloorPlanner, lookup, permutation, Advice, AdviceQuery, Any, Assigned, Assignment,
        Challenge, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, FixedQuery,
        Instance, InstanceQuery, Selector,
    },
    poly::{
        batch_invert_assigned,
        commitment::{Blind, Params},
        empty_lagrange, empty_lagrange_assigned, EvaluationDomain, LagrangeCoeff, Polynomial,
    },
};

use super::accumulator::committed::{batch_commit_transparent, Committed};

/// Contains all fixed data for a circuit that is required to create a Protostar `Accumulator`
#[derive(Debug, Clone)]
pub struct ProvingKey<C: CurveAffine> {
    pub domain: EvaluationDomain<C::Scalar>,
    // maximum number of rows in the trace (including blinding factors)
    pub num_rows: usize,
    // max active rows, without blinding
    pub usable_rows: Range<usize>,

    // The circuit's unmodified constraint system
    pub cs: ConstraintSystem<C::Scalar>,

    // Fixed columns
    pub fixed: Vec<Committed<C>>,
    pub selectors: Vec<Committed<C>>,
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Generate algebraic representation of the circuit.
    pub fn new<'params, P, ConcreteCircuit>(
        params: &P,
        circuit: &ConcreteCircuit,
    ) -> Result<ProvingKey<C>, Error>
    where
        P: Params<'params, C>,
        ConcreteCircuit: Circuit<C::Scalar>,
    {
        let num_rows = params.n() as usize;
        // k = log2(num_rows)
        let k = params.k();
        // Get `config` from the `ConstraintSystem`
        let mut cs = ConstraintSystem::default();
        #[cfg(feature = "circuit-params")]
        let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
        #[cfg(not(feature = "circuit-params"))]
        let config = ConcreteCircuit::configure(&mut cs);

        let cs = cs;

        let degree = cs.degree();

        let domain = EvaluationDomain::new(degree as u32, k);

        // TODO(@adr1anh): Blinding will be different for Protostar
        if num_rows < cs.minimum_rows() {
            return Err(Error::not_enough_rows_available(k));
        }

        let mut assembly: Assembly<C::Scalar> = Assembly {
            usable_rows: 0..num_rows - (cs.blinding_factors() + 1),
            k,

            fixed: vec![empty_lagrange_assigned(num_rows); cs.num_fixed_columns],
            permutation: permutation::keygen::Assembly::new(num_rows, &cs.permutation),
            selectors: vec![domain.empty_lagrange(); cs.num_selectors],

            _marker: std::marker::PhantomData,
        };

        // Synthesize the circuit to obtain URS
        ConcreteCircuit::FloorPlanner::synthesize(
            &mut assembly,
            circuit,
            config,
            cs.constants.clone(),
        )?;

        let fixed = batch_invert_assigned(assembly.fixed);

        let fixed: Vec<_> = fixed
            .into_iter()
            .map(|values| {
                let blind = Blind::default();
                let commitment = params.commit_lagrange(&values, blind).to_affine();
                Committed {
                    values,
                    commitment,
                    blind,
                }
            })
            .collect();

        let selectors = assembly
            .selectors
            .into_iter()
            .map(|values| {
                let blind = Blind::default();
                let commitment = params.commit_lagrange(&values, blind).to_affine();
                Committed {
                    values,
                    commitment,
                    blind,
                }
            })
            .collect();

        // We don't want to compress selectors for our usecase,
        // let (cs, selector_polys) = cs.compress_selectors(assembly.selectors);
        // fixed.extend(
        //     selector_polys
        //         .into_iter()
        //         .map(|poly| domain.lagrange_from_vec(poly)),
        // );

        Ok(ProvingKey {
            domain,
            num_rows,
            usable_rows: assembly.usable_rows,
            cs,
            fixed,
            selectors,
        })
    }

    /// Maximum degree over all gates in the circuit
    pub fn max_folding_constraints_degree(&self) -> usize {
        let mut max_degree = 0;

        // Get maximum degree over all gate polynomials
        for gate in &self.cs.gates {
            for poly in gate.polynomials() {
                max_degree = std::cmp::max(max_degree, poly.folding_degree());
            }
        }

        // Get maximum of all lookup constraints.
        // Add 1 to account for theta challenge
        // Add 1 to account for h/g
        for lookup in &self.cs.lookups {
            for poly in lookup
                .input_expressions
                .iter()
                .chain(lookup.table_expressions.iter())
            {
                max_degree = std::cmp::max(max_degree, poly.folding_degree() + 2);
            }
        }
        // add 1 for beta
        // add 1 for ys
        max_degree + 2
    }

    /// Total number of linearly-independent constraints, whose degrees are larger than 1
    pub fn num_folding_constraints(&self) -> usize {
        self.cs
            .gates
            .iter()
            .map(|gate| gate.polynomials().len())
            .sum::<usize>()
            + 2 * self.cs.lookups.len()
    }

    pub fn selector_ref(&self) -> Vec<&[C::Scalar]> {
        self.selectors.iter().map(|c| c.values.as_ref()).collect()
    }

    pub fn fixed_ref(&self) -> Vec<&[C::Scalar]> {
        self.fixed.iter().map(|c| c.values.as_ref()).collect()
    }
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    usable_rows: Range<usize>,
    k: u32,

    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<Polynomial<F, LagrangeCoeff>>,

    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> SyncDeps for Assembly<F> {}

impl<F: Field> Assignment<F> for Assembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.selectors[selector.0][row] = F::ONE;

        Ok(())
    }

    fn query_instance(&self, _: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // There is no instance in this context.
        Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about fixed columns here
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.permutation
            .copy(left_column, left_row, right_column, right_row)
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let col = self
            .fixed
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;

        let filler = to.assign()?;
        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = filler;
        }

        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

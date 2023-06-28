use std::ops::Range;

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        circuit::FloorPlanner, permutation, Advice, Any, Assigned, Assignment, Challenge, Circuit,
        Column, ConstraintSystem, Error, Fixed, Instance, Selector,
    },
    poly::{
        batch_invert_assigned, commitment::Params, empty_lagrange, empty_lagrange_assigned,
        EvaluationDomain, LagrangeCoeff, Polynomial,
    },
};

pub struct CircuitData<C: CurveAffine> {
    // number of rows
    pub n: u64,
    // ceil(log(n))
    pub k: u32,
    // max active rows, without blinding
    pub usable_rows: Range<usize>,
    // gates, lookups, advice, fixed columns
    pub cs: ConstraintSystem<C::Scalar>,
    // fixed[fixed_col][row]
    pub fixed: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    // selectors[gate][row]
    pub selectors: Vec<Vec<bool>>,
    // permutations[advice_col][row] = sigma(advice_col*n+row)
    // advice_col are from cs.permutation
    // maybe store as Vec<Vec<(usize,usize)>)
    pub permutations: Vec<Vec<usize>>,
    // lookups?
}

impl<C: CurveAffine> CircuitData<C> {
    /// Generate from the circuit, parametrized on the commitment scheme.
    pub fn new<'params, P, ConcreteCircuit>(
        params: &P,
        circuit: &ConcreteCircuit,
    ) -> Result<CircuitData<C>, Error>
    where
        C: CurveAffine,
        P: Params<'params, C>,
        ConcreteCircuit: Circuit<C::Scalar>,
    {
        let mut cs = ConstraintSystem::default();
        #[cfg(feature = "circuit-params")]
        let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
        #[cfg(not(feature = "circuit-params"))]
        let config = ConcreteCircuit::configure(&mut cs);

        // We probably want a different degree, but we may not even need this
        // let degree = cs.degree();

        let cs = cs;

        let n = params.n() as usize;

        if (params.n() as usize) < cs.minimum_rows() {
            return Err(Error::not_enough_rows_available(params.k()));
        }

        let mut assembly: Assembly<C::Scalar> = Assembly {
            k: params.k(),
            fixed: vec![empty_lagrange_assigned(n); cs.num_fixed_columns],
            permutation: permutation::keygen::Assembly::new(params.n() as usize, &cs.permutation),
            selectors: vec![vec![false; params.n() as usize]; cs.num_selectors],
            // We don't need blinding factors for Protostar, but later for the Decider,
            // leave for now
            usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
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

        // We don't want to compress selectors for our usecase,
        // TODO(@adr1anh): Handle panics for place which assume the simple selectors were removed
        // let (cs, selector_polys) = cs.compress_selectors(assembly.selectors);
        // fixed.extend(
        //     selector_polys
        //         .into_iter()
        //         .map(|poly| domain.lagrange_from_vec(poly)),
        // );

        let permutations = assembly
            .permutation
            .build_permutations(params, &cs.permutation);

        // Compute the optimized evaluation data structure
        // TODO(@adr1anh): Define different Evaluator structuse
        // let ev = Evaluator::new(&vk.cs);

        Ok(CircuitData {
            n: params.n(),
            k: params.k(),
            usable_rows: assembly.usable_rows,
            cs,
            fixed,
            selectors: assembly.selectors,
            permutations,
            // ev,
        })
    }
}

// TODO(@adr1anh): Derive from ProvingKey
// /// This is a verifying key which allows for the verification of proofs for a
// /// particular circuit.
// #[derive(Clone, Debug)]
// pub struct VerifyingKey<C: CurveAffine> {
//     domain: EvaluationDomain<C::Scalar>,
//     fixed_commitments: Vec<C>,
//     permutation: permutation::VerifyingKey<C>,
//     cs: ConstraintSystem<C::Scalar>,
//     /// Cached maximum degree of `cs` (which doesn't change after construction).
//     cs_degree: usize,
//     /// The representative of this `VerifyingKey` in transcripts.
//     transcript_repr: C::Scalar,
//     selectors: Vec<Vec<bool>>,
// }

// TODO: generate VerifyingKey after ProvingKey is generated

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    k: u32,
    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<Vec<bool>>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
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

        self.selectors[selector.0][row] = true;

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

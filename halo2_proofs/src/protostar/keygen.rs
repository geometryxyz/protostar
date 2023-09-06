use core::num;
use std::{collections::BTreeSet, ops::Range};

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    arithmetic::log2_ceil,
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        circuit::FloorPlanner, lookup, permutation, Advice, AdviceQuery, Any, Assigned, Assignment,
        Challenge, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, FixedQuery,
        Instance, InstanceQuery, Selector,
    },
    poly::{
        batch_invert_assigned, commitment::Params, empty_lagrange, empty_lagrange_assigned,
        EvaluationDomain, LagrangeCoeff, Polynomial,
    },
};

use super::error_check;

/// Contains all fixed data for a circuit that is required to create a Protostar `Accumulator`
#[derive(Debug)]
pub struct ProvingKey<C: CurveAffine> {
    // maximum number of rows in the trace (including blinding factors)
    pub num_rows: usize,
    // max active rows, without blinding
    pub usable_rows: Range<usize>,

    // The circuit's unmodified constraint system
    pub cs: ConstraintSystem<C::Scalar>,

    // Fixed columns
    // fixed[col][row]
    pub fixed: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    // Selector columns as `bool`s
    // selectors[col][row]
    // TODO(@adr1anh): Replace with a `BTreeMap` to save memory
    pub selectors: Vec<BTreeSet<usize>>,
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

        // TODO(@adr1anh): Blinding will be different for Protostar
        if num_rows < cs.minimum_rows() {
            return Err(Error::not_enough_rows_available(k));
        }

        let mut assembly: Assembly<C::Scalar> = Assembly {
            usable_rows: 0..num_rows - (cs.blinding_factors() + 1),
            k,

            fixed: vec![empty_lagrange_assigned(num_rows); cs.num_fixed_columns],
            permutation: permutation::keygen::Assembly::new(num_rows, &cs.permutation),
            selectors: vec![BTreeSet::new(); cs.num_selectors],

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
        // let (cs, selector_polys) = cs.compress_selectors(assembly.selectors);
        // fixed.extend(
        //     selector_polys
        //         .into_iter()
        //         .map(|poly| domain.lagrange_from_vec(poly)),
        // );

        Ok(ProvingKey {
            num_rows,
            usable_rows: assembly.usable_rows,
            cs,
            fixed,
            selectors: assembly.selectors,
        })
    }

    // /// Maximum degree over all gates in the circuit
    // pub fn max_degree(&self) -> usize {
    //     let mut max_degree = 0;

    //     // Get maximum degree over all gate polynomials
    //     for gate in &self.cs.gates {
    //         for poly in gate.polynomials() {
    //             max_degree = std::cmp::max(max_degree, poly.folding_degree());
    //         }
    //     }

    //     // Get maximum of all lookup constraints.
    //     // Add 1 to account for theta challenge
    //     // Add 1 to account for h/g
    //     for lookup in &self.cs.lookups {
    //         for poly in lookup
    //             .input_expressions
    //             .iter()
    //             .chain(lookup.table_expressions.iter())
    //         {
    //             max_degree = std::cmp::max(max_degree, poly.folding_degree() + 2);
    //         }
    //     }
    //     max_degree
    // }

    /// Total number of linearly-independent constraints, whose degrees are larger than 1
    pub fn num_folding_constraints(&self) -> usize {
        self.cs
            .gates
            .iter()
            .map(|gate| gate.polynomials().len())
            .sum::<usize>()
            + 2 * self.cs.lookups.len()
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
    usable_rows: Range<usize>,
    // TODO(@adr1anh): Only needed for the Error, remove later
    k: u32,

    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    // TODO(@adr1anh): Replace with Vec<BTreeSet<bool>>
    selectors: Vec<BTreeSet<usize>>,

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

        self.selectors[selector.0].insert(row);

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

// struct VerifyingKey<C: CurveAffine> {
//     _phantom: PhantomData<C>,
// }

// /// A Protostar proving key that augments `CircuitData`
// pub struct ProvingKey<'cd, F: Field> {
//     circuit_data: &'cd CircuitData<F>,
//     pub gates: Vec<Gate<F>>,
//     pub max_challenge_power: Vec<usize>,
//     // vk: VerifyingKey<C>,
//     // ev etc?
// }

// impl<'cd, F: Field> ProvingKey<'cd, F> {
//     pub fn new(circuit_data: &'cd CircuitData<F>) -> Result<ProvingKey<'cd, F>, Error> {
//         // let vk = VerifyingKey {
//         //     _phantom: Default::default(),
//         // };
//         Ok(ProvingKey {
//             circuit_data,
//             gates,
//             max_challenge_power,
//         })
//     }
// }

/// Undo `Constraints::with_selector` and return the common top-level `Selector` along with the `Expression` it selects.
/// If no simple `Selector` is found, returns the original list of polynomials.
pub fn extract_common_simple_selector<F: Field>(
    polys: &[Expression<F>],
) -> (Vec<Expression<F>>, Option<Selector>) {
    let (extracted_polys, simple_selectors): (Vec<_>, Vec<_>) = polys
        .iter()
        .map(|poly| {
            // Check whether the top node is a multiplication by a selector
            let (simple_selector, poly) = match poly {
                // If the whole polynomial is multiplied by a simple selector,
                // return it along with the expression it selects
                Expression::Product(e1, e2) => match (&**e1, &**e2) {
                    (Expression::Selector(s), e) | (e, Expression::Selector(s)) => (Some(*s), e),
                    _ => (None, poly),
                },
                _ => (None, poly),
            };
            (poly.clone(), simple_selector)
        })
        .unzip();

    // Check if all simple selectors are the same and if so select it
    let potential_selector = match simple_selectors.as_slice() {
        [head, tail @ ..] => {
            if let Some(s) = *head {
                tail.iter().all(|x| x.is_some_and(|x| s == x)).then(|| s)
            } else {
                None
            }
        }
        [] => None,
    };

    // if we haven't found a common simple selector, then we just use the previous polys
    if potential_selector.is_none() {
        (polys.to_vec(), None)
    } else {
        (extracted_polys, potential_selector)
    }
}

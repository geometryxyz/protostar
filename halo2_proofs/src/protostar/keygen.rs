use std::{collections::BTreeSet, ops::Range};

use super::gate::{Expr, Gate};
use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        circuit::FloorPlanner, permutation, Advice, AdviceQuery, Any, Assigned, Assignment,
        Challenge, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, FixedQuery,
        Instance, InstanceQuery, Selector,
    },
    poly::{
        batch_invert_assigned, commitment::Params, empty_lagrange, empty_lagrange_assigned,
        EvaluationDomain, LagrangeCoeff, Polynomial,
    },
};

/// CircuitData is the minimal algebraic representation of a PlonK-ish circuit.
/// From this, we can derive the Proving/Verification keys for a chosen proof system.
pub struct CircuitData<F: Field> {
    // maximum number of rows in the trace
    pub n: usize,
    // ceil(log(n))
    pub k: u32,
    // max active rows, without blinding
    pub usable_rows: Range<usize>,
    // original constraint system
    pub cs: ConstraintSystem<F>,

    // actual number of elements in each column for each column type.
    pub num_selector_rows: Vec<usize>,
    pub num_fixed_rows: Vec<usize>,
    pub num_advice_rows: Vec<usize>,
    pub num_instance_rows: Vec<usize>,

    // fixed[fixed_col][row]
    pub fixed: Vec<Polynomial<F, LagrangeCoeff>>,
    // selectors[gate][row]
    // TODO(@adr1anh): Replace with a `BTreeMap` to save memory
    pub selectors: Vec<Vec<bool>>,

    // permutations[advice_col][row] = sigma(advice_col*n+row)
    // advice_col are from cs.permutation
    // TODO(@adr1anh): maybe store as Vec<Vec<(usize,usize)>)
    pub permutations: Vec<Vec<usize>>,
    // lookups?
}

impl<F: Field> CircuitData<F> {
    /// Generate algebraic representation of the circuit.
    pub fn new<ConcreteCircuit>(
        n: usize,
        circuit: &ConcreteCircuit,
    ) -> Result<CircuitData<F>, Error>
    where
        ConcreteCircuit: Circuit<F>,
    {
        // Get `config` from the `ConstraintSystem`
        let mut cs = ConstraintSystem::default();
        #[cfg(feature = "circuit-params")]
        let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
        #[cfg(not(feature = "circuit-params"))]
        let config = ConcreteCircuit::configure(&mut cs);

        let cs = cs;
        let k = n.next_power_of_two() as u32;

        // TODO(@adr1anh): Blinding will be different for Protostar
        if n < cs.minimum_rows() {
            return Err(Error::not_enough_rows_available(k));
        }

        let mut assembly: Assembly<F> = Assembly {
            k,
            fixed: vec![empty_lagrange_assigned(n); cs.num_fixed_columns],
            permutation: permutation::keygen::Assembly::new(n, &cs.permutation),
            selectors: vec![vec![false; n]; cs.num_selectors],
            // We don't need blinding factors for Protostar, but later for the Decider,
            // leave for now
            usable_rows: 0..n,

            num_selector_rows: vec![0; cs.num_selectors],
            num_fixed_rows: vec![0; cs.num_fixed_columns],
            num_advice_rows: vec![0; cs.num_advice_columns],
            num_instance_rows: vec![0; cs.num_instance_columns],

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

        let permutations = assembly.permutation.build_permutations(n, &cs.permutation);

        Ok(CircuitData {
            n,
            k,
            usable_rows: assembly.usable_rows,
            cs,
            fixed,
            selectors: assembly.selectors,
            permutations,
            num_selector_rows: assembly.num_selector_rows,
            num_fixed_rows: assembly.num_fixed_rows,
            num_advice_rows: assembly.num_advice_rows,
            num_instance_rows: assembly.num_instance_rows,
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

    // keep track of actual number of elements in each column
    num_selector_rows: Vec<usize>,
    num_fixed_rows: Vec<usize>,
    num_advice_rows: Vec<usize>,
    num_instance_rows: Vec<usize>,

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
        self.num_selector_rows[selector.0] = std::cmp::max(self.num_selector_rows[selector.0], row);

        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.selectors[selector.0][row] = true;

        Ok(())
    }

    fn query_instance(&mut self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        let column_index = column.index();
        self.num_instance_rows[column_index] =
            std::cmp::max(self.num_instance_rows[column_index], row);
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // There is no instance in this context.
        Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Advice>,
        row: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let column_index = column.index();
        self.num_advice_rows[column_index] = std::cmp::max(self.num_advice_rows[column_index], row);
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
        let column_index = column.index();
        self.num_fixed_rows[column_index] = std::cmp::max(self.num_fixed_rows[column_index], row);

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
        let left_column_index = left_column.index();
        let right_column_index = right_column.index();

        match left_column.column_type() {
            Any::Advice(_) => {
                self.num_advice_rows[left_column_index] =
                    std::cmp::max(self.num_advice_rows[left_column_index], left_row);
            }
            Any::Fixed => {
                self.num_fixed_rows[left_column_index] =
                    std::cmp::max(self.num_fixed_rows[left_column_index], left_row);
            }
            Any::Instance => {
                self.num_instance_rows[left_column_index] =
                    std::cmp::max(self.num_instance_rows[left_column_index], left_row);
            }
        }
        match right_column.column_type() {
            Any::Advice(_) => {
                self.num_advice_rows[right_column_index] =
                    std::cmp::max(self.num_advice_rows[right_column_index], right_row);
            }
            Any::Fixed => {
                self.num_fixed_rows[right_column_index] =
                    std::cmp::max(self.num_fixed_rows[right_column_index], right_row);
            }
            Any::Instance => {
                self.num_instance_rows[right_column_index] =
                    std::cmp::max(self.num_instance_rows[right_column_index], right_row);
            }
        }

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

/// A Protostar proving key that augments `CircuitData`
pub struct ProvingKey<'cd, F: Field> {
    pub circuit_data: &'cd CircuitData<F>,
    pub gates: Vec<Gate<F>>,
    pub max_degree: usize,
    pub max_challenge_power: Vec<usize>,
    // vk: VerifyingKey<C>,
    // ev etc?
}

impl<'cd, F: Field> ProvingKey<'cd, F> {
    pub fn new(circuit_data: &'cd CircuitData<F>) -> Result<ProvingKey<'cd, F>, Error> {
        let gates: Vec<Gate<_>> = circuit_data.cs.gates().iter().map(Gate::from).collect();

        // for each challenge, find the maximum power of it appearing across all expressions
        let mut max_challenge_power = vec![0; circuit_data.cs.num_challenges];
        for gate in gates.iter() {
            for poly in gate.polys.iter() {
                poly.traverse(&mut |v| {
                    if let Expr::Challenge(c, d) = v {
                        max_challenge_power[*c] = std::cmp::max(max_challenge_power[*c], *d);
                    }
                });
            }
        }

        let max_degree = *gates
            .iter()
            .map(|g| g.degrees.iter().max().unwrap())
            .max()
            .unwrap();
        // let vk = VerifyingKey {
        //     _phantom: Default::default(),
        // };
        Ok(ProvingKey {
            circuit_data,
            gates,
            max_degree,
            max_challenge_power,
        })
    }
}

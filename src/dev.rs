//! Tools for developing circuits.

use ff::Field;

use crate::{
    arithmetic::{FieldExt, Group},
    plonk::{permutation, Any, Assignment, Circuit, Column, ConstraintSystem, Error},
    poly::{EvaluationDomain, LagrangeCoeff, Polynomial},
};

#[derive(Debug)]
struct Cell(usize, usize);

/// The reasons why a particular circuit is not satisfied.
#[derive(Debug, PartialEq)]
pub enum VerifyFailure {
    /// A gate was not satisfied for a particular row.
    Gate {
        /// The index of the gate that is not satisfied. These indices are assigned in the
        /// order in which `ConstraintSystem::create_gate` is called during
        /// `Circuit::configure`.
        gate_index: usize,
        /// The row on which this gate is not satisfied.
        row: usize,
    },
    /// A lookup input did not exist in its corresponding table.
    Lookup {
        /// The index of the lookup that is not satisfied. These indices are assigned in
        /// the order in which `ConstraintSystem::lookup` is called during
        /// `Circuit::configure`.
        lookup_index: usize,
        /// The row on which this lookup is not satisfied.
        row: usize,
    },
    /// A permutation did not preserve the original value of a cell.
    Permutation {
        /// The index of the permutation that is not satisfied. These indices are assigned
        /// in the order in which `ConstraintSystem::lookup` is called during
        /// `Circuit::configure`.
        perm_index: usize,
        /// The column in which this permutation is not satisfied.
        column: usize,
        /// The row on which this permutation is not satisfied.
        row: usize,
    },
}

/// A test prover for debugging circuits.
///
/// The normal proving process, when applied to a buggy circuit implementation, might
/// return proofs that do not validate when they should, but it can't indicate anything
/// other than "something is invalid". `MockProver` can be used to figure out _why_ these
/// are invalid: it stores all the private inputs along with the circuit internals, and
/// then checks every constraint manually.
///
/// # Examples
///
/// ```
/// use halo2::{
///     arithmetic::FieldExt,
///     dev::{MockProver, VerifyFailure},
///     pasta::Fp,
///     plonk::{Advice, Assignment, Circuit, Column, ConstraintSystem, Error},
/// };
/// const K: u32 = 5;
///
/// struct MyConfig {
///     a: Column<Advice>,
///     b: Column<Advice>,
///     c: Column<Advice>,
/// }
///
/// struct MyCircuit {
///     a: Option<u64>,
///     b: Option<u64>,
/// }
///
/// impl<F: FieldExt> Circuit<F> for MyCircuit {
///     type Config = MyConfig;
///
///     fn configure(meta: &mut ConstraintSystem<F>) -> MyConfig {
///         let a = meta.advice_column();
///         let b = meta.advice_column();
///         let c = meta.advice_column();
///
///         meta.create_gate(|meta| {
///             let a = meta.query_advice(a, 0);
///             let b = meta.query_advice(b, 0);
///             let c = meta.query_advice(c, 0);
///
///             // BUG: Should be a * b - c
///             a * b + c
///         });
///
///         MyConfig { a, b, c }
///     }
///
///     fn synthesize(&self, cs: &mut impl Assignment<F>, config: MyConfig) -> Result<(), Error> {
///         cs.assign_advice(config.a, 0, || {
///             self.a.map(|v| F::from_u64(v)).ok_or(Error::SynthesisError)
///         })?;
///         cs.assign_advice(config.b, 0, || {
///             self.b.map(|v| F::from_u64(v)).ok_or(Error::SynthesisError)
///         })?;
///         cs.assign_advice(config.c, 0, || {
///             self.a
///                 .and_then(|a| self.b.map(|b| F::from_u64(a * b)))
///                 .ok_or(Error::SynthesisError)
///         })
///     }
/// }
///
/// // Assemble the private inputs to the circuit.
/// let circuit = MyCircuit {
///     a: Some(2),
///     b: Some(4),
/// };
///
/// // This circuit has no public inputs.
/// let aux = vec![];
///
/// let prover = MockProver::<Fp>::run(K, &circuit, aux).unwrap();
/// assert_eq!(
///     prover.verify(),
///     Err(VerifyFailure::Gate {
///         gate_index: 0,
///         row: 0
///     })
/// );
/// ```
#[derive(Debug)]
pub struct MockProver<F: Group> {
    n: u32,
    domain: EvaluationDomain<F>,
    cs: ConstraintSystem<F>,

    // The fixed cells in the circuit, arranged as [column][row].
    fixed: Vec<Polynomial<F, LagrangeCoeff>>,
    // The advice cells in the circuit, arranged as [column][row].
    advice: Vec<Polynomial<F, LagrangeCoeff>>,
    // The aux cells in the circuit, arranged as [column][row].
    aux: Vec<Polynomial<F, LagrangeCoeff>>,

    permutations: Vec<permutation::keygen::Assembly>,
}

impl<F: Field + Group> Assignment<F> for MockProver<F> {
    fn assign_advice(
        &mut self,
        column: crate::plonk::Column<crate::plonk::Advice>,
        row: usize,
        to: impl FnOnce() -> Result<F, crate::plonk::Error>,
    ) -> Result<(), crate::plonk::Error> {
        *self
            .advice
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to()?;

        Ok(())
    }

    fn assign_fixed(
        &mut self,
        column: crate::plonk::Column<crate::plonk::Fixed>,
        row: usize,
        to: impl FnOnce() -> Result<F, crate::plonk::Error>,
    ) -> Result<(), crate::plonk::Error> {
        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to()?;

        Ok(())
    }

    fn copy(
        &mut self,
        permutation: usize,
        left_column: usize,
        left_row: usize,
        right_column: usize,
        right_row: usize,
    ) -> Result<(), crate::plonk::Error> {
        // Check bounds first
        if permutation >= self.permutations.len() {
            return Err(Error::BoundsFailure);
        }

        self.permutations[permutation].copy(left_column, left_row, right_column, right_row)
    }
}

impl<F: FieldExt> MockProver<F> {
    /// Runs a synthetic keygen-and-prove operation on the given circuit, collecting data
    /// about the constraints and their assignments.
    pub fn run<ConcreteCircuit: Circuit<F>>(
        k: u32,
        circuit: &ConcreteCircuit,
        aux: Vec<Polynomial<F, LagrangeCoeff>>,
    ) -> Result<Self, Error> {
        let n = 1 << k;

        let mut cs = ConstraintSystem::default();
        let config = ConcreteCircuit::configure(&mut cs);

        // The permutation argument will serve alongside the gates, so must be
        // accounted for.
        let mut degree = cs
            .permutations
            .iter()
            .map(|p| p.required_degree())
            .max()
            .unwrap_or(1);

        // Account for each gate to ensure our quotient polynomial is the
        // correct degree and that our extended domain is the right size.
        for poly in cs.gates.iter() {
            degree = std::cmp::max(degree, poly.degree());
        }

        let domain = EvaluationDomain::new(degree as u32, k);

        let fixed = vec![domain.empty_lagrange(); cs.num_fixed_columns];
        let advice = vec![domain.empty_lagrange(); cs.num_advice_columns];
        let permutations = cs
            .permutations
            .iter()
            .map(|p| permutation::keygen::Assembly::new(n as usize, p))
            .collect();

        let mut prover = MockProver {
            n,
            domain,
            cs,
            fixed,
            advice,
            aux,
            permutations,
        };

        circuit.synthesize(&mut prover, config)?;

        Ok(prover)
    }

    /// Returns `Ok(())` if this `MockProver` is satisfied, or an error indicating the
    /// first encountered reason that the circuit is not satisfied.
    pub fn verify(&self) -> Result<(), VerifyFailure> {
        let n = self.n as i32;

        // Check that all gates are satisfied for all rows.
        for (gate_index, gate) in self.cs.gates.iter().enumerate() {
            // We iterate from n..2n so we can just reduce to handle wrapping.
            for row in n..(2 * n) {
                if gate.evaluate(
                    &|index| {
                        let (column, at) = self.cs.fixed_queries[index];
                        let resolved_row = (row + at.0) % n;
                        self.fixed[column.index()][resolved_row as usize].clone()
                    },
                    &|index| {
                        let (column, at) = self.cs.advice_queries[index];
                        let resolved_row = (row + at.0) % n;
                        self.advice[column.index()][resolved_row as usize].clone()
                    },
                    &|index| {
                        let (column, at) = self.cs.aux_queries[index];
                        let resolved_row = (row + at.0) % n;
                        self.aux[column.index()][resolved_row as usize].clone()
                    },
                    &|a, b| a + &b,
                    &|a, b| a * &b,
                    &|a, scalar| a * scalar,
                ) != F::zero()
                {
                    return Err(VerifyFailure::Gate {
                        gate_index,
                        row: (row - n) as usize,
                    });
                }
            }
        }

        // Check that all lookups exist in their respective tables.
        for (lookup_index, lookup) in self.cs.lookups.iter().enumerate() {
            for input_row in 0..n {
                let load = |column: &Column<Any>, row| match column.column_type() {
                    Any::Fixed => self.fixed[column.index()][row as usize],
                    Any::Advice => self.advice[column.index()][row as usize],
                    Any::Aux => self.aux[column.index()][row as usize],
                };

                let inputs: Vec<_> = lookup
                    .input_columns
                    .iter()
                    .map(|c| load(c, input_row))
                    .collect();
                if !(0..n)
                    .map(|table_row| lookup.table_columns.iter().map(move |c| load(c, table_row)))
                    .any(|table_row| table_row.eq(inputs.iter().cloned()))
                {
                    return Err(VerifyFailure::Lookup {
                        lookup_index,
                        row: input_row as usize,
                    });
                }
            }
        }

        // Check that permutations preserve the original values of the cells.
        for (perm_index, assembly) in self.permutations.iter().enumerate() {
            // Original values of columns involved in the permutation
            let original = self.cs.permutations[perm_index]
                .get_columns()
                .iter()
                .map(|c| self.advice[c.index()].clone())
                .collect::<Vec<_>>();

            // Iterate over each column of the permutation
            for (column, values) in assembly.mapping.iter().enumerate() {
                // Iterate over each row of the column to check that the cell's
                // value is preserved by the mapping.
                for (row, cell) in values.iter().enumerate() {
                    let original_cell = original[column][row];
                    let permuted_cell = original[cell.0][cell.1];
                    if original_cell != permuted_cell {
                        return Err(VerifyFailure::Permutation {
                            perm_index,
                            column,
                            row,
                        });
                    }
                }
            }
        }

        // TODO: Implement the rest of the verification checks.

        Ok(())
    }
}

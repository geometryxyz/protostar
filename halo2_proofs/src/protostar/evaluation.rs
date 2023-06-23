use crate::plonk::permutation::Argument;
use crate::plonk::{permutation, AdviceQuery, Any, Challenge, FixedQuery, InstanceQuery};
use crate::poly::Basis;
use crate::protostar::expression::Expr;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    poly::{
        commitment::Params, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
        Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use crate::{multicore, poly::empty_lagrange};
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field, PrimeField, WithSmallOrderMulGroup},
    Curve,
};
use std::any::TypeId;
use std::convert::TryInto;
use std::num::ParseIntError;
use std::slice;
use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, Mul, MulAssign},
};

use crate::plonk::circuit::{ConstraintSystem, Expression};

use super::keygen::ProvingKey;

/// Return the index in the polynomial of size `isize` after rotation `rot`.
// TODO(@adr1anh) always inline
fn get_rotation_idx(idx: usize, rot: i32, isize: i32) -> usize {
    (((idx as i32) + rot).rem_euclid(isize)) as usize
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a slack variable whose purpose is to make the equation homogeneous
    Slack(usize),
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a selector column
    Selector(usize, usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge, where the second parameter represents the power of a Challenge.
    /// Two challenges with different powers are treated as separate variables.
    Challenge(usize, usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        slacks: &[F],
        constants: &[F],
        intermediates: &[F],
        selectors: &[Vec<bool>],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[&[F]],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Slack(idx) => slacks[*idx],
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Selector(column_index, rotation) => {
                if selectors[*column_index][rotations[*rotation]] {
                    F::ONE
                } else {
                    F::ZERO
                }
            }
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index, power) => challenges[*index][*power - 1],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        slacks: &[F],
        constants: &[F],
        intermediates: &[F],
        selectors: &[Vec<bool>],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[&[F]],
        // challenges: &[&[F]],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                slacks,
                constants,
                intermediates,
                selectors,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                // This should probably be a panic ?
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<C>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<C>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Slacks
    pub slacks: Vec<C::ScalarExt>,
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(gate.polynomials().iter().map(|poly| {
                let e = Expr::<C::ScalarExt>::from(poly.clone());
                ev.custom_gates.add_expression(&e)
            }));
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // TODO(@adr1anh)
        // // Lookups
        // for lookup in cs.lookups.iter() {
        //     let mut graph = GraphEvaluator::default();

        //     let mut evaluate_lc = |expressions: &Vec<Expr<_>>| {
        //         let parts = expressions
        //             .iter()
        //             .map(|expr| graph.add_expression(expr))
        //             .collect();
        //         graph.add_calculation(Calculation::Horner(
        //             ValueSource::Constant(0),
        //             parts,
        //             ValueSource::Theta(),
        //         ))
        //     };

        //     // Input coset
        //     // Map halo2 Expression to Protostar Expression
        //     let compressed_input_coset = evaluate_lc(
        //         &lookup
        //             .input_expressions
        //             .iter()
        //             .map(|expr| Expr::from(*expr))
        //             .collect(),
        //     );
        //     // table coset
        //     // Map halo2 Expression to Protostar Expression
        //     let compressed_table_coset = evaluate_lc(
        //         &lookup
        //             .table_expressions
        //             .iter()
        //             .map(|expr| Expr::from(*expr))
        //             .collect(),
        //     );
        //     // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
        //     let right_gamma = graph.add_calculation(Calculation::Add(
        //         compressed_table_coset,
        //         ValueSource::Gamma(),
        //     ));
        //     let lc = graph.add_calculation(Calculation::Add(
        //         compressed_input_coset,
        //         ValueSource::Beta(),
        //     ));
        //     graph.add_calculation(Calculation::Mul(lc, right_gamma));

        //     ev.lookups.push(graph);
        // }

        ev
    }

    /// Evaluate h poly
    fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice: &[&[Polynomial<C::ScalarExt, LagrangeCoeff>]],
        instance: &[&[Polynomial<C::ScalarExt, LagrangeCoeff>]],
        challenges: &[&[C::ScalarExt]],
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        // lookups: &[Vec<lookup::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Polynomial<C::ScalarExt, LagrangeCoeff> {
        // TODO: extended domain can be removed, no ffts are needed
        let size = pk.n as usize;
        // let fixed = &pk.fixed_cosets[..];
        // let extended_omega = domain.get_extended_omega();
        let isize = size as i32;
        let one = C::ScalarExt::ONE;

        let mut values = empty_lagrange(size);

        // Core expression evaluations
        let num_threads = multicore::current_num_threads();
        for (advice, instance) in advice.iter().zip(instance.iter())
        // for ((advice, instance), lookups) in advice.iter().zip(instance.iter()).zip(lookups.iter())
        {
            multicore::scope(|scope| {
                let chunk_size = (size + num_threads - 1) / num_threads;
                for (thread_idx, values) in values.chunks_mut(chunk_size).enumerate() {
                    let start = thread_idx * chunk_size;
                    scope.spawn(move |_| {
                        let mut eval_data = self.custom_gates.instance();
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;
                            *value = self.custom_gates.evaluate(
                                &mut eval_data,
                                &pk.selectors,
                                &pk.fixed,
                                advice,
                                instance,
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                value,
                                idx,
                                isize,
                            );
                        }
                    });
                }
            });

            // // Permutations

            // // Lookups
            // for (n, lookup) in lookups.iter().enumerate() {
            //     // Polynomials required for this lookup.
            //     // Calculated here so these only have to be kept in memory for the short time
            //     // they are actually needed.
            //     let product_coset = pk.vk.domain.coeff_to_extended(lookup.product_poly.clone());
            //     let permuted_input_coset = pk
            //         .vk
            //         .domain
            //         .coeff_to_extended(lookup.permuted_input_poly.clone());
            //     let permuted_table_coset = pk
            //         .vk
            //         .domain
            //         .coeff_to_extended(lookup.permuted_table_poly.clone());

            //     // Lookup constraints
            //     parallelize(&mut values, |values, start| {
            //         let lookup_evaluator = &self.lookups[n];
            //         let mut eval_data = lookup_evaluator.instance();
            //         for (i, value) in values.iter_mut().enumerate() {
            //             let idx = start + i;

            //             let table_value = lookup_evaluator.evaluate(
            //                 &mut eval_data,
            //                 fixed,
            //                 advice,
            //                 instance,
            //                 challenges,
            //                 &beta,
            //                 &gamma,
            //                 &theta,
            //                 &y,
            //                 &C::ScalarExt::ZERO,
            //                 idx,
            //                 rot_scale,
            //                 isize,
            //             );

            //             let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
            //             let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);

            //             let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
            //             // l_0(X) * (1 - z(X)) = 0
            //             *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
            //             // l_last(X) * (z(X)^2 - z(X)) = 0
            //             *value = *value * y
            //                 + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
            //                     * l_last[idx]);
            //             // (1 - (l_last(X) + l_blind(X))) * (
            //             //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            //             //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
            //             //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            //             // ) = 0
            //             *value = *value * y
            //                 + ((product_coset[r_next]
            //                     * (permuted_input_coset[idx] + beta)
            //                     * (permuted_table_coset[idx] + gamma)
            //                     - product_coset[idx] * table_value)
            //                     * l_active_row[idx]);
            //             // Check that the first values in the permuted input expression and permuted
            //             // fixed expression are the same.
            //             // l_0(X) * (a'(X) - s'(X)) = 0
            //             *value = *value * y + (a_minus_s * l0[idx]);
            //             // Check that each value in the permuted lookup input expression is either
            //             // equal to the value above it, or the value at the same index in the
            //             // permuted table expression.
            //             // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            //             *value = *value * y
            //                 + (a_minus_s
            //                     * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
            //                     * l_active_row[idx]);
            //         }
            //     });
            // }
        }
        values
    }
}

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            slacks: Vec::new(),
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::ZERO,
                C::ScalarExt::ONE,
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expr<C::ScalarExt>) -> ValueSource {
        match expr {
            // TODO(gnosed): check if Slack should be saved in the Calculation::Store or not
            Expr::Slack(d) => unreachable!(),
            Expr::Constant(scalar) => self.add_constant(scalar),
            Expr::Selector(_selector) => unreachable!(),
            Expr::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expr::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expr::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expr::Challenge(value, power) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(value.index(), *power),
            )),
            Expr::Negated(a) => match **a {
                Expr::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expr::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expr::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expr::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expr::Scaled(a, f) => {
                if *f == C::ScalarExt::ZERO {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<C> {
        EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[&[C::ScalarExt]],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,

        isize: i32,
    ) -> C::ScalarExt {
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, isize);
        }
        let slack = C::ScalarExt::ONE;

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &[slack],
                &self.constants,
                &data.intermediates,
                selectors,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            C::ScalarExt::ZERO
        }
    }
}

// this function is called once in plonk::lookup::prover::commit_permuted
/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expr<F>,
    size: usize,
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    slacks: &[F],
    challenges: &[&[&F]],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|slack| slacks[slack],
                &|scalar| scalar,
                &|selector| {
                    if selectors[selector.index()][idx] {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                }, //TODO
                &|query| fixed[query.column_index][get_rotation_idx(idx, query.rotation.0, isize)],
                &|query| advice[query.column_index][get_rotation_idx(idx, query.rotation.0, isize)],
                &|query| {
                    instance[query.column_index][get_rotation_idx(idx, query.rotation.0, isize)]
                },
                &|challenge, degree| *challenges[challenge.index()][degree - 1],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}

mod test {
    #[test]
    fn test_evaluate_h() {
        // TODO(gnosed):
        // Evaluate the h(X) polynomial
        // let h_poly = pk.ev.evaluate_h(
        //     pk,
        //     &advice
        //         .iter()
        //         .map(|a| a.advice_polys.as_slice())
        //         .collect::<Vec<_>>(),
        //     &instance
        //         .iter()
        //         .map(|i| i.instance_polys.as_slice())
        //         .collect::<Vec<_>>(),
        //     &challenges,
        //     *y,
        //     *beta,
        //     *gamma,
        //     *theta,
        //     &lookups,
        //     &permutations,
        // );
    }
}

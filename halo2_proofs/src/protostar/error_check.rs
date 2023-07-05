use core::num;
use std::collections::BTreeMap;

use crate::poly::{commitment::Blind, empty_lagrange, LagrangeCoeff, Polynomial, Rotation};
use ff::Field;
use halo2curves::CurveAffine;

use super::{
    gate::Gate,
    keygen::{CircuitData, ProvingKey},
};

///
#[derive(Clone)]
pub struct AccumulatorInstance<F: Field> {
    slack: F,
    challenges: Vec<Vec<F>>,
    instance: Vec<Polynomial<F, LagrangeCoeff>>,
}

impl<F: Field> AccumulatorInstance<F> {
    pub fn new_empty(pk: &ProvingKey<F>) -> Self {
        let challenges = vec![F::ZERO; pk.circuit_data.cs.num_challenges];
        let instance = vec![
            empty_lagrange(pk.circuit_data.n as usize);
            pk.circuit_data.cs.num_instance_columns
        ];
        AccumulatorInstance::new(pk, challenges, instance)
    }

    pub fn new(
        pk: &ProvingKey<F>,
        challenges: Vec<F>,
        instance: Vec<Polynomial<F, LagrangeCoeff>>,
    ) -> Self {
        let challenge_powers: Vec<_> = challenges
            .iter()
            .map(|c| powers_of(c, pk.max_degree))
            .collect();
        Self {
            slack: F::ONE,
            challenges: challenge_powers,
            instance,
        }
    }
}
#[derive(Clone)]
pub struct AccumulatorWitness<F: Field> {
    advice: Vec<Polynomial<F, LagrangeCoeff>>,
    advice_blinds: Vec<Blind<F>>,
}

impl<F: Field> AccumulatorWitness<F> {
    pub fn new_empty(n: usize, num_advice: usize) -> Self {
        Self {
            advice: vec![empty_lagrange(n); num_advice],
            advice_blinds: vec![Blind::default(); num_advice],
        }
    }
    pub fn new(advice: Vec<Polynomial<F, LagrangeCoeff>>, advice_blinds: Vec<Blind<F>>) -> Self {
        Self {
            advice,
            advice_blinds,
        }
    }
}

#[derive(Clone)]
pub struct GateEvaluationCache<F: Field> {
    gate: Gate<F>,
    // required number of evaluations for each gate
    num_evals: Vec<usize>,
    max_num_evals: usize,

    // precomputed powers of slack and challenges
    slack_powers_evals: Vec<Vec<F>>,
    challenge_powers_evals: Vec<EvaluatedChallenge<F>>,

    // Cached data for a row
    simple_selector: Option<bool>,
    selectors: Vec<F>,
    fixed: Vec<F>,
    instance_acc: Vec<F>,
    instance_new: Vec<F>,
    advice_acc: Vec<F>,
    advice_new: Vec<F>,

    // Local evaluation
    // gate_eval[gate_idx][gate_deg]
    gate_eval: Vec<Vec<F>>,
}

impl<F: Field> GateEvaluationCache<F> {
    pub fn new(
        gate: &Gate<F>,
        gate_accumulator: &AccumulatorInstance<F>,
        gate_transcript: &AccumulatorInstance<F>,
        num_extra_evaluations: usize,
    ) -> Self {
        let num_evals: Vec<_> = gate
            .degrees
            .iter()
            .map(|d| d + num_extra_evaluations)
            .collect();

        let max_poly_degree = *gate.degrees.iter().max().unwrap();
        let max_num_evals = max_poly_degree + num_extra_evaluations;

        // The accumual
        // Challenge evaluations: challenge_powers_evals[j][X][power] =
        //  challenges_acc[j][power] + X * challenges_new[j]^power
        // for
        //    j     = 0, 1, ..., num_challenges  - 1
        //    X     = 0, 1, ..., num_evals       - 1
        //    power = 0, 1, ..., max_poly_degree - 1
        let challenge_powers_evals: Vec<_> = {
            let challenges_acc = &gate_accumulator.challenges;
            let challenges_new = &gate_transcript.challenges;
            debug_assert_eq!(challenges_acc.len(), challenges_new.len());

            challenges_acc
                .iter()
                .zip(challenges_new.iter())
                .map(|(c_acc, c_new)| {
                    EvaluatedChallenge::new(c_acc, c_new, max_num_evals, max_poly_degree)
                })
                .collect()
        };

        // slack_powers_evals[X][power] = (slack + X)^power
        let slack_powers_evals = {
            let slack = gate_accumulator.slack;

            let mut slack_powers_evals = Vec::with_capacity(max_num_evals);
            // X = 0 => slack_powers_evals[0] = [1, slack, slack^2, ...]
            let mut acc = slack;
            slack_powers_evals.push(powers_of(&acc, max_poly_degree));
            for _i in 1..max_num_evals {
                acc += F::ONE;
                slack_powers_evals.push(powers_of(&acc, max_poly_degree));
            }

            slack_powers_evals
        };

        Self {
            gate: gate.clone(),
            num_evals,
            max_num_evals,
            slack_powers_evals,
            challenge_powers_evals,
            simple_selector: None,
            selectors: vec![F::ZERO; gate.queried_selectors.len()],
            fixed: vec![F::ZERO; gate.queried_fixed.len()],
            instance_acc: vec![F::ZERO; gate.queried_instance.len()],
            instance_new: vec![F::ZERO; gate.queried_instance.len()],
            advice_acc: vec![F::ZERO; gate.queried_advice.len()],
            advice_new: vec![F::ZERO; gate.queried_advice.len()],
            gate_eval: gate
                .degrees
                .iter()
                .map(|d| vec![F::ZERO; *d + num_extra_evaluations])
                .collect(),
        }
    }

    /// Fills the local variables buffers with data from the accumulator and new transcript
    fn populate(
        &mut self,
        row: usize,
        isize: i32,
        circuit_data: &CircuitData<F>,
        acc: &AccumulatorWitness<F>,
        new: &AccumulatorWitness<F>,
    ) {
        /// Return the index in the polynomial of size `isize` after rotation `rot`.
        fn get_rotation_idx(idx: usize, rot: Rotation, isize: i32) -> usize {
            (((idx as i32) + rot.0).rem_euclid(isize)) as usize
        }

        // Check if the gate is guarded by a simple selector and whether it is active
        if let Some(simple_selector) = self.gate.simple_selector {
            let selector_column = simple_selector.0;
            let value = circuit_data.selectors[selector_column][row];
            self.simple_selector = Some(value);
            // do nothing and return
            if !value {
                return;
            }
        }

        // Fill selectors
        for (i, selector) in self.gate.queried_selectors.iter().enumerate() {
            let selector_column = selector.0;
            let value = circuit_data.selectors[selector_column][row];
            self.selectors[i] = if value { F::ONE } else { F::ZERO };
        }

        // Fill fixed
        for (i, fixed) in self.gate.queried_fixed.iter().enumerate() {
            let fixed_column = fixed.column_index();
            let fixed_row = get_rotation_idx(row, fixed.rotation(), isize);
            self.fixed[i] = circuit_data.fixed[fixed_column][fixed_row];
        }

        // Fill instance
        for (i, instance) in self.gate.queried_instance.iter().enumerate() {
            let instance_column = instance.column_index();
            let instance_row = get_rotation_idx(row, instance.rotation(), isize);
            self.instance_acc[i] = acc.advice[instance_column][instance_row];
            self.instance_new[i] = new.advice[instance_column][instance_row];
        }

        // Fill advice
        for (i, advice) in self.gate.queried_instance.iter().enumerate() {
            let advice_column = advice.column_index();
            let advice_row = get_rotation_idx(row, advice.rotation(), isize);
            self.advice_acc[i] = acc.advice[advice_column][advice_row];
            self.advice_new[i] = new.advice[advice_column][advice_row];
        }
    }

    /// Evaluates the error polynomial for the populated row
    /// WARN: `populate` must have been called before.
    pub fn evaluate(
        &mut self,
        row: usize,
        isize: i32,
        circuit_data: &CircuitData<F>,
        acc: &AccumulatorWitness<F>,
        new: &AccumulatorWitness<F>,
    ) -> Option<&[Vec<F>]> {
        self.populate(row, isize, circuit_data, acc, new);

        // exit early if there is a simple selector and it is off
        if let Some(simple_selector_value) = self.simple_selector {
            if !simple_selector_value {
                return None;
            }
        }
        let max_num_evals = self.max_num_evals;
        let mut instance_tmp = self.instance_acc.to_vec();
        let mut advice_tmp = self.advice_acc.to_vec();

        // TODO(@adr1anh): Check whether we are not off-by-one
        // Iterate over all evaluations points X = 0, ..., d-1
        for eval_idx in 0..max_num_evals {
            // After the first iteration, add the contents of the new instance and advice to the tmp buffer
            if eval_idx > 0 {
                for (i_tmp, i_new) in instance_tmp.iter_mut().zip(self.instance_new.iter()) {
                    *i_tmp += i_new;
                }
                for (a_tmp, a_new) in advice_tmp.iter_mut().zip(self.advice_new.iter()) {
                    *a_tmp += a_new;
                }
            }
            // Iterate over each polynomial constraint
            for (poly_idx, (poly, num_evals)) in
                self.gate.polys.iter().zip(&self.num_evals).enumerate()
            {
                // if the degree of this constraint is less than the max degree,
                // we don't need to evaluate it
                if *num_evals > eval_idx {
                    continue;
                }
                // evaluate the j-th constraint G_j at X=d
                self.gate_eval[poly_idx][eval_idx] = poly.evaluate(
                    &|slack_power| self.slack_powers_evals[eval_idx][slack_power],
                    &|constant| constant,
                    &|selector_idx| self.selectors[selector_idx],
                    &|fixed_idx| self.fixed[fixed_idx],
                    &|advice_idx| advice_tmp[advice_idx],
                    &|instance_idx| instance_tmp[instance_idx],
                    &|challenge_idx, challenge_power| {
                        self.challenge_powers_evals[challenge_idx].powers_evals[eval_idx]
                            [challenge_power]
                    },
                    &|negated| -negated,
                    &|sum_a, sum_b| sum_a + sum_b,
                    &|prod_a, prod_b| prod_a * prod_b,
                    &|scaled, v| scaled * v,
                );
            }
        }
        Some(&self.gate_eval)
    }
}

///
#[derive(Clone)]
struct EvaluatedChallenge<F: Field> {
    // powers_evals[X][power] = acc[power] + X * new^power
    pub powers_evals: Vec<Vec<F>>,
}

fn powers_of<F: Field>(v: &F, num_powers: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(num_powers);
    let mut acc = F::ONE;
    powers.push(acc);

    for _i in 1..num_powers {
        acc *= v;
        powers.push(acc);
    }
    powers
}

impl<F: Field> EvaluatedChallenge<F> {
    fn new(acc: &[F], new: &[F], num_evals: usize, max_degree: usize) -> Self {
        debug_assert_eq!(acc.len(), new.len());
        // debug_assert!(
        //     acc.len() <= max_degree,
        //     "number of accumulated challenge powers must be at least `max_degree`"
        // );
        // d = max_degree, D = num_evals
        // [
        //     [1        , acc_1          , ..., acc_{d-1}                ],
        //     [1 +     1, acc_1 +     new, ..., acc_{d-1} +     new^{d-1}],
        //     ...,
        //     [1 + D * 1, acc_1 + D * new, ..., acc_{d-1} + D * new^{d-1}],
        // ]

        let mut powers_evals = Vec::with_capacity(num_evals);
        // set first row for D=0
        debug_assert_eq!(acc[0], F::ONE);
        powers_evals.push(acc.to_vec());

        for eval in 1..max_degree {
            // compute next row by adding `new` to the previous row
            powers_evals.push(
                powers_evals[eval - 1]
                    .iter()
                    .zip(new.iter())
                    .map(|(prev, new)| *prev + new)
                    .collect(),
            );
        }
        EvaluatedChallenge { powers_evals }
    }
}

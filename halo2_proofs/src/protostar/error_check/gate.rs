use std::iter::zip;

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    poly::Rotation,
    protostar::{gate::Gate, keygen::ProvingKey},
};

use super::{Accumulator, ErrorEvaluations};

/// When evaluating a `Gate` over all rows, we fetch the input data from the previous accumulator
/// and the new witness and store it in this struct.
/// This struct can be cloned when paralellizing the evaluation over chunks of rows,
/// so that each chunk operates over independent data.
pub struct GateEvaluationCache<F: Field> {
    pub gate: Gate<F>,
    // Number of evaluations requried to recover each `poly` in `gate`.
    // Equal to `poly.degree() + 1 + num_extra_evaluations`,
    // which is the number of coefficients of the polynomial,
    // plus any additional evaluations required if we want to multiply by another variable
    // later on.
    num_evals: Vec<usize>,
    // Maximum of `num_evals`
    max_num_evals: usize,

    // `challenge_powers_evals[X][j][d]` = (1-X)⋅c'ⱼᵈ + X⋅cⱼᵈ
    // - X = 0, 1, ..., max_num_evals - 1`
    // - j = 0, 1, ..., num_challenges - 1`
    // - d = 0, 1, ..., max_challenge_power[j] - 1`
    challenge_powers_evals: Vec<Vec<Vec<F>>>,

    // Cache for storing the fixed, accumulated, and witness values for evaluating this
    // `gate` at a single row.
    //
    // If all `polys` are multiplied by the same `Selector`, store its value here
    // and allow the evaluation of the gate to be skipped if it is `false`
    simple_selector: Option<bool>,
    // Contains the remaining selectors inside `polys` defined by `gate.queried_selectors`
    selectors: Vec<F>,
    // Values for the row from `circuit_data` at the indices defined by `gate.queried_fixed`
    fixed: Vec<F>,

    // Values for the row from `acc` at the indices defined by `gate.queried_instance`
    instance_acc: Vec<F>,
    // Values of the difference between the rows `new` and `acc` at the indices defined by `gate.queried_instance`
    instance_diff: Vec<F>,
    // Values for the row from `acc` at the indices defined by `gate.queried_advice`
    advice_acc: Vec<F>,
    // Values of the difference between the rows `new` and `acc` at the indices defined by `gate.queried_advice`
    advice_diff: Vec<F>,

    // For each `poly` at index `j` in `gate`,
    // store the evaluations of the error polynomial `e_j(X)`
    // at X = 0, 1, ..., num_evals[j] - 1 in `gate_eval[j][X]`
    pub gate_eval: Vec<ErrorEvaluations<F>>,
}

impl<F: Field> GateEvaluationCache<F> {
    /// Initialize a chache for a given `Gate` for combining an existing accumulator with
    /// the data from a new transcript.
    /// Evaluations of the linear combination of the slack and challenge variables
    /// are recomputed and stored locally to ensure no data races between different caches.
    /// The `evaluate` method can compute additional evaluations defined by `num_extra_evaluations`,
    /// so that the resulting evaluations can be multiplied afterwards, for example by the challenge `y`.
    pub fn new(
        gate: &Gate<F>,
        acc_challenges: &[Vec<F>],
        new_challenges: &[Vec<F>],
        num_extra_evaluations: usize,
    ) -> Self {
        // Each `poly` Gⱼ(X) has degree dⱼ and dⱼ+1 coefficients,
        // therefore we need at least dⱼ+1 evaluations of Gⱼ(X) to recover
        // the coefficients.
        let num_evals: Vec<_> = gate
            .degrees
            .iter()
            .map(|d| d + 1 + num_extra_evaluations)
            .collect();
        // d = maxⱼ{dⱼ}
        let max_poly_degree = *gate.degrees.iter().max().unwrap();
        // maximum number of evaluations over all Gⱼ(X)
        let max_num_evals = max_poly_degree + 1 + num_extra_evaluations;

        // Challenge evaluations: challenge_powers_evals[X][j][power] =
        //  challenges_acc[j][power] + X⋅challenges_new[j]^power
        // for
        //    X     = 0, 1, ..., max_num_evals   - 1
        //    j     = 0, 1, ..., num_challenges  - 1
        //    power = 0, 1, ..., max_poly_degree
        let challenge_powers_evals =
            evaluated_challenge_powers(acc_challenges, new_challenges, max_num_evals);

        // for each polynomial, allocate a buffer for storing all the evaluations
        let gate_eval = num_evals
            .iter()
            .map(|d| ErrorEvaluations::new(*d))
            .collect();

        Self {
            gate: gate.clone(),
            num_evals,
            max_num_evals,
            challenge_powers_evals,
            simple_selector: None,
            selectors: vec![F::ZERO; gate.queried_selectors.len()],
            fixed: vec![F::ZERO; gate.queried_fixed.len()],
            instance_acc: vec![F::ZERO; gate.queried_instance.len()],
            instance_diff: vec![F::ZERO; gate.queried_instance.len()],
            advice_acc: vec![F::ZERO; gate.queried_advice.len()],
            advice_diff: vec![F::ZERO; gate.queried_advice.len()],
            gate_eval,
        }
    }

    /// Fills the local variables buffers with data from the accumulator and new transcript
    fn populate<C>(
        &mut self,
        row: usize,
        isize: i32,
        pk: &ProvingKey<C>,
        acc: &Accumulator<C>,
        new: &Accumulator<C>,
    ) where
        C: CurveAffine<ScalarExt = F>,
    {
        /// Return the index in the polynomial of size `isize` after rotation `rot`.
        fn get_rotation_idx(idx: usize, rot: Rotation, isize: i32) -> usize {
            (((idx as i32) + rot.0).rem_euclid(isize)) as usize
        }

        // Check if the gate is guarded by a simple selector and whether it is active
        if let Some(simple_selector) = self.gate.simple_selector {
            let selector_column = simple_selector.0;
            let value = pk.selectors[selector_column][row];
            self.simple_selector = Some(value);
            // do nothing and return
            if !value {
                return;
            }
        }

        // Fill selectors
        for (i, selector) in self.gate.queried_selectors.iter().enumerate() {
            let selector_column = selector.0;
            let value = pk.selectors[selector_column][row];
            self.selectors[i] = if value { F::ONE } else { F::ZERO };
        }

        // Fill fixed
        for (i, fixed) in self.gate.queried_fixed.iter().enumerate() {
            let fixed_column = fixed.column_index();
            let fixed_row = get_rotation_idx(row, fixed.rotation(), isize);
            self.fixed[i] = pk.fixed[fixed_column][fixed_row];
        }

        // Fill instance
        for (i, instance) in self.gate.queried_instance.iter().enumerate() {
            let instance_column = instance.column_index();
            let instance_row = get_rotation_idx(row, instance.rotation(), isize);
            self.instance_acc[i] =
                acc.instance_transcript.instance_polys[instance_column][instance_row];
            self.instance_diff[i] = new.instance_transcript.instance_polys[instance_column]
                [instance_row]
                - self.instance_acc[i];
        }

        // Fill advice
        for (i, advice) in self.gate.queried_advice.iter().enumerate() {
            let advice_column = advice.column_index();
            let advice_row = get_rotation_idx(row, advice.rotation(), isize);
            self.advice_acc[i] = acc.advice_transcript.advice_polys[advice_column][advice_row];
            self.advice_diff[i] =
                new.advice_transcript.advice_polys[advice_column][advice_row] - self.advice_acc[i];
        }
    }

    /// Evaluates the error polynomial for the populated row.
    /// Returns `None` if the common selector for the gate is false,
    /// otherwise returns a list of vectors containing the evaluations for
    /// each `poly` Gⱼ(X) in `gate`.
    pub fn evaluate<C>(
        &mut self,
        row: usize,
        isize: i32,
        pk: &ProvingKey<C>,
        acc: &Accumulator<C>,
        new: &Accumulator<C>,
    ) -> Option<&[ErrorEvaluations<F>]>
    where
        C: CurveAffine<ScalarExt = F>,
    {
        // Fill the buffers with data from the fixed circuit_data, the previous accumulator
        // and the new witness.
        self.populate(row, isize, pk, acc, new);

        // exit early if there is a simple selector and it is off
        if let Some(simple_selector_value) = self.simple_selector {
            if !simple_selector_value {
                return None;
            }
        }
        let max_num_evals = self.max_num_evals;
        // Use the `*_acc` buffers to store the linear combination evaluations.
        // In the first iteration with X=0, it is already equal to `*_acc`
        let instance_tmp = &mut self.instance_acc;
        let advice_tmp = &mut self.advice_acc;

        // Iterate over all evaluations points X = 0, ..., max_num_evals-1
        for eval_idx in 0..max_num_evals {
            // After the first iteration, add the contents of the new instance and advice to the tmp buffer
            if eval_idx > 0 {
                for (i_tmp, i_diff) in zip(instance_tmp.iter_mut(), self.instance_diff.iter()) {
                    *i_tmp += i_diff;
                }
                for (a_tmp, a_diff) in zip(advice_tmp.iter_mut(), self.advice_diff.iter()) {
                    *a_tmp += a_diff;
                }
            }
            // Iterate over each polynomial constraint Gⱼ, along with its required number of evaluations
            for (poly_idx, (poly, num_evals)) in
                zip(self.gate.polys.iter(), self.num_evals.iter()).enumerate()
            {
                // If the eval_idx X is larger than the required number of evaluations for the current poly,
                // we don't evaluate it and continue to the next poly.
                if eval_idx > *num_evals {
                    continue;
                }
                // evaluate the j-th constraint G_j at X = eval_idx
                let e = poly.evaluate(
                    &|constant| constant,
                    &|selector_idx| self.selectors[selector_idx],
                    &|fixed_idx| self.fixed[fixed_idx],
                    &|advice_idx| advice_tmp[advice_idx],
                    &|instance_idx| instance_tmp[instance_idx],
                    &|challenge_idx, challenge_power| {
                        self.challenge_powers_evals[eval_idx][challenge_idx][challenge_power - 1]
                    },
                    &|negated| -negated,
                    &|sum_a, sum_b| sum_a + sum_b,
                    &|prod_a, prod_b| prod_a * prod_b,
                    &|scaled, v| scaled * v,
                );
                self.gate_eval[poly_idx].evals[eval_idx] = e;
            }
        }
        Some(&self.gate_eval)
    }

    /// Returns a zero-initialized error polynomial for storing all evaluations of
    /// the polynomials for this gate.
    pub fn empty_error_polynomials(&self) -> Vec<ErrorEvaluations<F>> {
        self.num_evals
            .iter()
            .map(|num_eval| ErrorEvaluations::new(*num_eval))
            .collect()
    }
}

/// Given two lists of challenge powers from an existing accumulator and a new circuit execution,
/// compute the evaluations at X = 0, 1, ..., num_evals-1 of "challenges_acc + X⋅challenges_new".
/// The returned vector satisfies:
/// challenge_powers_evals[X][power][j] = (1-X)⋅challenges_acc[power][j] + X⋅challenges_new[j]^power
/// for
///    X     = 0, 1, ..., num_evals       - 1
///    j     = 0, 1, ..., num_challenges  - 1
///    power = 0, 1, ...,
pub fn evaluated_challenge_powers<F: Field>(
    challenges_acc: &[Vec<F>],
    challenges_new: &[Vec<F>],
    num_evals: usize,
) -> Vec<Vec<Vec<F>>> {
    debug_assert_eq!(
        challenges_acc.len(),
        challenges_new.len(),
        "number of challenges in both accumulators must be the same"
    );

    let challenges_diff: Vec<Vec<_>> = zip(challenges_acc.iter(), challenges_new.iter())
        .map(|(c_acc, c_new)| {
            zip(c_acc.iter(), c_new.iter())
                .map(|(c_acc, c_new)| *c_new - c_acc)
                .collect()
        })
        .collect();

    let mut challenge_powers_evals = Vec::with_capacity(num_evals);

    // Add the evaluation of the challenge variables at X=0,
    // corresponding to challenges_acc
    challenge_powers_evals.push(challenges_acc.to_vec());

    // Iterate over eval = X = 1, ..., num_evals - 1
    for eval in 1..num_evals {
        // Get previous evalutions at X-1
        let prev_evals = &challenge_powers_evals[eval - 1];
        // compute next row by adding `new` to the previous row
        let curr_eval = zip(prev_evals.iter(), challenges_diff.iter())
            .map(|(prev_powers_j, diff_powers_j)| {
                // `prev_powers_j` and `new_powers_j` are vectors of the powers of the challenge j
                // this loop adds them up to get the evaluation
                zip(prev_powers_j.iter(), diff_powers_j.iter())
                    .map(|(prev, diff)| *prev + diff)
                    .collect()
            })
            .collect();

        challenge_powers_evals.push(curr_eval);
    }

    challenge_powers_evals
}

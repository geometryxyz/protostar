use core::num;
use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Add, Deref, DerefMut, Index, IndexMut, RangeFrom, RangeFull, Sub},
};

use crate::{
    arithmetic::{field_integers, lagrange_interpolate, powers},
    poly::{commitment::Blind, empty_lagrange, LagrangeCoeff, Polynomial, Rotation},
};
use ff::Field;
use halo2curves::CurveAffine;

use super::{gate::Gate, keygen::ProvingKey};

pub struct Accumulator<F: Field> {
    instance: AccumulatorInstance<F>,
    witness: AccumulatorWitness<F>,
}

/// The "instance" portion of a Protostar accumulator
/// TODO(@adr1anh): Implement `Add` so we can combine two accumulators.
#[derive(Clone)]
pub struct AccumulatorInstance<F: Field> {
    // Variable cⱼᵈ, where `j` is the challenge index,
    // and `d` is its power in the polynomial constraint.
    // cⱼᵈ = `challenges[j][d]`
    // The number of "powers" `d` of each challenge is determined by
    // looking at the maximum `d` over all `polys` in each `gate`.
    // For each `j`, we have `challenges[j].len() = d+1`,
    // since we include `challenge[j][0] = 1`
    challenges: Vec<Vec<F>>,
    // Public inputs and outputs of the circuit
    // The shape of `instance` is determined by `CircuitData::num_instance_rows`
    instance: Vec<Polynomial<F, LagrangeCoeff>>,

    // initial challenge for combining all rows
    beta: F,
    beta_sqrt: F,

    // challenges for randomly combining all gate polys
    pub ys: Vec<F>,
    // field error eval
    error: F,
}

#[derive(Clone)]
/// The "witness" portion of the accumulator containing the verifier messages
pub struct AccumulatorWitness<F: Field> {
    advice: Vec<Polynomial<F, LagrangeCoeff>>,
    advice_blinds: Vec<Blind<F>>,
    betas: Vec<F>,
    betas_sqrt: Vec<F>,
}

impl<F: Field> Accumulator<F> {
    pub fn new(instance: AccumulatorInstance<F>, witness: AccumulatorWitness<F>) -> Self {
        Self { instance, witness }
    }

    pub fn instance(&self) -> &AccumulatorInstance<F> {
        &self.instance
    }

    pub fn witness(&self) -> &AccumulatorWitness<F> {
        &self.witness
    }

    pub fn fold(mut self, pk: &ProvingKey<F>, new: Accumulator<F>) -> Self {
        // 2 for beta, 1 for y
        let num_extra_evaluations = 3;
        let mut gate_caches: Vec<_> = pk
            .gates()
            .iter()
            .map(|gate| {
                GateEvaluationCache::new(
                    gate,
                    self.instance(),
                    new.instance(),
                    num_extra_evaluations,
                )
            })
            .collect();

        let num_rows_i = pk.num_rows() as i32;
        let num_evals = pk.max_degree() + 1 + num_extra_evaluations;

        let mut beta_iterator = BetaIterator::new(
            self.witness().betas(),
            self.witness().betas_sqrt(),
            new.witness().betas(),
            new.witness().betas_sqrt(),
            pk.log2_sqrt_num_rows(),
            num_evals,
        );

        // Store the sum of the
        // eval_sums[gate_idx][constraint_idx][eval_idx]
        let mut error_polys: Vec<_> = gate_caches
            .iter()
            .map(|gate_cache| gate_cache.empty_error_polynomials())
            .collect();

        for row_idx in 0..pk.num_rows() {
            // Get the next evaluation of ((1-X) * acc.b1 + X * acc.b2)*((1-X) * new.b1 + X * new.b2)
            let beta_evals = beta_iterator.evals_at_row(row_idx);
            for (gate_idx, gate_cache) in gate_caches.iter_mut().enumerate() {
                let evals =
                    gate_cache.evaluate(row_idx, num_rows_i, &pk, self.witness(), new.witness());
                if let Some(evals) = evals {
                    // For each constraint in the current gate, scale it by the beta vector
                    // and add it to the corresponding error polynomial accumulator `error_polys`
                    for (acc_error_poly, new_error_poly) in
                        zip(error_polys[gate_idx].iter_mut(), evals.iter())
                    {
                        // Multiply each poly by b1*b2 and add it to the corresponding accumulator
                        acc_error_poly.add_multiplied(new_error_poly, beta_evals);
                    }
                }
            }
        }
        let final_poly = {
            let mut final_poly = vec![F::ZERO; num_evals];

            // TODO(@adr1anh): The inner loop can be optimized since `to_coefficients` does a lot of common preprocessing.
            for (evals, (y_acc, y_new)) in error_polys
                .iter_mut()
                .flat_map(|polys| polys.iter_mut())
                .zip(zip(self.instance().ys.iter(), new.instance().ys.iter()))
            {
                // Multiply e_j(X) by y_j
                evals.multiply_by_challenge_var(*y_acc, *y_new);
                // Obtain the coefficients of the polynomial
                let coeffs = evals.to_coefficients();
                // Add the coefficients to the final accumulator
                for (final_coeff, coeff) in zip(final_poly.iter_mut(), coeffs.iter()) {
                    *final_coeff += coeff;
                }
            }
            final_poly
        };
        let expected_error_acc: F = final_poly[0];
        let expected_error_new: F = final_poly.iter().sum();

        assert_eq!(expected_error_acc, self.instance.error);
        assert_eq!(expected_error_new, new.instance.error);
        // TODO(@adr1anh): Commit to `final_poly` but ignore first and last coefficient
        // The first one is equal to the error of `self`
        // The last one should be 0 due to homogeneity.

        // TODO(@adr1anh): Get alpha challenge from transcript
        let alpha = F::ONE;

        self = self.linear_combination(new, alpha, &final_poly);
        self
    }

    // TODO(@adr1anh) no discard
    fn linear_combination(mut self, new: Accumulator<F>, alpha: F, final_poly: &[F]) -> Self {
        self.instance = self
            .instance
            .linear_combination(new.instance, alpha, final_poly);
        self.witness = self.witness.linear_combination(new.witness, alpha);
        self
    }
}

impl<F: Field> AccumulatorInstance<F> {
    /// Generate an empty accumulator with all values set to zero.
    pub fn new_empty(pk: &ProvingKey<F>) -> Self {
        // For each challenge `j`, with max degee `power`,
        // we need `power + 1` elements
        let challenges: Vec<_> = pk
            .max_challenge_powers()
            .iter()
            .map(|power| vec![F::ZERO; *power + 1])
            .collect();

        let instance = pk
            .num_instance_rows()
            .iter()
            .map(|num_rows| empty_lagrange(*num_rows))
            .collect();

        let ys = vec![F::ZERO; pk.num_constraints()];

        Self {
            challenges,
            instance,
            beta: F::ZERO,
            beta_sqrt: F::ZERO,
            ys,
            error: F::ZERO,
        }
    }

    // Create accumulator instance given data obtained by generating the witness.
    pub fn new(
        pk: &ProvingKey<F>,
        challenges: Vec<F>,
        instance: Vec<Polynomial<F, LagrangeCoeff>>,
        beta: F,
        y: F,
    ) -> Self {
        #[cfg(feature = "sanity-checks")]
        {
            assert_eq!(
                challenges.len(),
                pk.num_challenges(),
                "invalid number of challenges supplied"
            );
            assert_eq!(
                instance.len(),
                pk.num_instance_columns(),
                "invalid number of instance columns supplied"
            );

            for (i, (instance_len, expected_len)) in zip(
                instance.iter().map(|instance| instance.len()),
                pk.num_instance_rows().iter(),
            )
            .enumerate()
            {
                assert_eq!(
                    instance_len, *expected_len,
                    "invalid size for instance column {i}"
                );
            }
        }

        let challenge_powers: Vec<_> = zip(challenges.iter(), pk.max_challenge_powers().iter())
            .map(|(c, power)| powers(*c).take(power + 1).collect())
            .collect();
        let k_sqrt = pk.log2_sqrt_num_rows();
        let n_sqrt = 1 << k_sqrt;
        let beta_sqrt = beta.pow_vartime([n_sqrt as u64]);

        let ys: Vec<_> = powers(y).take(pk.num_constraints()).collect();
        Self {
            challenges: challenge_powers,
            instance,
            beta,
            beta_sqrt,
            ys,
            error: F::ZERO,
        }
    }

    // TODO(@adr1anh) no discard
    fn linear_combination(
        mut self,
        new: AccumulatorInstance<F>,
        alpha: F,
        final_poly: &[F],
    ) -> Self {
        self.beta += alpha * (new.beta - self.beta);
        self.beta_sqrt += alpha * (new.beta_sqrt - self.beta_sqrt);

        // horner eval of the polynomial e(X) in alpha
        assert_eq!(final_poly[0], self.error);
        self.error = F::ZERO;
        for coeff in final_poly.iter().rev() {
            self.error *= alpha;
            self.error += coeff;
        }

        for (acc_challenge, new_challenge) in zip(
            self.challenges.iter_mut().flat_map(|c| c.iter_mut()),
            new.challenges.iter().flat_map(|c| c.iter()),
        ) {
            *acc_challenge += alpha * (*new_challenge - *acc_challenge);
        }

        for (acc_instance, new_instance) in zip(self.instance.iter_mut(), new.instance.iter()) {
            *acc_instance =
                Polynomial::boolean_linear_combination(acc_instance, new_instance, alpha);
        }

        for (acc_y, new_y) in zip(self.ys.iter_mut(), new.ys.iter()) {
            *acc_y += alpha * (*new_y - *acc_y);
        }

        self
    }
}

impl<F: Field> AccumulatorWitness<F> {
    // Initialized a new empty `AccumulatorWitness` for a given circuit,
    // with all values set to zero.
    pub fn new_empty(pk: &ProvingKey<F>) -> Self {
        let advice = vec![empty_lagrange(pk.num_rows()); pk.num_advice_columns()];
        let k_sqrt = pk.log2_sqrt_num_rows();
        let n_sqrt = 1 << k_sqrt;
        // TODO(@adr1anh): replace with actual lengths
        // let advice = pk
        //     .circuit_data
        //     .num_advice_rows
        //     .iter()
        //     .map(|advice_len| empty_lagrange(*advice_len))
        //     .collect();
        let betas = vec![F::ONE; n_sqrt];
        let betas_sqrt = vec![F::ONE; n_sqrt];
        Self {
            advice,
            advice_blinds: vec![Blind::default(); pk.num_advice_columns()],
            betas,
            betas_sqrt,
        }
    }

    // Initializes a new `AccumulatorWitness` given a list of advice columns generated by the prover.
    pub fn new(
        pk: &ProvingKey<F>,
        advice: Vec<Polynomial<F, LagrangeCoeff>>,
        advice_blinds: Vec<Blind<F>>,
        betas: Vec<F>,
        betas_sqrt: Vec<F>,
    ) -> Self {
        let k_sqrt = pk.log2_sqrt_num_rows();
        let n_sqrt = 1 << k_sqrt;
        #[cfg(feature = "sanity-checks")]
        {
            assert_eq!(
                advice.len(),
                advice_blinds.len(),
                "number of advice columns and blinds must match"
            );
            assert_eq!(betas.len(), n_sqrt, "betas must be of size sqrt(num_rows)");
            assert_eq!(
                betas_sqrt.len(),
                n_sqrt,
                "betas_sqrt must be of size sqrt(num_rows)"
            );
            // TODO(@adr1anh): check the lengths of `advice`
        }

        Self {
            advice,
            advice_blinds,
            betas,
            betas_sqrt,
        }
    }

    pub fn betas(&self) -> &[F] {
        &self.betas
    }
    pub fn betas_sqrt(&self) -> &[F] {
        &self.betas_sqrt
    }

    // TODO(@adr1anh) no discard
    fn linear_combination(mut self, new: AccumulatorWitness<F>, alpha: F) -> Self {
        // TODO(@adr1anh): sanity checks for same size
        for (acc_advice, new_advice) in zip(self.advice.iter_mut(), new.advice.iter()) {
            *acc_advice = Polynomial::boolean_linear_combination(acc_advice, new_advice, alpha);
        }
        for (acc_advice_blind, new_advice_blind) in
            zip(self.advice_blinds.iter_mut(), new.advice_blinds.iter())
        {
            *acc_advice_blind += *new_advice_blind * alpha;
        }

        for (acc_beta0, new_beta0) in zip(self.betas.iter_mut(), new.betas.iter()) {
            *acc_beta0 += alpha * new_beta0;
        }
        for (acc_beta1, new_beta1) in zip(self.betas_sqrt.iter_mut(), new.betas_sqrt.iter()) {
            *acc_beta1 += alpha * new_beta1;
        }
        self
    }
}

#[derive(Clone)]
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
        gate_accumulator: &AccumulatorInstance<F>,
        gate_transcript: &AccumulatorInstance<F>,
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
        let challenge_powers_evals = evaluated_challenge_powers(
            &gate_accumulator.challenges,
            &gate_transcript.challenges,
            max_num_evals,
        );

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
    fn populate(
        &mut self,
        row: usize,
        isize: i32,
        circuit_data: &ProvingKey<F>,
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
            self.instance_diff[i] =
                new.advice[instance_column][instance_row] - self.instance_acc[i];
        }

        // Fill advice
        for (i, advice) in self.gate.queried_advice.iter().enumerate() {
            let advice_column = advice.column_index();
            let advice_row = get_rotation_idx(row, advice.rotation(), isize);
            self.advice_acc[i] = acc.advice[advice_column][advice_row];
            self.advice_diff[i] = new.advice[advice_column][advice_row] - self.advice_acc[i];
        }
    }

    /// Evaluates the error polynomial for the populated row.
    /// Returns `None` if the common selector for the gate is false,
    /// otherwise returns a list of vectors containing the evaluations for
    /// each `poly` Gⱼ(X) in `gate`.
    pub fn evaluate(
        &mut self,
        row: usize,
        isize: i32,
        pk: &ProvingKey<F>,
        acc: &AccumulatorWitness<F>,
        new: &AccumulatorWitness<F>,
    ) -> Option<&[ErrorEvaluations<F>]> {
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
                        self.challenge_powers_evals[eval_idx][challenge_idx][challenge_power]
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
/// challenge_powers_evals[X][j][power] = challenges_acc[j][power] + X⋅challenges_new[j]^power
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

#[derive(Clone)]
// This could be more general, i.e. a
// - the evaluation of a row
// - eval of a challenge
pub struct ErrorEvaluations<F: Field> {
    evals: Vec<F>,
}

impl<F: Field> ErrorEvaluations<F> {
    pub fn new(num_evals: usize) -> Self {
        Self {
            evals: vec![F::ZERO; num_evals],
        }
    }

    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    pub fn multiply_by_challenge_var(&mut self, c_acc: F, c_new: F) {
        let c_evals_iter = boolean_evaluations(c_acc, c_new);
        for (e, c) in zip(self.evals.iter_mut(), c_evals_iter) {
            *e *= c;
        }
    }

    pub fn add_multiplied(&mut self, other: &ErrorEvaluations<F>, scalar_evals: &[F]) {
        for (s, (o, m)) in self
            .evals
            .iter_mut()
            .zip(zip(other.evals.iter(), scalar_evals.iter()))
        {
            *s += *m * o;
        }
    }

    pub fn to_coefficients(&self) -> Vec<F> {
        let points: Vec<_> = field_integers().take(self.evals.len()).collect();
        lagrange_interpolate(&points, &self.evals)
    }
}

/// For a linear polynomial p(X) such that p(0) = eval0, p(1) = eval1,
/// return an iterator yielding the evaluations p(j) for j=0,1,...
fn boolean_evaluations<F: Field>(eval0: F, eval1: F) -> impl Iterator<Item = F> {
    let linear = eval1 - eval0;
    std::iter::successors(Some(eval0), move |acc| Some(linear + acc))
}

/// Non-comforming iterator yielding evaluations of
/// (acc.β₀ + X⋅new.β₀)⋅(acc.β₁ + X⋅new.β₁)
struct BetaIterator<'a, 'b, F: Field> {
    betas0_acc: &'a [F],
    betas1_acc: &'a [F],
    betas0_new: &'b [F],
    betas1_new: &'b [F],

    // log2(n^{1/2})
    k_sqrt: u32,
    // 2^k_sqrt - 1
    mask: usize,

    evals: Vec<F>,
}

impl<'a, 'b, F: Field> BetaIterator<'a, 'b, F> {
    fn new(
        betas0_acc: &'a [F],
        betas1_acc: &'a [F],
        betas0_new: &'b [F],
        betas1_new: &'b [F],
        k_sqrt: u32,
        num_evals: usize,
    ) -> Self {
        let n_sqrt = 1 << k_sqrt;
        let mask = n_sqrt - 1;
        Self {
            betas0_acc,
            betas1_acc,
            betas0_new,
            betas1_new,
            k_sqrt,
            mask,
            evals: vec![F::ZERO; num_evals],
        }
    }

    fn evals_at_row(&mut self, row_index: usize) -> &[F] {
        let i0 = row_index & self.mask;
        let i1 = row_index >> self.k_sqrt;
        let mut beta0 = self.betas0_acc[i0];
        let mut beta1 = self.betas1_acc[i1];
        let beta0_new = self.betas0_new[i0];
        let beta1_new = self.betas1_new[i1];

        self.evals[0] = beta0 * beta1;
        for i in 1..self.evals.len() {
            beta0 += beta0_new;
            beta1 += beta1_new;
            self.evals[i] = beta0 * beta1;
        }

        self.evals.as_slice()
    }
}

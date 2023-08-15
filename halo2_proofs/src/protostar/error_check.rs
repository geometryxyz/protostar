use core::num;
use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Add, Deref, DerefMut, Index, IndexMut, RangeFrom, RangeFull, Sub},
};

mod gate;
mod row;

use crate::{
    arithmetic::{field_integers, parallelize, powers},
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        empty_lagrange, LagrangeCoeff, Polynomial, Rotation,
    },
    protostar::error_check::gate::GateEvaluator,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::{BatchInvert, Field};
use group::Curve;
use halo2curves::CurveAffine;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use super::{
    keygen::ProvingKey,
    transcript::{
        advice::AdviceTranscript, compressed_verifier::CompressedVerifierTranscript,
        instance::InstanceTranscript, lookup::LookupTranscipt,
    },
};

/// Each constraint error polynomial must be multiplied by the challenge(s) beta.
/// This allows us to compute the degree and therefore number of evaluations of e(X).
const BETA_POLY_DEGREE: usize = 1;
/// As an optimization, we can store the evaluations [e₀(X), …, eₘ₋₁(X)] in the accumulator.
/// This allows us to skip the evaluations in X ∈ {0,1}.
const STARTING_EVAL_IDX: usize = 2;

/// An `Accumulator` contains the entirety of the IOP transcript,
/// including commitments and verifier challenges.
#[derive(Debug, Clone, PartialEq)]
pub struct Accumulator<C: CurveAffine> {
    instance_transcript: InstanceTranscript<C>,
    advice_transcript: AdviceTranscript<C>,
    lookup_transcript: Option<LookupTranscipt<C>>,
    compressed_verifier_transcript: CompressedVerifierTranscript<C>,

    // Powers of a challenge y for taking a random linear-combination of all constraints.
    ys: Vec<C::Scalar>,

    // For each constraint of degree > 1, we cache its error polynomial evaluation here
    // so we can interpolate all of them individually.
    constraint_errors: Vec<C::Scalar>,

    // Error value for all constraints
    error: C::Scalar,
}

impl<C: CurveAffine> Accumulator<C> {
    // Create a new `Accumulator` from the different transcripts.
    pub fn new(
        pk: &ProvingKey<C>,
        instance_transcript: InstanceTranscript<C>,
        advice_transcript: AdviceTranscript<C>,
        lookup_transcript: Option<LookupTranscipt<C>>,
        compressed_verifier_transcript: CompressedVerifierTranscript<C>,
        y: C::Scalar,
    ) -> Self {
        let num_constraints = pk.num_folding_constraints();

        // If the initial transcript is correct, then the constraint polynomials should all
        // evaluate to 0.
        let constraint_errors = vec![C::Scalar::ZERO; num_constraints];

        // Compute powers of the challenge y to do a random-linear combination of the
        // constraint error polynomials.
        let ys: Vec<_> = powers(y).skip(1).take(num_constraints).collect();

        Self {
            instance_transcript,
            advice_transcript,
            lookup_transcript,
            compressed_verifier_transcript,
            ys,
            constraint_errors,
            error: C::Scalar::ZERO,
        }
    }

    /// Fold another accumulator in `self`.
    pub fn fold<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        &mut self,
        pk: &ProvingKey<C>,
        new: Accumulator<C>,
        transcript: &mut T,
    ) {
        /*
        Constraints are grouped by gate. Most constraint groups will have a common simple selector which
        selects whether the constraints are active. We can skip the evaluation of the entire group by first checking
        whether the selector is switched off.

        The constraints in one gate will all be applied to the same subset of columns,
        so we only fetch the entries in the row for the columns of the active constraints.
        We index the gates by `gate_idx` and the constraints in gate `gate_idx` by `gate_constraint_idx`.
        After evaluating all error polynomials for each gate, we consider the flattened list of constraints,
        and index them by `constraint_idx`
        */
        let gate_selectors = pk.simple_selectors();

        /*
        Compute the number of constraints in each gate as well as the total number of constraints in all gates.
        */
        let num_constraints_per_gate: Vec<_> = pk
            .folding_constraints()
            .iter()
            .map(|polys| polys.len())
            .collect();
        let num_constraints = num_constraints_per_gate.iter().sum();

        /*
        For each gate at index `gate_idx` with polynomials G₀, …, Gₘ₋₁ and degrees d₀, …, dₘ₋₁,
        we evaluate at each row i the polynomials e₀,ᵢ(X), …, eₘ₋₁,ᵢ(X), where
            eⱼ,ᵢ(X) = βᵢ(X)⋅Gⱼ((1−X)⋅acc[i] + X⋅new[i]),
        where we use the shorthand notation `acc[i]` and `new[i]` to denote the row-vector of values from the
        accumulators `acc` and `new` respectively.

        Each eⱼ,ᵢ(X) has degree dⱼ + `BETA_POLY_DEGREE`, so we need (dⱼ + `BETA_POLY_DEGREE` + 1) evaluations in order to
        interpolate the full polynomial.
        */
        let gates_error_polys_num_evals: Vec<Vec<usize>> = pk
            .folding_constraints()
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| poly.folding_degree() + BETA_POLY_DEGREE + 1)
                    .collect()
            })
            .collect();

        /*
        For each gate, we allocate a buffer for storing running sum of the evaluation e₀(X), …, eₘ₋₁(X).
        The polynomial eⱼ(X) is given by
            eⱼ(X) = ∑ᵢ eⱼ,ᵢ(X)

        For each polynomial eⱼ(X), we define its evaluation domain as Dⱼ = {0, 1, …, dⱼ + BETA_POLY_DEGREE + 1},
        and we denote by eⱼ(Dⱼ) = { p(i) | i ∈ Dⱼ } the list of evaluations of eⱼ(X) over Dⱼ.

        Therefore, for each gate, `gates_error_polys_evals[gate_idx]` corresponds to the list
            [e₀(D₀), …, eₘ₋₁(Dₘ₋₁)]

        As an optimization, we observe that we can store in the accumulators the evaluations
            [e₀(0), …, eₘ₋₁(0)] in `acc`, and
            [e₀(1), …, eₘ₋₁(1)] in `new`
        Therefore, we only actually evaluate the polynomials on the restricted sets Dⱼ' = Dⱼ \ {0,1}.
        */
        let mut gates_error_polys_evals: Vec<Vec<_>> = gates_error_polys_num_evals
            .iter()
            .map(|error_polys_num_evals| {
                error_polys_num_evals
                    .iter()
                    .map(|num_evals| vec![C::Scalar::ZERO; *num_evals])
                    .collect()
            })
            .collect();

        /*
        For gate at index `gate_idx` and each row i of the accumulator, we temporarily store the list of evaluations
            [e₀,ᵢ(D₀), …, eₘ₋₁,ᵢ(Dₘ₋₁)]
        in `row_gates_error_polys_evals[gate_idx]`
        */
        let mut row_gates_error_polys_evals = gates_error_polys_evals.clone();

        /*
        Compute the maximum number of evaluations over all error polynomials over all gates.
        This defines the maximal evaluation domain D = {0, 1, …, dₘₐₓ + BETA_POLY_DEGREE + 1}
        This will allow us to compute the evaluations βᵢ(D)
        */
        let max_num_evals = *gates_error_polys_num_evals
            .iter()
            .flat_map(|gates_num_evals| gates_num_evals.iter())
            .max()
            .unwrap();

        /*
        A `GateEvaluator` pre-processes a gate's polynomials for more efficient evaluation.
        In particular, it collects all column queries and allocates buffers where the corresponding
        values can be stored. Each polynomial in the gate is evaluated several times,
        so it is advantageous to only fetch the values from the columns once.
         */
        let mut gate_evs: Vec<_> = pk
            .folding_constraints()
            .iter()
            .zip(gates_error_polys_num_evals.iter())
            .map(|(polys, num_evals)| {
                let max_num_evals = num_evals.iter().max().unwrap();
                GateEvaluator::new(
                    polys,
                    &self.advice_transcript.challenges,
                    &new.advice_transcript.challenges,
                    *max_num_evals,
                )
            })
            .collect();

        for row_idx in 0..pk.num_usable_rows() {
            // Get the next evaluation βᵢ(D) of βᵢ(X) = ((1-X)⋅acc.βᵢ + X⋅acc.βᵢ)
            let beta_acc = self.compressed_verifier_transcript.beta_poly()[row_idx];
            let beta_new = new.compressed_verifier_transcript.beta_poly()[row_idx];
            let beta_poly_evals: Vec<_> = boolean_evaluations(beta_acc, beta_new)
                .take(max_num_evals)
                .collect();

            for (gate_idx, num_gate_constraints) in num_constraints_per_gate.iter().enumerate() {
                // Return early if the selector for the gate is off at this row.
                if let Some(selector) = gate_selectors[gate_idx] {
                    if !pk.selectors[selector.index()][row_idx] {
                        continue;
                    }
                }
                let gate_ev = &mut gate_evs[gate_idx];

                // Evaluate [e₀,ᵢ(D₀), …, eₘ₋₁,ᵢ(Dₘ₋₁)] into `row_error_polys_evals`
                //
                // As an optimization, we skip the evaluations at X ∈ {0,1}
                let row_error_polys_evals = &mut row_gates_error_polys_evals[gate_idx];
                gate_ev.evaluate_all_from(
                    row_error_polys_evals,
                    STARTING_EVAL_IDX,
                    row_idx,
                    &pk.selectors,
                    &pk.fixed,
                    &self.instance_transcript.instance_polys,
                    &new.instance_transcript.instance_polys,
                    &self.advice_transcript.advice_polys,
                    &new.advice_transcript.advice_polys,
                );

                // Get the running sums for [e₀(D₀), …, eₘ₋₁(Dₘ₋₁)] for the current gate
                // to which we want to add the row's error polynomials evaluations.
                let error_polys_evals = &mut gates_error_polys_evals[gate_idx];
                for j in 0..*num_gate_constraints {
                    // eⱼ(Dⱼ)
                    let error_poly_evals = &mut error_polys_evals[j];
                    // eⱼ,ᵢ(Dⱼ)
                    let row_error_poly_evals = &row_error_polys_evals[j];

                    /*
                    Update the gate error polynomial evaluations
                    eⱼ(Dⱼ) += βᵢ(Dⱼ) ⋅ eⱼ,ᵢ(Dⱼ)

                    NOTE: `beta_poly_evals` is most-likely longer than the other iterators,
                    but `zip` will only produce as many elements as the shortest of the iterators.
                    */
                    for (error_poly_eval, (row_error_poly_eval, beta_eval)) in error_poly_evals
                        .iter_mut()
                        .zip(zip(row_error_poly_evals.iter(), beta_poly_evals.iter()))
                        .skip(STARTING_EVAL_IDX)
                    {
                        *error_poly_eval += *beta_eval * row_error_poly_eval;
                    }
                }
            }
        }

        /*
        Now that we have evaluated all gates, we no longer need to keep track of the nested structure.
        We flatten `gates_error_polys_evals` into `error_polys_evals`, and let `m = num_constraints`.
        The result is the list of error polynomial evaluations e₀(D₀), …, eₘ₋₁(Dₘ₋₁)
        */
        let error_polys_evals: Vec<_> = {
            // First collect all gate error polynomial's evaluations e₀(D₀'), …, eₘ₋₁(Dₘ₋₁')
            let mut error_polys_evals: Vec<_> = gates_error_polys_evals
                .into_iter()
                .flat_map(|poly| poly.into_iter())
                .collect();

            // Sanity checks
            debug_assert_eq!(error_polys_evals.len(), num_constraints);
            debug_assert_eq!(self.constraint_errors.len(), num_constraints);
            debug_assert_eq!(new.constraint_errors.len(), num_constraints);

            // Re-insert evaluations at X = 0, 1
            for (j, error_poly_evals) in error_polys_evals.iter_mut().enumerate() {
                // Get eⱼ(0), eⱼ(1) from the accumulators `self` and `new`
                error_poly_evals[0] = self.constraint_errors[j];
                error_poly_evals[1] = new.constraint_errors[j];
            }
            error_polys_evals
        };

        /*
        Convert polynomials evaluations e₀(D₀), …, eₘ₋₁(Dₘ₋₁)
        into their coefficient representation  e₀(X), …, eₘ₋₁(X)
        */
        let error_polys = batch_lagrange_interpolate_integers(&error_polys_evals);

        /*
        For linear independence of all error polynomials, we multiply each polynomial eⱼ(X) by
          yⱼ(X) = (1−X)⋅acc.yⱼ + X⋅new.yⱼ = acc.yⱼ + X⋅(new.yⱼ - acc.yⱼ)
        We then compute `final_error_poly` as e(X) = ∑ⱼ yⱼ(X)⋅eⱼ(X), which has degree
          d = `max_degree` + NUM_EXTRA_EVALUATIONS (for βᵢ(X)) + 1 (for yⱼ(X))
        We add 1 for the length.
        */
        let final_error_poly_len = error_polys.iter().map(|poly| poly.len()).max().unwrap() + 1;
        let final_error_poly = {
            debug_assert_eq!(self.ys.len(), num_constraints);
            debug_assert_eq!(new.ys.len(), num_constraints);
            error_polys
                .iter()
                .enumerate()
                // Map each eⱼ(X) to yⱼ(X)⋅eⱼ(X)
                .map(|(j, error_poly)| {
                    let acc_y = self.ys[j];
                    let new_y = new.ys[j];
                    // yⱼ(X) = acc.yⱼ + X⋅(new.yⱼ - acc.yⱼ)
                    let y_poly = [acc_y, new_y - acc_y];

                    let mut final_error_poly = vec![C::Scalar::ZERO; error_poly.len() + 1];

                    for (i, a_i) in error_poly.iter().enumerate() {
                        final_error_poly[i] += y_poly[0] * a_i;
                        final_error_poly[i + 1] += y_poly[1] * a_i;
                    }
                    final_error_poly
                })
                // Compute sum e(X) = ∑ⱼ yⱼ(X)⋅eⱼ(X)
                .fold(
                    vec![C::Scalar::ZERO; final_error_poly_len],
                    |mut final_error_poly, final_error_poly_j| {
                        debug_assert!(final_error_poly_j.len() <= final_error_poly_len);

                        for (i, poly_j_coeff_i) in final_error_poly_j.iter().enumerate() {
                            final_error_poly[i] += poly_j_coeff_i;
                        }
                        final_error_poly
                    },
                )
        };

        /*
        Sanity check for ensuring that the accumulators have the expected error evaluations.
         e(0) = e₀
         e(1) = ∑ᵢ eᵢ
        */
        let final_error_poly_eval_0: C::Scalar = final_error_poly[0];
        let final_error_poly_eval_1: C::Scalar = final_error_poly.iter().sum();

        debug_assert_eq!(final_error_poly_eval_0, self.error);
        debug_assert_eq!(final_error_poly_eval_1, new.error);

        /*
        Compute and commit to quotient

                 e(X) - (1-X)⋅e(0) - X⋅e(1)
        e'(X) = ----------------------------
                           (X-1)X

        Since the verifier already knows e(0), e(1),
        we can send this polynomial instead and save 2 hashes.
        The verifier evaluates e(X) as
          e(X) = (1-X)X⋅e'(X) + (1-X)⋅e(0) + X⋅e(1)
        */
        let quotient_final_error_poly = {
            let mut pre_quotient_poly = final_error_poly.clone();
            // Subtract (1-X)⋅acc.error + X⋅new.error = acc.error + X⋅(new.error - acc.error)
            pre_quotient_poly[0] -= self.error;
            pre_quotient_poly[1] -= new.error - self.error;

            quotient_by_boolean_vanishing(&pre_quotient_poly)
        };
        for coef in quotient_final_error_poly.iter() {
            let _ = transcript.write_scalar(*coef);
        }

        // Get alpha challenge
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();

        // Cache the constraint errors eⱼ(α), for use in the next folding iteration
        for (j, error_poly_j) in error_polys.iter().enumerate() {
            self.constraint_errors[j] = evaluate_poly(error_poly_j, alpha)
        }

        // Evaluation of e(X) = (1-X)X⋅e'(X) + (1-X)⋅e(0) + X⋅e(1) in α
        self.error = evaluate_poly(&final_error_poly, alpha);

        // fold challenges
        for (c_acc, c_new) in zip(self.challenges_iter_mut(), new.challenges_iter()) {
            *c_acc += alpha * (*c_new - *c_acc);
        }

        // fold polynomials
        for (poly_acc, poly_new) in zip(self.polynomials_iter_mut(), new.polynomials_iter()) {
            poly_acc.boolean_linear_combination(poly_new, alpha);
        }

        // fold commitments
        {
            // Compute folded commitments in projective coordinates
            let commitments_projective: Vec<_> =
                zip(self.commitments_iter(), new.commitments_iter())
                    .map(|(c_acc, c_new)| (*c_new - *c_acc) * alpha + *c_acc)
                    .collect();
            // Convert to affine coordinates
            let mut commitments_affine = vec![C::identity(); commitments_projective.len()];
            C::CurveExt::batch_normalize(&commitments_projective, &mut commitments_affine);
            for (c_acc, c_new) in zip(self.commitments_iter_mut(), commitments_affine.into_iter()) {
                *c_acc = c_new
            }
        }

        // fold blinds
        for (blind_acc, blind_new) in zip(self.blinds_iter_mut(), new.blinds_iter()) {
            *blind_acc += (*blind_new - *blind_acc) * alpha
        }
    }

    /// Checks whether this accumulator is valid by recomputing all commitments and the error.
    /// The error evaluation is not optimized since this function should only be used for debugging.
    pub fn decide<'params, P: Params<'params, C>>(&self, params: &P, pk: &ProvingKey<C>) -> bool {
        // Recompute commitments and compare them to those in the accumulator
        let commitments_ok = {
            // Commitments in projective coordinates
            let commitments_projective: Vec<_> = zip(self.polynomials_iter(), self.blinds_iter())
                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                .collect();
            // Convert to affine coordinates
            let mut commitments = vec![C::identity(); commitments_projective.len()];
            C::CurveExt::batch_normalize(&commitments_projective, &mut commitments);
            // Compare with accumulator commitments
            zip(commitments.into_iter(), self.commitments_iter())
                .all(|(actual, expected)| actual == *expected)
        };

        // evaluate the error at each row
        let mut errors = vec![C::Scalar::ZERO; pk.num_usable_rows()];

        parallelize(&mut errors, |errors, start| {
            let ys = self.ys.clone();

            let (mut folding_errors, mut folding_gate_evs): (Vec<_>, Vec<_>) = pk
                .folding_constraints()
                .iter()
                .map(|polys| {
                    (
                        vec![C::Scalar::ZERO; polys.len()],
                        GateEvaluator::new_single(&polys, &self.advice_transcript.challenges),
                    )
                })
                .unzip();

            let mut linear_gate_ev = GateEvaluator::new_single(
                pk.linear_constraints(),
                &self.advice_transcript.challenges,
            );
            let mut linear_errors = vec![C::Scalar::ZERO; pk.linear_constraints().len()];

            let linear_challenge = self
                .compressed_verifier_transcript
                .beta()
                .pow_vartime([pk.num_usable_rows() as u64]);
            let linear_challenges: Vec<_> = powers(linear_challenge)
                .skip(1)
                .take(linear_errors.len())
                .collect();

            for (i, error) in errors.iter_mut().enumerate() {
                let row = start + i;

                for (es, (selector, gate_ev)) in folding_errors.iter_mut().zip(
                    pk.simple_selectors()
                        .iter()
                        .zip(folding_gate_evs.iter_mut()),
                ) {
                    if let Some(selector) = selector {
                        if !pk.selectors[selector.index()][row] {
                            es.fill(C::Scalar::ZERO);
                            continue;
                        }
                    }

                    gate_ev.evaluate_single(
                        es,
                        row,
                        &pk.selectors,
                        &pk.fixed,
                        &self.instance_transcript.instance_polys,
                        &self.advice_transcript.advice_polys,
                    );
                }

                linear_gate_ev.evaluate_single(
                    &mut linear_errors,
                    row,
                    &pk.selectors,
                    &pk.fixed,
                    &self.instance_transcript.instance_polys,
                    &self.advice_transcript.advice_polys,
                );

                *error += ys
                    .iter()
                    .zip(folding_errors.iter().flat_map(|e| e.iter()))
                    .fold(C::Scalar::ZERO, |acc, (y, e)| acc + *y * e);
                *error += linear_challenges
                    .iter()
                    .zip(linear_errors.iter())
                    .fold(C::Scalar::ZERO, |acc, (y, e)| acc + *y * e);
                *error *= self.compressed_verifier_transcript.beta_poly()[row];
            }
        });

        let error = errors.par_iter().sum::<C::Scalar>();

        let error_ok = error == self.error;

        commitments_ok & error_ok
    }

    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        self.instance_transcript
            .challenges_iter()
            .chain(self.advice_transcript.challenges_iter())
            .chain(
                self.lookup_transcript
                    .iter()
                    .flat_map(|lookup_transcript| lookup_transcript.challenges_iter()),
            )
            .chain(self.compressed_verifier_transcript.challenges_iter())
            .chain(self.ys.iter())
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        self.instance_transcript
            .challenges_iter_mut()
            .chain(self.advice_transcript.challenges_iter_mut())
            .chain(
                self.lookup_transcript
                    .iter_mut()
                    .flat_map(|lookup_transcript| lookup_transcript.challenges_iter_mut()),
            )
            .chain(self.compressed_verifier_transcript.challenges_iter_mut())
            .chain(self.ys.iter_mut())
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        self.instance_transcript
            .polynomials_iter()
            .chain(self.advice_transcript.polynomials_iter())
            .chain(
                self.lookup_transcript
                    .iter()
                    .flat_map(|lookup_transcript| lookup_transcript.polynomials_iter()),
            )
            .chain(self.compressed_verifier_transcript.polynomials_iter())
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        self.instance_transcript
            .polynomials_iter_mut()
            .chain(self.advice_transcript.polynomials_iter_mut())
            .chain(
                self.lookup_transcript
                    .iter_mut()
                    .flat_map(|lookup_transcript| lookup_transcript.polynomials_iter_mut()),
            )
            .chain(self.compressed_verifier_transcript.polynomials_iter_mut())
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        self.instance_transcript
            .commitments_iter()
            .chain(self.advice_transcript.commitments_iter())
            .chain(
                self.lookup_transcript
                    .iter()
                    .flat_map(|lookup_transcript| lookup_transcript.commitments_iter()),
            )
            .chain(self.compressed_verifier_transcript.commitments_iter())
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.instance_transcript
            .commitments_iter_mut()
            .chain(self.advice_transcript.commitments_iter_mut())
            .chain(
                self.lookup_transcript
                    .iter_mut()
                    .flat_map(|lookup_transcript| lookup_transcript.commitments_iter_mut()),
            )
            .chain(self.compressed_verifier_transcript.commitments_iter_mut())
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        self.instance_transcript
            .blinds_iter()
            .chain(self.advice_transcript.blinds_iter())
            .chain(
                self.lookup_transcript
                    .iter()
                    .flat_map(|lookup_transcript| lookup_transcript.blinds_iter()),
            )
            .chain(self.compressed_verifier_transcript.blinds_iter())
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        self.instance_transcript
            .blinds_iter_mut()
            .chain(self.advice_transcript.blinds_iter_mut())
            .chain(
                self.lookup_transcript
                    .iter_mut()
                    .flat_map(|lookup_transcript| lookup_transcript.blinds_iter_mut()),
            )
            .chain(self.compressed_verifier_transcript.blinds_iter_mut())
    }
}

/// For a linear polynomial p(X) such that p(0) = eval0, p(1) = eval1,
/// return an iterator yielding the evaluations p(j) for j = 0, 1, ... .
fn boolean_evaluations<F: Field>(eval0: F, eval1: F) -> impl Iterator<Item = F> {
    let linear = eval1 - eval0;
    std::iter::successors(Some(eval0), move |acc| Some(linear + acc))
}

/// For a sequence of linear polynomial p_k(X) such that p_k(0) = evals0[k], p_k(1) = evals1[k],
/// return an iterator yielding a vector of evaluations [p_0(j), p_1(j), ...] for j = 0, 1, ... .
fn boolean_evaluations_vec<F: Field>(
    evals0: Vec<F>,
    evals1: Vec<F>,
) -> impl Iterator<Item = Vec<F>> {
    debug_assert_eq!(evals0.len(), evals1.len());

    let mut diff = evals1;
    diff.iter_mut()
        .zip(evals0.iter())
        .for_each(|(eval1, eval0)| *eval1 -= eval0);

    let init = evals0;
    std::iter::successors(Some(init), move |acc| {
        let mut next = acc.clone();
        next.iter_mut()
            .zip(diff.iter())
            .for_each(|(next, diff)| *next += diff);
        Some(next)
    })
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn batch_lagrange_interpolate_integers<F: Field>(evals: &[Vec<F>]) -> Vec<Vec<F>> {
    let num_evals: Vec<_> = evals.iter().map(|evals| evals.len()).collect();
    let max_num_evals = *num_evals.iter().max().unwrap();

    let points: Vec<F> = field_integers().take(max_num_evals).collect();

    let mut denoms = Vec::with_capacity(points.len());
    for (j, x_j) in points.iter().enumerate() {
        let mut denom = Vec::with_capacity(points.len() - 1);
        for x_k in points
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != j)
            .map(|a| a.1)
        {
            denom.push(*x_j - x_k);
        }
        denoms.push(denom);
    }
    // Compute (x_j - x_k)^(-1) for each j != i
    denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

    let mut final_polys: Vec<_> = num_evals
        .iter()
        .map(|num_evals| vec![F::ZERO; *num_evals])
        .collect();

    for (final_poly, evals) in final_polys.iter_mut().zip(evals.iter()) {
        for (j, (denoms, eval)) in denoms.iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(evals.len());
            let mut product = Vec::with_capacity(evals.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .take(evals.len())
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.iter())
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-*denom * x_k) + *b * *denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), evals.len());
            assert_eq!(product.len(), evals.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp.into_iter()) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
    }

    final_polys
}

// Given a polynomial p(X) of degree d > 1, compute its quotient q(X)
// such that p(X) = (1-X)X⋅q(X).
// Panics if deg(p) ≤ 1 or if p(0) ≠ 0 or p(1) ≠ 0
fn quotient_by_boolean_vanishing<F: Field>(poly: &[F]) -> Vec<F> {
    let n = poly.len();
    assert!(n >= 2, "deg(poly) < 2");
    assert!(poly[0].is_zero_vartime(), "poly(0) != 0");

    let mut tmp = F::ZERO;

    let mut quotient = vec![F::ZERO; n - 2];
    for i in 0..(n - 2) {
        tmp += poly[i + 1];
        quotient[i] = tmp;
    }

    // p(1) = ∑p_i = 0
    assert_eq!(
        *quotient.last().unwrap(),
        poly.last().unwrap().neg(),
        "poly(1) != 0"
    );
    quotient
}

/// Evaluate a polynomial `poly` given as list of coefficients.
fn evaluate_poly<F: Field>(poly: &[F], point: F) -> F {
    let mut error = F::ZERO;
    for coeff in poly.iter().rev() {
        error *= point;
        error += coeff;
    }
    error
}

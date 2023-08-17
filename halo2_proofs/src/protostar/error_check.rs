use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Add, Deref, DerefMut, Index, IndexMut, RangeFrom, RangeFull, Sub},
};

use ff::{BatchInvert, Field};
use group::Curve;
use halo2curves::CurveAffine;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{
    arithmetic::{eval_polynomial, lagrange_interpolate, parallelize, powers},
    plonk::{self, lookup::Argument, Expression},
    poly::{
        commitment::{Blind, Params},
        LagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::{
    error_check::lookup::{error_poly_lookup_inputs, error_poly_lookup_tables},
    keygen::ProvingKey,
    row_evaluator::{
        evaluated_poly::EvaluatedFrom2, interpolate::LagrangeInterpolater, PolyBooleanEvaluator,
        RowBooleanEvaluator, RowEvaluator,
    },
    transcript::{
        advice::AdviceTranscript, compressed_verifier::CompressedVerifierTranscript,
        instance::InstanceTranscript, lookup::LookupTranscipt,
    },
};

mod lookup;

/// Each constraint error polynomial must be multiplied by the challenge(s) beta.
/// This allows us to compute the degree and therefore number of evaluations of e(X).
const BETA_POLY_DEGREE: usize = 1;

/// An `Accumulator` contains the entirety of the IOP transcript,
/// including commitments and verifier challenges.
#[derive(Debug, Clone, PartialEq)]
pub struct Accumulator<C: CurveAffine> {
    instance_transcript: InstanceTranscript<C>,
    advice_transcript: AdviceTranscript<C>,
    lookup_transcript: LookupTranscipt<C>,
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
        lookup_transcript: LookupTranscipt<C>,
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
        let num_rows = pk.num_usable_rows();
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
        let gate_selectors = pk.folding_constraints_selectors();

        /*
        Compute the number of constraints in each gate as well as the total number of constraints in all gates.
        */
        let num_constraints_per_gate: Vec<_> = pk
            .folding_constraints()
            .iter()
            .map(|polys| polys.len())
            .collect();
        let num_constraints: usize = num_constraints_per_gate.iter().sum();

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
        let mut gates_error_polys_evals_from_2: Vec<Vec<_>> = gates_error_polys_num_evals
            .iter()
            .map(|error_polys_num_evals| {
                error_polys_num_evals
                    .iter()
                    .map(|num_evals| EvaluatedFrom2::new_empty(*num_evals))
                    .collect()
            })
            .collect();

        /*
        Compute the maximum number of evaluations over all error polynomials over all gates.
        This defines the maximal evaluation domain D = {0, 1, …, dₘₐₓ + BETA_POLY_DEGREE + 1}
        This will allow us to compute the evaluations βᵢ(D)
        */
        let max_gate_num_evals = *gates_error_polys_num_evals
            .iter()
            .flat_map(|gates_num_evals| gates_num_evals.iter())
            .max()
            .unwrap();

        /*
        A `GateEvaluator` pre-processes a gate's polynomials for more efficient evaluation.
        In particular, it collects all column queries and allocates buffers where the corresponding
        values can be stored. Each polynomial in the gate is evaluated several times,
        so it is advantageous to only fetch the values from the columns once.

        For gate at index `gate_idx` and each row i of the accumulator, the `GateEvaluator` will store the following evaluations
            [e₀,ᵢ(D₀), …, eₘ₋₁,ᵢ(Dₘ₋₁)]
        and return a reference to it when calling `evaluate`.
        We give a copy of the empty buffer of [e₀(D₀), …, eₘ₋₁(Dₘ₋₁)] so the `GateEvaluator` knows how many evaluations we want.
         */
        let mut gate_evs: Vec<_> = pk
            .folding_constraints()
            .iter()
            .zip(gates_error_polys_evals_from_2.iter())
            .map(|(polys, gate_error_polys_evals)| {
                RowBooleanEvaluator::new(
                    polys,
                    &self.advice_transcript.challenges,
                    &new.advice_transcript.challenges,
                    gate_error_polys_evals.clone(),
                )
            })
            .collect();

        /*
        Create an evaluator for the beta polynomial.
        TODO(@adr1anh): investigate whether we should precompute them all and cache the results.
        */
        let mut beta_ev =
            PolyBooleanEvaluator::new(self.beta_poly(), new.beta_poly(), max_gate_num_evals);

        /*
        TODO(@adr1anh): Parallelize this loop.
         */
        for row_idx in 0..num_rows {
            // Get the next evaluation βᵢ(D) of βᵢ(X) = ((1-X)⋅acc.βᵢ + X⋅acc.βᵢ)
            let beta_poly_evals = beta_ev.evaluate_from_2(row_idx);

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
                let row_error_polys_evals = gate_ev.evaluate_all_from_2(
                    row_idx,
                    &pk.selectors,
                    &pk.fixed,
                    [&self.instance_polys(), &new.instance_polys()],
                    [&self.advice_polys(), &new.advice_polys()],
                );

                // Get the running sums for [e₀(D₀), …, eₘ₋₁(Dₘ₋₁)] for the current gate
                // to which we want to add the row's error polynomials evaluations.
                let error_polys_evals = &mut gates_error_polys_evals_from_2[gate_idx];
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
                    error_poly_evals.add_prod(row_error_poly_evals, beta_poly_evals);
                }
            }
        }

        // Compute evaluations of error polynomials for the lookup constraints
        let lookup_error_polys_evals: Vec<_> = pk
            .cs()
            .lookups
            .iter()
            .enumerate()
            .flat_map(|(lookup_idx, lookup_arg)| {
                let tx0 = &self.lookup_transcript.singles_transcript[lookup_idx];
                let tx1 = &new.lookup_transcript.singles_transcript[lookup_idx];

                // SAFETY: We can safely unwrap the challenges since they are Some whenever pk.cs().lookups() is non-empty

                let inputs_error_poly_evals = error_poly_lookup_inputs(
                    num_rows,
                    &pk.selectors,
                    &pk.fixed,
                    [&self.advice_challenges(), &new.advice_challenges()],
                    [&self.beta_poly(), &new.beta_poly()],
                    [&self.advice_polys(), &new.advice_polys()],
                    [&self.instance_polys(), &new.instance_polys()],
                    lookup_arg.input_expressions(),
                    [
                        &self.lookup_transcript.r.unwrap(),
                        &new.lookup_transcript.r.unwrap(),
                    ],
                    [
                        &self.lookup_transcript.thetas.as_ref().unwrap(),
                        &new.lookup_transcript.thetas.as_ref().unwrap(),
                    ],
                    [&tx0.g_poly, &tx1.g_poly],
                );
                let tables_error_poly_evals = error_poly_lookup_tables(
                    num_rows,
                    &pk.selectors,
                    &pk.fixed,
                    [&self.advice_challenges(), &new.advice_challenges()],
                    [&self.beta_poly(), &new.beta_poly()],
                    [&self.advice_polys(), &new.advice_polys()],
                    [&self.instance_polys(), &new.instance_polys()],
                    lookup_arg.table_expressions(),
                    [
                        &self.lookup_transcript.r.unwrap(),
                        &new.lookup_transcript.r.unwrap(),
                    ],
                    [
                        &self.lookup_transcript.thetas.as_ref().unwrap(),
                        &new.lookup_transcript.thetas.as_ref().unwrap(),
                    ],
                    [&tx0.m_poly, &tx1.m_poly],
                    [&tx0.h_poly, &tx1.h_poly],
                );
                [inputs_error_poly_evals, tables_error_poly_evals].into_iter()
            })
            .collect();

        /*
        Now that we have evaluated all gates, we no longer need to keep track of the nested structure.
        We flatten `gates_error_polys_evals` into `error_polys_evals`, and let `m = num_constraints`.
        The result is the list of error polynomial evaluations e₀(D₀'), …, eₘ₋₁(Dₘ₋₁')
        */
        let error_polys_evals_from_2: Vec<_> = gates_error_polys_evals_from_2
            .into_iter()
            .flat_map(|poly| poly.into_iter())
            .chain(lookup_error_polys_evals.into_iter())
            .collect();

        // Sanity checks
        debug_assert_eq!(error_polys_evals_from_2.len(), num_constraints);
        debug_assert_eq!(self.constraint_errors.len(), num_constraints);
        debug_assert_eq!(new.constraint_errors.len(), num_constraints);

        /*
        Precompute `LagrangeInterpolator` which containts the inverse Vandermonde matrices
        for computing a polynomial's coefficients from its evaluations.
        */
        let interpolator = {
            let max_num_evals = error_polys_evals_from_2
                .iter()
                .map(|evals| evals.num_evals())
                .max()
                .unwrap();

            LagrangeInterpolater::<C::Scalar>::new_integer_eval_domain(max_num_evals)
        };

        /*
        Compute the polynomials e₀(X), …, eₘ₋₁(X) in coefficient coefficient
        */
        let error_polys: Vec<Vec<C::Scalar>> = error_polys_evals_from_2
            .into_iter()
            .zip(zip(
                self.constraint_errors.iter(),
                new.constraint_errors.iter(),
            ))
            // Get eⱼ(0), eⱼ(1) from the accumulators `self` and `new` and map eⱼ(Dⱼ') to eⱼ(Dⱼ)
            .map(|(error_poly_evals_from_2, (eval0, eval1))| {
                error_poly_evals_from_2.to_evaluated(*eval0, *eval1)
            })
            /*
            Map polynomials evaluations e₀(D₀), …, eₘ₋₁(Dₘ₋₁) into their coefficient representation  e₀(X), …, eₘ₋₁(X)
            */
            .map(|error_poly_evals| interpolator.interpolate(&error_poly_evals))
            .collect();

        /*
        For linear independence of all error polynomials, we multiply each polynomial eⱼ(X) by
          yⱼ(X) = (1−X)⋅acc.yⱼ + X⋅new.yⱼ = acc.yⱼ + X⋅(new.yⱼ - acc.yⱼ)
        We then compute `final_error_poly` as e(X) = ∑ⱼ yⱼ(X)⋅eⱼ(X), which has degree
          d = `max_degree` + NUM_EXTRA_EVALUATIONS (for βᵢ(X)) + 1 (for yⱼ(X))
        We add 1 for the length.
        */
        let final_error_poly_len = error_polys.iter().map(|poly| poly.len()).max().unwrap() + 1;

        debug_assert_eq!(self.ys.len(), num_constraints);
        debug_assert_eq!(new.ys.len(), num_constraints);

        let final_error_poly = error_polys
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
            );

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
            self.constraint_errors[j] = eval_polynomial(error_poly_j, alpha)
        }

        // Evaluation of e(X) = (1-X)X⋅e'(X) + (1-X)⋅e(0) + X⋅e(1) in α
        self.error = eval_polynomial(&final_error_poly, alpha);

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

        // Recompute e = ∑ᵢ βᵢ⋅eᵢ, where eᵢ = ∑ⱼ yⱼ⋅Gⱼ(acc[i])
        let folding_errors_ok = {
            // Store βᵢ⋅eᵢ for each row i
            let mut folding_errors = vec![C::Scalar::ZERO; pk.num_usable_rows()];

            parallelize(&mut folding_errors, |errors, start| {
                let gates_selectors = pk.folding_constraints_selectors();
                let folding_constraints = pk.folding_constraints();

                // For each gate, create a slice of y's corresponding to the polynomials in the gate
                let gates_ys: Vec<&[C::Scalar]> = {
                    let mut start = 0;
                    let mut gates_ys = Vec::with_capacity(folding_constraints.len());

                    for polys in folding_constraints {
                        let size = polys.len(); // Get size of inner vector
                        let end = start + size;
                        gates_ys.push(&self.ys[start..end]);
                        start = end;
                    }

                    gates_ys
                };

                // Create a RowEvaluator for each gate
                let mut folding_gate_evs: Vec<_> = folding_constraints
                    .iter()
                    .map(|polys| RowEvaluator::new(&polys, &self.advice_challenges()))
                    .collect();

                for (i, error) in errors.iter_mut().enumerate() {
                    let row = start + i;

                    for (gate_ev, (gate_ys, gate_selector)) in folding_gate_evs
                        .iter_mut()
                        .zip(zip(gates_ys.iter(), gates_selectors.iter()))
                    {
                        // Check whether the gate's selector is active at this row
                        if let Some(selector) = gate_selector {
                            if !pk.selectors[selector.index()][row] {
                                continue;
                            }
                        }

                        // Evaluate [Gⱼ(acc[i])] for all Gⱼ in the gate
                        let row_evals = gate_ev.evaluate(
                            row,
                            &pk.selectors,
                            &pk.fixed,
                            &self.instance_polys(),
                            &self.advice_polys(),
                        );

                        // Add ∑ⱼ yⱼ⋅Gⱼ(acc[i]) to eᵢ
                        *error += zip(gate_ys.iter(), row_evals.iter())
                            .fold(C::Scalar::ZERO, |acc, (y, eval)| acc + *y * eval);
                    }
                    // eᵢ *= βᵢ⋅eᵢ
                    *error *= self.beta_poly()[row];
                }
            });

            // Sum all eᵢ
            let folding_error = folding_errors.par_iter().sum::<C::Scalar>();

            folding_error == self.error
        };

        // Check all linear constraints, which were ignored during folding
        let linear_errors_ok = {
            let mut linear_errors = vec![true; pk.num_usable_rows()];

            parallelize(&mut linear_errors, |errors, start| {
                let linear_constraints = pk.linear_constraints();

                // Create a single evaluator for all linear constraints
                let mut linear_gate_ev =
                    RowEvaluator::new(linear_constraints, &self.advice_challenges());

                for (i, error) in errors.iter_mut().enumerate() {
                    let row = start + i;

                    let row_evals = linear_gate_ev.evaluate(
                        row,
                        &pk.selectors,
                        &pk.fixed,
                        &self.instance_polys(),
                        &self.advice_polys(),
                    );

                    // Check that all constraints evaluate to 0.
                    *error = row_evals.iter().all(|eval| eval.is_zero_vartime());
                }
            });

            linear_errors.par_iter().all(|row_ok| *row_ok)
        };

        // TODO(@adr1anh): Check lookups

        // TODO(@adr1anh): Maybe check permutations too?

        commitments_ok & folding_errors_ok & linear_errors_ok
    }

    fn advice_challenges(&self) -> &[Vec<C::Scalar>] {
        &self.advice_transcript.challenges
    }

    fn advice_polys(&self) -> &[Polynomial<C::Scalar, LagrangeCoeff>] {
        &self.advice_transcript.advice_polys
    }

    fn instance_polys(&self) -> &[Polynomial<C::Scalar, LagrangeCoeff>] {
        &self.instance_transcript.instance_polys
    }

    fn beta_poly(&self) -> &Polynomial<C::Scalar, LagrangeCoeff> {
        &self.compressed_verifier_transcript.beta_poly()
    }

    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        self.instance_transcript
            .challenges_iter()
            .chain(self.advice_transcript.challenges_iter())
            .chain(self.lookup_transcript.challenges_iter())
            .chain(self.compressed_verifier_transcript.challenges_iter())
            .chain(self.ys.iter())
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        self.instance_transcript
            .challenges_iter_mut()
            .chain(self.advice_transcript.challenges_iter_mut())
            .chain(self.lookup_transcript.challenges_iter_mut())
            .chain(self.compressed_verifier_transcript.challenges_iter_mut())
            .chain(self.ys.iter_mut())
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        self.instance_transcript
            .polynomials_iter()
            .chain(self.advice_transcript.polynomials_iter())
            .chain(self.lookup_transcript.polynomials_iter())
            .chain(self.compressed_verifier_transcript.polynomials_iter())
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        self.instance_transcript
            .polynomials_iter_mut()
            .chain(self.advice_transcript.polynomials_iter_mut())
            .chain(self.lookup_transcript.polynomials_iter_mut())
            .chain(self.compressed_verifier_transcript.polynomials_iter_mut())
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        self.instance_transcript
            .commitments_iter()
            .chain(self.advice_transcript.commitments_iter())
            .chain(self.lookup_transcript.commitments_iter())
            .chain(self.compressed_verifier_transcript.commitments_iter())
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.instance_transcript
            .commitments_iter_mut()
            .chain(self.advice_transcript.commitments_iter_mut())
            .chain(self.lookup_transcript.commitments_iter_mut())
            .chain(self.compressed_verifier_transcript.commitments_iter_mut())
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        self.instance_transcript
            .blinds_iter()
            .chain(self.advice_transcript.blinds_iter())
            .chain(self.lookup_transcript.blinds_iter())
            .chain(self.compressed_verifier_transcript.blinds_iter())
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        self.instance_transcript
            .blinds_iter_mut()
            .chain(self.advice_transcript.blinds_iter_mut())
            .chain(self.lookup_transcript.blinds_iter_mut())
            .chain(self.compressed_verifier_transcript.blinds_iter_mut())
    }
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

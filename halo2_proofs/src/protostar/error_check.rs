use core::num;
use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Add, Deref, DerefMut, Index, IndexMut, RangeFrom, RangeFull, Sub},
};

mod gate;
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

/// A constraint whose degree is 1 has a trivial (i.e. zero) error polynomial.
/// During the evaluation, we skip these constraints since they would
/// otherwise need to be evaluated NUM_EXTRA_EVALUATIONS many times.
pub(super) const MIN_GATE_DEGREE: usize = 1;
/// Each constraint error polynomial must be multiplied by the challenges
/// beta and y, so we need to evaluate it in 2 additional points.
const NUM_EXTRA_EVALUATIONS: usize = 2;
/// The evaluations of the error polynomials at 0 and 1 correspond to the
/// evaluation of the constraints in the accumulator and the new witness.
/// We cache these in `Accumulator` and start evaluating at X=2
const NUM_SKIPPED_EVALUATIONS: usize = 2;

/// An `Accumulator` contains the entirety of the IOP transcript,
/// including commitments and verifier challenges.
#[derive(Debug)]
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
        let num_rows_i = pk.num_rows() as i32;
        let mut gate_caches: Vec<_> = pk
            .folding_constraints()
            .iter()
            .map(|polys| {
                GateEvaluator::new(
                    polys,
                    &self.advice_transcript.challenges,
                    &new.advice_transcript.challenges,
                    num_rows_i,
                )
            })
            .collect();

        // Total number of evaluations of the final error polynomial, ignoring the evaluations at 0, 1.
        let max_num_evals = pk.max_degree() + 1 + NUM_EXTRA_EVALUATIONS - NUM_SKIPPED_EVALUATIONS;

        // Store the sum of the
        // eval_sums[gate_idx][constraint_idx][eval_idx]
        let mut error_polys: Vec<_> = gate_caches
            .iter()
            .map(|gate_cache| gate_cache.empty_error_polynomials())
            .collect();

        for row_idx in 0..pk.num_usable_rows() {
            // Get the next evaluation of ((1-X) * acc.b + X * acc.b)
            let beta_acc = self.compressed_verifier_transcript.beta_poly()[row_idx];
            let beta_new = new.compressed_verifier_transcript.beta_poly()[row_idx];
            let beta_evals: Vec<_> = boolean_evaluations_skip_2(beta_acc, beta_new)
                .take(max_num_evals)
                .collect();

            for (gate_idx, gate_cache) in gate_caches.iter_mut().enumerate() {
                // Return early if the selector is off.
                if let Some(selector) = pk.simple_selectors()[gate_idx] {
                    if !pk.selectors[selector.index()][row_idx] {
                        continue;
                    }
                }
                // Compute evaluations of the gate constraints at X = 2, 3, ... ,
                // multiply them by the evaluations of beta,
                // and add them to error_polys_evals.
                let evals = gate_cache.evaluate_and_accumulate_errors(
                    row_idx,
                    &pk.selectors,
                    &pk.fixed,
                    &self.instance_transcript.instance_polys,
                    &new.instance_transcript.instance_polys,
                    &self.advice_transcript.advice_polys,
                    &new.advice_transcript.advice_polys,
                );
                // For each constraint in the current gate, scale it by the beta vector
                // and add it to the corresponding error polynomial accumulator `error_polys`
                for (acc_error_poly, new_error_poly) in
                    zip(error_polys[gate_idx].iter_mut(), evals.iter())
                {
                    // Multiply each poly by beta and add it to the corresponding accumulator
                    for (acc_error_eval, (new_error_eval, beta_eval)) in acc_error_poly
                        .iter_mut()
                        .zip(zip(new_error_poly.iter(), beta_evals.iter()))
                    {
                        *acc_error_eval += *new_error_eval * beta_eval;
                    }
                }
            }
        }

        // collect all lists of evaluations
        let mut final_evals: Vec<_> = error_polys
            .into_iter()
            .flat_map(|poly| poly.into_iter())
            .collect();

        // multiply each list of evaluations by the challenge y
        final_evals
            .iter_mut()
            .zip(zip(self.ys.iter(), new.ys.iter()))
            .for_each(|(evals, (y_acc, y_new))| {
                let y_evals_iter = boolean_evaluations_skip_2(*y_acc, *y_new);
                for (error_eval, y_eval) in zip(evals.iter_mut(), y_evals_iter) {
                    *error_eval *= y_eval;
                }
            });

        // re-insert the evaluations at 0,1 from the cache for each constraint
        final_evals
            .iter_mut()
            .zip(zip(
                self.constraint_errors.iter(),
                new.constraint_errors.iter(),
            ))
            .for_each(|(final_evals, (acc_eval, new_eval))| {
                final_evals.insert(0, *new_eval);
                final_evals.insert(0, *acc_eval);
            });

        // convert evaluations into coefficients
        let final_polys = batch_lagrange_interpolate_integers(&final_evals);

        // add all polynomials together
        let mut final_poly = final_polys.iter().fold(
            vec![C::Scalar::ZERO; max_num_evals + NUM_SKIPPED_EVALUATIONS],
            |mut final_poly, coeffs| {
                assert!(coeffs.len() <= final_poly.len());
                final_poly
                    .iter_mut()
                    .zip(coeffs.iter())
                    .for_each(|(final_coeff, coeff)| *final_coeff += coeff);
                final_poly
            },
        );

        // The error term for the existing accumulator should be e(0),
        // which is equal to the first coefficient of `final_poly`
        let expected_error_acc: C::Scalar = final_poly[0];
        assert_eq!(expected_error_acc, self.error);

        // The error term for the fresh accumulator should be e(1),
        // which is equal to the sum of the coefficient of `final_poly`
        let expected_error_new: C::Scalar = final_poly.iter().sum();
        assert_eq!(expected_error_new, new.error);

        // Subtract (1-X)⋅acc.error + X⋅new.error
        final_poly[0] -= self.error;
        final_poly[1] -= new.error - self.error;

        // Compute and commit to quotient
        //
        //          e(X) - (1-X)⋅e(0) - X⋅e(1)
        // e'(X) = ----------------------------
        //                    (X-1)X
        //
        // Since the verifier already knows e(0), e(1),
        // we can send this polynomial instead and save 2 hashes.
        // The verifier evaluates e(X) as
        //   e(X) = (1-X)X⋅e'(X) + (1-X)⋅e(0) + X⋅e(1)
        let quotient_poly = quotient_by_boolean_vanishing(&final_poly);
        for coef in quotient_poly.iter() {
            let _ = transcript.write_scalar(*coef);
        }

        // Get alpha challenge
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();

        // Evaluate all cached constraint errors
        for (error, final_poly) in self.constraint_errors.iter_mut().zip(final_polys.iter()) {
            let mut eval = C::Scalar::ZERO;
            for coeff in final_poly.iter().rev() {
                eval *= alpha;
                eval += coeff;
            }
            *error = eval;
        }

        // Evaluation of e(X) = (1-X)X⋅e'(X) + (1-X)⋅e(0) + X⋅e(1) in alpha
        self.error = {
            let mut error = C::Scalar::ZERO;
            for coeff in quotient_poly.iter().rev() {
                error *= alpha;
                error += coeff;
            }
            error *= alpha.square() - alpha;
            error += (C::Scalar::ONE - alpha) * self.error;
            error += alpha * new.error;
            error
        };

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
            let commitments_projective: Vec<_> =
                zip(self.commitments_iter(), new.commitments_iter())
                    .map(|(c_acc, c_new)| (*c_new - *c_acc) * alpha + *c_acc)
                    .collect();
            let mut commitments_affine = vec![C::identity(); commitments_projective.len()];
            C::CurveExt::batch_normalize(&commitments_projective, &mut commitments_affine);
            for (c_acc, c_new) in zip(self.commitments_iter_mut(), commitments_affine) {
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
        let commitments_ok = {
            let commitments_projective: Vec<_> = zip(self.polynomials_iter(), self.blinds_iter())
                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                .collect();
            let mut commitments = vec![C::identity(); commitments_projective.len()];
            C::CurveExt::batch_normalize(&commitments_projective, &mut commitments);
            zip(commitments, self.commitments_iter()).all(|(actual, expected)| actual == *expected)
        };

        /// Return the index in the polynomial of size `isize` after rotation `rot`.
        fn get_rotation_idx(idx: usize, rot: Rotation, isize: i32) -> usize {
            (((idx as i32) + rot.0).rem_euclid(isize)) as usize
        }

        // evaluate the error at each row
        let mut errors = vec![C::Scalar::ZERO; pk.num_usable_rows()];

        parallelize(&mut errors, |errors, start| {
            let num_rows_i = pk.num_rows() as i32;

            // re-multiply by selectors
            let polys: Vec<_> = pk
                .folding_constraints()
                .iter()
                .zip(pk.simple_selectors().iter())
                .flat_map(|(polys, selector)| {
                    polys.iter().map(move |poly| {
                        if let Some(s) = selector {
                            poly.clone() * s.expr()
                        } else {
                            poly.clone()
                        }
                    })
                })
                .collect();
            let ys = self.ys.clone();

            let mut es: Vec<_> = vec![C::Scalar::ZERO; polys.len()];
            for (i, error) in errors.iter_mut().enumerate() {
                let row = start + i;

                for (e, poly) in zip(es.iter_mut(), polys.iter()) {
                    *e = poly.evaluate(
                        &|constant| constant,
                        &|selector| {
                            if pk.selectors[selector.0][row] {
                                C::Scalar::ONE
                            } else {
                                C::Scalar::ZERO
                            }
                        },
                        &|query| {
                            pk.fixed[query.column_index()]
                                [get_rotation_idx(row, query.rotation(), num_rows_i)]
                        },
                        &|query| {
                            self.advice_transcript.advice_polys[query.column_index()]
                                [get_rotation_idx(row, query.rotation(), num_rows_i)]
                        },
                        &|query| {
                            self.instance_transcript.instance_polys[query.column_index()]
                                [get_rotation_idx(row, query.rotation(), num_rows_i)]
                        },
                        &|challenge| {
                            self.advice_transcript.challenges[challenge.index()]
                                [challenge.power() - 1]
                        },
                        &|negated| -negated,
                        &|sum_a, sum_b| sum_a + sum_b,
                        &|prod_a, prod_b| prod_a * prod_b,
                        &|scaled, v| scaled * v,
                    );
                }

                *error =
                    zip(ys.iter(), es.iter()).fold(C::Scalar::ZERO, |acc, (y, e)| acc + *y * e);
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

/// Same as `boolean_evaluations` but starting at j = 2.
fn boolean_evaluations_skip_2<F: Field>(eval0: F, eval1: F) -> impl Iterator<Item = F> {
    let linear = eval1 - eval0;
    // p(2) = (1-2)⋅eval0 + 2⋅eval1 = eval1 + (eval1 - eval0) = eval1 + linear
    std::iter::successors(Some(eval1 + linear), move |acc| Some(linear + acc))
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

/// Same as boolean_evaluations_vec but starting at j=2.
fn boolean_evaluations_vec_skip_2<F: Field>(
    evals0: Vec<F>,
    evals1: Vec<F>,
) -> impl Iterator<Item = Vec<F>> {
    debug_assert_eq!(evals0.len(), evals1.len());

    // diff = evals1 - evals0
    let mut diff = evals0;
    diff.iter_mut()
        .zip(evals1.iter())
        .for_each(|(eval0, eval1)| {
            let diff = *eval1 - *eval0;
            *eval0 = diff;
        });

    // init = evals1 + diff
    let mut init = evals1;
    init.iter_mut()
        .zip(diff.iter())
        .for_each(|(eval1, diff)| *eval1 += diff);
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

    let quotient: Vec<_> = poly
        .iter()
        .skip(1)
        .map(|a_i| {
            tmp -= a_i;
            tmp
        })
        .take(n - 2)
        .collect();
    // p(1) = ∑p_i = 0
    assert_eq!(
        quotient.last().unwrap(),
        poly.last().unwrap(),
        "poly(1) != 0"
    );
    quotient
}

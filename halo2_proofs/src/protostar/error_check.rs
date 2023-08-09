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

/// An `Accumulator` contains the entirety of the IOP transcript,
/// including commitments and verifier challenges.
#[derive(Debug)]
pub struct Accumulator<C: CurveAffine> {
    instance_transcript: InstanceTranscript<C>,
    advice_transcript: AdviceTranscript<C>,
    compressed_verifier_transcript: CompressedVerifierTranscript<C>,
    lookup_transcript: Option<LookupTranscipt<C>>,

    // Powers of a challenge y for taking a random linear-combination of all constraints.
    ys: Vec<C::Scalar>,

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
        let ys: Vec<_> = powers(y).skip(1).take(pk.num_constraints()).collect();

        Self {
            instance_transcript,
            advice_transcript,
            compressed_verifier_transcript,
            lookup_transcript,
            ys,
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
        // 1 for beta, 1 for y
        let num_extra_evaluations = 2;

        let mut gate_caches: Vec<_> = pk
            .cs()
            .gates()
            .iter()
            .map(|gate| {
                GateEvaluator::new(
                    gate,
                    &self.advice_transcript.challenges,
                    &new.advice_transcript.challenges,
                    num_extra_evaluations,
                )
            })
            .collect();

        let num_rows_i = pk.num_rows() as i32;
        let num_evals = pk.max_degree() + 1 + num_extra_evaluations;

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
            let beta_evals: Vec<_> = boolean_evaluations(beta_acc, beta_new)
                .take(num_evals)
                .collect();

            for (gate_idx, gate_cache) in gate_caches.iter_mut().enumerate() {
                let evals = gate_cache.evaluate(row_idx, num_rows_i, &pk, self, &new);
                if let Some(evals) = evals {
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
                let y_evals_iter = boolean_evaluations(*y_acc, *y_new);
                for (error_eval, y_eval) in zip(evals.iter_mut(), y_evals_iter) {
                    *error_eval *= y_eval;
                }
            });

        // convert evaluations into coefficients
        let final_polys = batch_lagrange_interpolate_integers(&final_evals);

        // add all polynomials together
        let final_poly = final_polys.into_iter().fold(
            vec![C::Scalar::ZERO; num_evals],
            |mut final_poly, coeffs| {
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

        // Commit to error poly
        for coef in final_poly.iter() {
            let _ = transcript.write_scalar(*coef);
        }

        // Get alpha challenge
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();

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

        // horner eval of the polynomial e(X) in alpha
        self.error = C::Scalar::ZERO;
        for coeff in final_poly.iter().rev() {
            self.error *= alpha;
            self.error += coeff;
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

            let polys: Vec<_> = pk
                .cs()
                .gates()
                .iter()
                .flat_map(|gate| {
                    gate.polynomials()
                        .iter()
                        .map(|poly| poly.clone().merge_challenge_products())
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

/// For a sequence of linear polynomial p_k(X) such that p_k(0) = evals0[k], p_k(1) = evals1[k],
/// return an iterator yielding a vector of evaluations [p_0(j), p_1(j), ...] for j = 0, 1, ... .
fn boolean_evaluations_vec<F: Field>(evals0: &[F], evals1: &[F]) -> impl Iterator<Item = Vec<F>> {
    assert_eq!(evals0.len(), evals1.len());
    let diff: Vec<_> = zip(evals0.iter(), evals1.iter())
        .map(|(eval0, eval1)| *eval1 - eval0)
        .collect();
    std::iter::successors(Some(evals0.to_vec()), move |acc| {
        Some(
            zip(acc.iter(), diff.iter())
                .map(|(acc, diff)| *acc + diff)
                .collect(),
        )
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

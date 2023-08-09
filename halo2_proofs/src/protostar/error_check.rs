use core::num;
use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Add, Deref, DerefMut, Index, IndexMut, RangeFrom, RangeFull, Sub},
};
mod gate;
use crate::{
    arithmetic::{field_integers, lagrange_interpolate, parallelize, powers},
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        empty_lagrange, LagrangeCoeff, Polynomial, Rotation,
    },
    protostar::error_check::gate::GateEvaluationCache,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;

use super::{
    gate::Gate,
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
        // #[cfg(feature = "sanity-checks")]
        // {
        //     assert_eq!(
        //         challenges.len(),
        //         pk.num_challenges(),
        //         "invalid number of challenges supplied"
        //     );
        //     assert_eq!(
        //         instance.len(),
        //         pk.num_instance_columns(),
        //         "invalid number of instance columns supplied"
        //     );

        //     for (i, (instance_len, expected_len)) in zip(
        //         instance.iter().map(|instance| instance.len()),
        //         pk.num_instance_rows().iter(),
        //     )
        //     .enumerate()
        //     {
        //         assert_eq!(
        //             instance_len, *expected_len,
        //             "invalid size for instance column {i}"
        //         );
        //     }
        // }

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
            .gates()
            .iter()
            .map(|gate| {
                GateEvaluationCache::new(
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
                        // Multiply each poly by b1*b2 and add it to the corresponding accumulator
                        acc_error_poly.add_multiplied(new_error_poly, &beta_evals);
                    }
                }
            }
        }

        let final_poly = {
            let mut final_poly = vec![C::Scalar::ZERO; num_evals];

            // TODO(@adr1anh): The inner loop can be optimized since `to_coefficients` does a lot of common preprocessing.
            for (evals, (y_acc, y_new)) in error_polys
                .iter_mut()
                .flat_map(|polys| polys.iter_mut())
                .zip(zip(self.ys.iter(), new.ys.iter()))
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

        // TODO(@adr1anh): Get alpha challenge from transcript
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
    pub fn decide<'params, P: Params<'params, C>>(&self, params: &P, pk: &ProvingKey<C>) -> bool {
        let commitments_ok = {
            let commitments_projective: Vec<_> = zip(self.polynomials_iter(), self.blinds_iter())
                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                .collect();
            let mut commitments = vec![C::identity(); commitments_projective.len()];
            C::CurveExt::batch_normalize(&commitments_projective, &mut commitments);
            zip(commitments, self.commitments_iter()).all(|(actual, expected)| actual == *expected)
        };

        // TODO(@adr1anh): Cleanup gate evaluation before implementing the rest
        // let num_rows_i = pk.num_rows() as i32;
        // /// Return the index in the polynomial of size `isize` after rotation `rot`.
        // fn get_rotation_idx(idx: usize, rot: Rotation, isize: i32) -> usize {
        //     (((idx as i32) + rot.0).rem_euclid(isize)) as usize
        // }

        // let mut errors = vec![C::Scalar::ZERO; pk.num_usable_rows()];
        // parallelize(&mut errors, |errors, start| {
        //     let mut es: Vec<_> = pk
        //         .gates()
        //         .iter()
        //         .map(|gate| vec![C::Scalar::ZERO; gate.polys.len()])
        //         .collect();
        //     for (i, e) in errors.iter_mut().enumerate() {
        //         let row = start + i;

        //         for (gate, es) in zip(pk.gates().iter(), es.iter_mut()) {
        //             if let Some(simple_selector) = gate.simple_selector {
        //                 let selector_column = simple_selector.0;
        //                 let value = pk.selectors[selector_column][row];
        //                 // do nothing and return
        //                 if !value {
        //                     continue;
        //                 }
        //             }
        //         }

        //         // let e_single = poly.evaluate(
        //         //     &|constant| constant,
        //         //     &|selector_idx| if pk.selectors[selector_idx][idx]{C::Scalar::ONE} else {C::Scalar::ZERO},
        //         //     &|fixed_idx| pk.fixed[fixed_idx],
        //         //     &|advice_idx| advice_tmp[advice_idx],
        //         //     &|instance_idx| instance_tmp[instance_idx],
        //         //     &|challenge_idx, challenge_power| {
        //         //         self.challenge_powers_evals[eval_idx][challenge_idx][challenge_power - 1]
        //         //     },
        //         //     &|negated| -negated,
        //         //     &|sum_a, sum_b| sum_a + sum_b,
        //         //     &|prod_a, prod_b| prod_a * prod_b,
        //         //     &|scaled, v| scaled * v,)
        //     }
        // });

        commitments_ok
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

// /// Non-comforming iterator yielding evaluations of
// /// (acc.β₀ + X⋅new.β₀)⋅(acc.β₁ + X⋅new.β₁)
// struct BetaIterator<'a, 'b, F: Field> {
//     betas0_acc: &'a [F],
//     betas1_acc: &'a [F],
//     betas0_new: &'b [F],
//     betas1_new: &'b [F],

//     // log2(n^{1/2})
//     k_sqrt: u32,
//     // 2^k_sqrt - 1
//     mask: usize,

//     evals: Vec<F>,
// }

// impl<'a, 'b, F: Field> BetaIterator<'a, 'b, F> {
//     fn new(
//         betas0_acc: &'a [F],
//         betas1_acc: &'a [F],
//         betas0_new: &'b [F],
//         betas1_new: &'b [F],
//         k_sqrt: u32,
//         num_evals: usize,
//     ) -> Self {
//         let n_sqrt = 1 << k_sqrt;
//         let mask = n_sqrt - 1;
//         Self {
//             betas0_acc,
//             betas1_acc,
//             betas0_new,
//             betas1_new,
//             k_sqrt,
//             mask,
//             evals: vec![F::ZERO; num_evals],
//         }
//     }

//     fn evals_at_row(&mut self, row_index: usize) -> &[F] {
//         let i0 = row_index & self.mask;
//         let i1 = row_index >> self.k_sqrt;
//         let mut beta0 = self.betas0_acc[i0];
//         let mut beta1 = self.betas1_acc[i1];
//         let beta0_new = self.betas0_new[i0];
//         let beta1_new = self.betas1_new[i1];

//         self.evals[0] = beta0 * beta1;
//         for i in 1..self.evals.len() {
//             beta0 += beta0_new;
//             beta1 += beta1_new;
//             self.evals[i] = beta0 * beta1;
//         }

//         self.evals.as_slice()
//     }
// }

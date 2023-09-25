use std::iter::{self, zip};

use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    arithmetic::{eval_polynomial, parallelize},
    dev::metadata::Gate,
    poly::{
        commitment::{Blind, Params},
        LagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use self::committed::Committed;

use super::{
    constraints::{paired::Paired, polynomial::PolynomialRef, Data},
    ProvingKey,
};

pub(super) mod advice;
pub(super) mod committed;
pub(super) mod compressed_verifier;
pub(super) mod instance;
pub(super) mod lookup;

/// An `Accumulator` contains the entirety of the IOP transcript,
/// including commitments and verifier challenges.
#[derive(Debug, Clone, PartialEq)]
pub struct Accumulator<C: CurveAffine> {
    pub instance: instance::Transcript<C>,
    pub advice: advice::Transcript<C>,
    pub lookups: Vec<lookup::Transcript<C>>,
    pub beta: compressed_verifier::Transcript<C>,

    pub ys: Vec<C::Scalar>,

    // Error value for all constraints
    pub error: C::Scalar,
}

impl<C: CurveAffine> Accumulator<C> {
    pub fn fold<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        pk: &ProvingKey<C>,
        acc0: Self,
        acc1: Self,
        transcript: &mut T,
    ) -> Self {
        let paired_data = Paired::<'_, C::Scalar>::new_data(pk, &acc0, &acc1);

        let full_constraint = paired_data.full_constraint(pk.cs.gates(), pk.cs.lookups());

        let error_poly = Paired::<'_, C::Scalar>::evaluate_compressed_polynomial(
            full_constraint,
            pk.usable_rows.clone(),
            pk.num_rows,
        );

        for coef in &error_poly {
            let _ = transcript.write_scalar(*coef);
        }
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();
        Self::merge(acc0, acc1, error_poly, alpha)
    }

    fn merge(acc0: Self, acc1: Self, error_poly: Vec<C::Scalar>, alpha: C::Scalar) -> Self {
        let instance = instance::Transcript::merge(alpha, acc0.instance, acc1.instance);
        let advice = advice::Transcript::merge(alpha, acc0.advice, acc1.advice);
        let lookups = zip(acc0.lookups.into_iter(), acc1.lookups.into_iter())
            .map(|(lookup0, lookup1)| lookup::Transcript::merge(alpha, lookup0, lookup1))
            .collect();
        let beta = compressed_verifier::Transcript::merge(alpha, acc0.beta, acc1.beta);

        let ys = zip(acc0.ys.into_iter(), acc1.ys.into_iter())
            .map(|(y0, y1)| y0 + alpha * (y1 - y0))
            .collect();

        let error = eval_polynomial(&error_poly, alpha);

        let error0 = eval_polynomial(&error_poly, C::Scalar::ZERO);
        let error1 = eval_polynomial(&error_poly, C::Scalar::ONE);

        assert_eq!(error0, acc0.error);
        assert_eq!(error1, acc1.error);

        Self {
            instance,
            advice,
            lookups,
            beta,
            ys,
            error,
        }
    }

    pub fn decide<'params, P: Params<'params, C>>(
        params: &P,
        pk: &ProvingKey<C>,
        acc: &Self,
    ) -> bool {
        let committed_iter: Vec<&Committed<C>> = acc
            .instance
            .committed
            .iter()
            .chain(&acc.advice.committed)
            .chain([&acc.beta.beta, &acc.beta.beta_shift].into_iter())
            .chain(
                acc.lookups
                    .iter()
                    .flat_map(|lookup| [&lookup.m, &lookup.g, &lookup.h].into_iter()),
            )
            .collect();

        let committed_ok = committed_iter.iter().all(|c| c.decide(params));

        let error_vec = Self::error_vector(pk, acc);

        let error = error_vec
            .iter()
            .zip(acc.beta.beta.values.iter())
            .fold(C::Scalar::ZERO, |acc, (e, b)| acc + (*e * b));

        let error_ok = error == acc.error;

        committed_ok && error_ok
    }

    pub fn error_vector(pk: &ProvingKey<C>, acc: &Self) -> Polynomial<C::Scalar, LagrangeCoeff> {
        let lagrange_data = Data::<PolynomialRef<'_, C::Scalar, LagrangeCoeff>>::new(&pk, &acc);

        let full_constraint = lagrange_data.full_constraint_no_beta(pk.cs.gates(), pk.cs.lookups());

        let mut error = pk.domain.empty_lagrange();
        parallelize(&mut error, |value, start| {
            for (i, v) in value.iter_mut().enumerate() {
                let row_idx = i + start;
                *v = full_constraint.evaluate(
                    &|c| c,
                    &|challenge| *challenge.value,
                    &|fixed| fixed.column[fixed.row_idx(row_idx, pk.num_rows)],
                    &|witness| witness.column[witness.row_idx(row_idx, pk.num_rows)],
                    &|e| -e,
                    &|a, b| a + b,
                    &|a, b| a * b,
                );
            }
        });
        error
    }
}

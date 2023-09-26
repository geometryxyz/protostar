use std::iter::zip;

use super::committed::{commit_transparent, Committed};
use crate::{
    arithmetic::parallelize,
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use crate::{
    poly::{empty_lagrange, LagrangeCoeff},
    protostar::keygen::ProvingKey,
};
use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;

/// Once all instance, advice and lookup witnesses have been sent to the verifier,
/// the Beta transcript allows all constraints to be batched into a single one.
/// The verifer sends a random value beta, and the prover commits to the vector of powers of beta.
/// The constraint to be checked becomes ∑ᵢ βᵢ ⋅ Gᵢ.
#[derive(PartialEq, Debug, Clone)]
pub struct Transcript<C: CurveAffine> {
    pub beta: Committed<C>,
    pub error: Committed<C>,
}

impl<C: CurveAffine> Transcript<C> {
    /// Runs the final IOP protocol to generate beta,
    /// and commit to the vector with the powers of beta.
    pub fn new<'params, P: Params<'params, C>, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        params: &P,
        transcript: &mut T,
    ) -> Self {
        let n = params.n();

        let beta = *transcript.squeeze_challenge_scalar::<C::Scalar>();

        // Vector of powers of `beta`
        let mut beta_values = empty_lagrange(n as usize);
        parallelize(&mut beta_values, |o, start| {
            let mut cur = beta.pow_vartime(&[start as u64]);
            for v in o.iter_mut() {
                *v = cur;
                cur *= &beta;
            }
        });

        // No need to blind since the contents are known by the verifier
        let committed = commit_transparent(params, beta_values, transcript);

        let error = Committed {
            values: empty_lagrange(n as usize),
            commitment: C::identity(),
            blind: Blind(C::Scalar::ZERO),
        };

        Self {
            beta: committed,
            error,
        }
    }

    pub(super) fn merge(alpha: C::Scalar, transcript0: Self, transcript1: Self) -> Self {
        let beta0 = transcript0.beta.values[1];
        let beta1 = transcript1.beta.values[1];
        let beta = Committed::merge(alpha, transcript0.beta.clone(), transcript1.beta.clone());

        // The error is given by e = r⋅a - b, where r is a challenge and a,b are vectors.
        // More precisely in this context,
        // - r is beta
        // - a is the vector of powers of beta
        // - b is a shifted by 1, s.t. b[i-1] = a[i] (only check from 1 onwards)
        //
        // We are given two transcripts, where
        // - e0 = r0⋅a0 - b0
        // - e1 = r1⋅a1 - b1
        //
        // The error polynomial is given by
        // e(t) = ((1-t)⋅r0 + t⋅r1)⋅((1-t)⋅a0 + t⋅a1) - ((1-t)⋅b0 + t⋅b1)
        // It is (almost) trivial to see, that
        // e(t) = (1-t)⋅e0 + t⋅e1 + (1-t)t⋅[(r1-r0)⋅a0 + (r0-r1)⋅a1]
        // Therefore, the new error is given by evaluating the above polynomial in t = alpha.
        let error = transcript0.error * (C::Scalar::ONE - alpha)
            + transcript1.error * alpha
            + (transcript0.beta * (beta1 - beta0) + transcript1.beta * (beta0 - beta1))
                * (alpha * (C::Scalar::ONE - alpha));

        Self { beta, error }
    }
}

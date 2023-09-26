use std::ops::{Add, Mul};

use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    arithmetic::parallelize,
    poly::{
        commitment::{self, Blind, Params},
        LagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

/// Represents a committed column sent that the verifier can query.
#[derive(PartialEq, Debug, Clone)]
pub struct Committed<C: CurveAffine> {
    pub values: Polynomial<C::Scalar, LagrangeCoeff>,
    pub commitment: C,
    pub blind: Blind<C::Scalar>,
}

impl<C: CurveAffine> Add for Committed<C> {
    type Output = Committed<C>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            values: self.values + &rhs.values,
            commitment: (self.commitment + rhs.commitment).to_affine(),
            blind: self.blind + rhs.blind,
        }
    }
}

impl<C: CurveAffine> Mul<C::Scalar> for Committed<C> {
    type Output = Committed<C>;

    fn mul(self, rhs: C::Scalar) -> Self::Output {
        Self {
            values: self.values * rhs,
            commitment: (self.commitment * rhs).to_affine(),
            blind: self.blind * rhs,
        }
    }
}

impl<C: CurveAffine> Committed<C> {
    /// Compute the linear combination (1−α)⋅ c₀ + α⋅c₁
    pub(super) fn merge(alpha: C::Scalar, committed0: Self, committed1: Self) -> Self {
        committed0 * (C::Scalar::ONE - alpha) + committed1 * alpha
    }

    /// Checks whether the commitment is valid with regards to the underlying column
    pub(super) fn decide<'params, P: Params<'params, C>>(&self, params: &P) -> bool {
        let commitment = params.commit_lagrange(&self.values, self.blind).to_affine();
        debug_assert_eq!(commitment, self.commitment);
        commitment == self.commitment
    }
}

/// Given a set of columns to be sent to the verifier, compute their commitments and write them to transcript.
/// Commitments are blinded.
pub fn batch_commit<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    I: Iterator<Item = Polynomial<C::Scalar, LagrangeCoeff>>,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    columns: I,
    mut rng: R,
    transcript: &mut T,
) -> Vec<Committed<C>> {
    let columns: Vec<_> = columns.collect();

    let blinds: Vec<_> = columns
        .iter()
        .map(|_| Blind(C::Scalar::random(&mut rng)))
        .collect();
    let commitments_projective: Vec<_> = columns
        .iter()
        .zip(blinds.iter())
        .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
        .collect();
    let mut commitments_affine = vec![C::identity(); commitments_projective.len()];
    C::CurveExt::batch_normalize(&commitments_projective, &mut commitments_affine);

    for commitment in &commitments_affine {
        let _ = transcript.write_point(*commitment);
    }

    columns
        .into_iter()
        .zip(commitments_affine.into_iter())
        .zip(blinds.into_iter())
        .map(|((values, commitment), blind)| Committed {
            values,
            commitment,
            blind,
        })
        .collect()
}

/// Given a set of columns to be sent to the verifier, compute their commitments and write them to transcript.
/// Commitments are transparent using a default blinding value.
pub fn batch_commit_transparent<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    I: Iterator<Item = Polynomial<C::Scalar, LagrangeCoeff>>,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    columns: I,
    transcript: &mut T,
) -> Vec<Committed<C>> {
    let columns: Vec<_> = columns.collect();

    let blinds: Vec<_> = columns
        .iter()
        .map(|_| Blind(C::Scalar::default()))
        .collect();
    let commitments_projective: Vec<_> = columns
        .iter()
        .zip(blinds.iter())
        .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
        .collect();
    let mut commitments_affine = vec![C::identity(); commitments_projective.len()];
    C::CurveExt::batch_normalize(&commitments_projective, &mut commitments_affine);

    for commitment in &commitments_affine {
        let _ = transcript.write_point(*commitment);
    }

    columns
        .into_iter()
        .zip(commitments_affine.into_iter())
        .zip(blinds.into_iter())
        .map(|((values, commitment), blind)| Committed {
            values,
            commitment,
            blind,
        })
        .collect()
}

/// Compute a single blinded commitment and write it to the transcript
pub fn commit<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    values: Polynomial<C::Scalar, LagrangeCoeff>,
    mut rng: R,
    transcript: &mut T,
) -> Committed<C> {
    let blind = Blind(C::Scalar::random(&mut rng));
    let commitment = params.commit_lagrange(&values, blind).to_affine();

    let _ = transcript.write_point(commitment);
    Committed {
        values,
        commitment,
        blind,
    }
}

/// Compute a single transparent commitment and write it to the transcript
pub fn commit_transparent<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    values: Polynomial<C::Scalar, LagrangeCoeff>,
    transcript: &mut T,
) -> Committed<C> {
    let blind = Blind(C::Scalar::default());
    let commitment = params.commit_lagrange(&values, blind).to_affine();

    let _ = transcript.write_point(commitment);
    Committed {
        values,
        commitment,
        blind,
    }
}

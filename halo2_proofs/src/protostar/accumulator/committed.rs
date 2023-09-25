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

#[derive(PartialEq, Debug, Clone)]
pub struct Committed<C: CurveAffine> {
    pub values: Polynomial<C::Scalar, LagrangeCoeff>,
    pub commitment: C,
    pub blind: Blind<C::Scalar>,
}

impl<C: CurveAffine> Committed<C> {
    pub(super) fn fold(alpha: C::Scalar, committed0: Self, committed1: Self) -> Self {
        let values = {
            let tmp = committed1.values - &committed0.values;
            let tmp = tmp * alpha;
            tmp + &committed0.values
        };
        let commitment = ((committed1.commitment - committed0.commitment) * alpha
            + &committed0.commitment)
            .to_affine();
        let blind = (committed1.blind - committed0.blind) * alpha + committed0.blind;
        Self {
            values,
            commitment,
            blind,
        }
    }

    pub(super) fn decide<'params, P: Params<'params, C>>(&self, params: &P) -> bool {
        let commitment = params.commit_lagrange(&self.values, self.blind).to_affine();
        debug_assert_eq!(commitment, self.commitment);
        commitment == self.commitment
    }
}

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

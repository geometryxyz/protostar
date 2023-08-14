use std::iter::zip;

use crate::{
    arithmetic::parallelize,
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        Polynomial,
    },
    transcript::{EncodedChallenge, Transcript, TranscriptWrite},
};
use crate::{
    poly::{empty_lagrange, LagrangeCoeff},
    protostar::keygen::ProvingKey,
};
use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;

/// Transcript for the "Compressed-Verifier" protocol
/// allowing the constraints over all rows to be compressed to a single one.
/// TODO(@adr1anh): Implement variant where we commit to two vector of size sqrt(n).
/// It is currently unsupported since the commitment scheme
/// only allows for commitments of vectors of size n.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedVerifierTranscript<C: CurveAffine> {
    beta_poly: Polynomial<C::Scalar, LagrangeCoeff>,
    beta_commitment: C,
    beta_blind: Blind<C::Scalar>,
}

/// Runs the final IOP protocol to generate beta,
/// and commit to the vector with the powers of beta.
pub fn create_compressed_verifier_transcript<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    transcript: &mut T,
) -> CompressedVerifierTranscript<C> {
    let n = params.n();

    let beta = *transcript.squeeze_challenge_scalar::<C::Scalar>();

    // Vector of powers of `beta`
    let mut beta_poly = empty_lagrange(n as usize);
    parallelize(&mut beta_poly, |o, start| {
        let mut cur = beta.pow_vartime(&[start as u64]);
        for v in o.iter_mut() {
            *v = cur;
            cur *= &beta;
        }
    });

    let beta_blind = Blind::default();
    let beta_commitment = params.commit_lagrange(&beta_poly, beta_blind).to_affine();

    let _ = transcript.write_point(beta_commitment);
    CompressedVerifierTranscript {
        beta_poly,
        beta_commitment,
        beta_blind,
    }
}

impl<C: CurveAffine> CompressedVerifierTranscript<C> {
    pub fn beta(&self) -> C::Scalar {
        self.beta_poly[0]
    }

    pub fn beta_poly(&self) -> &Polynomial<C::Scalar, LagrangeCoeff> {
        &self.beta_poly
    }

    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        std::iter::empty()
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        std::iter::empty()
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        std::iter::once(&self.beta_poly)
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        std::iter::once(&mut self.beta_poly)
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        std::iter::once(&self.beta_commitment)
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        std::iter::once(&mut self.beta_commitment)
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        std::iter::once(&self.beta_blind)
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        std::iter::once(&mut self.beta_blind)
    }
}

// TODO(@adr1anh): Below is the code for the sqrt beta technique.
// We don't use it now because we can't commit to vectors of size sqrt(n) with the current params.

// pub struct CompressedVerifierTranscript<C: CurveAffine> {
//     betas_polys: [Polynomial<C::Scalar, LagrangeCoeff>; 2],
//     betas_commitments: [C; 2],

//     k_sqrt: u32,
//     n_sqrt_mask: u64,
// }
// pub struct CompressedVerifierTranscript<C: CurveAffine> {
//     betas_polys: [Polynomial<C::Scalar, LagrangeCoeff>; 2],
//     betas_commitments: [C; 2],

//     k_sqrt: u32,
//     n_sqrt_mask: u64,
// }

// pub fn create_compressed_verifier_transcript<
//     Scheme: CommitmentScheme,
//     E: EncodedChallenge<Scheme::Curve>,
//     T: TranscriptWrite<Scheme::Curve, E>,
// >(
//     params: &Scheme::ParamsProver,
//     pk: &ProvingKey<Scheme::Curve>,
//     transcript: &mut T,
// ) -> CompressedVerifierTranscript<Scheme::Curve> {
//     let k_sqrt = pk.log2_sqrt_num_rows();
//     let n_sqrt = 1 << k_sqrt;
//     let n_sqrt_mask = n_sqrt - 1;

//     let beta0 = *transcript.squeeze_challenge_scalar::<Scheme::Scalar>();
//     let beta1 = beta0.pow_vartime([n_sqrt]);

//     let mut betas_polys = [(); 2].map(|_| empty_lagrange(n_sqrt as usize));

//     parallelize(&mut betas_polys[0], |o, start| {
//         let mut cur = beta0.pow_vartime(&[start as u64]);
//         for v in o.iter_mut() {
//             *v = cur;
//             cur *= &beta0;
//         }
//     });

//     parallelize(&mut betas_polys[1], |o, start| {
//         let mut cur = beta1.pow_vartime(&[start as u64]);
//         for v in o.iter_mut() {
//             *v = cur;
//             cur *= &beta1;
//         }
//     });

//     let beta0_commitment = params
//         .commit_lagrange(&betas_polys[0], Blind::default())
//         .to_affine();
//     let beta1_commitment = params
//         .commit_lagrange(&betas_polys[1], Blind::default())
//         .to_affine();

//     let _ = transcript.write_point(beta0_commitment);
//     let _ = transcript.write_point(beta1_commitment);

//     CompressedVerifierTranscript {
//         betas_polys,
//         betas_commitments: [beta0_commitment, beta1_commitment],
//         k_sqrt,
//         n_sqrt_mask,
//     }
// }

// impl<C: CurveAffine> CompressedVerifierTranscript<C> {
//     pub fn beta0(&self) -> C::Scalar {
//         self.betas_polys[0][0]
//     }
//     pub fn beta1(&self) -> C::Scalar {
//         self.betas_polys[1][0]
//     }

//     pub fn beta0_poly(&self) -> &Polynomial<C::Scalar, LagrangeCoeff> {
//         &self.betas_polys[0]
//     }
//     pub fn beta1_poly(&self) -> &Polynomial<C::Scalar, LagrangeCoeff> {
//         &self.betas_polys[1]
//     }

//     pub fn betas_for_row(&self, row_index: usize) -> (C::Scalar, C::Scalar) {
//         let i0 = row_index & self.n_sqrt_mask as usize;
//         let i1 = row_index >> self.k_sqrt;
//         (self.betas_polys[0][i0], self.betas_polys[1][i1])
//     }

//     pub fn boolean_linear_combination(&mut self, other: &Self, alpha: C::Scalar) {
//         let _ = zip(self.betas_polys.iter_mut(), other.betas_polys.iter())
//             .map(|(lhs, rhs)| lhs.boolean_linear_combination(rhs, alpha));

//         let _ = zip(
//             self.betas_commitments.iter_mut(),
//             other.betas_commitments.iter(),
//         )
//         .map(|(lhs, rhs)| *lhs = ((*rhs - *lhs) * alpha + *lhs).to_affine());
//     }
// }

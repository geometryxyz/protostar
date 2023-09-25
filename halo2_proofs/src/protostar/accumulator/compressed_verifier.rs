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

#[derive(PartialEq, Debug, Clone)]
pub struct Transcript<C: CurveAffine> {
    pub beta: Committed<C>,
    pub beta_shift: Committed<C>,
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

        // Prover and Verifier compute shifted beta column to linearize the constraint
        //      beta[i+1] = beta[i] * beta
        // We compute beta_shift s.t.
        //      beta_shift[i] = beta[i] * beta = beta[i+1]
        // and the constraints becomes
        //      beta[0] = beta
        //      beta_shift[i] = beta[i] * beta
        let beta_shift_values = committed.values.clone() * beta;
        let beta_shift_commitment = (committed.commitment * beta).to_affine();
        let beta_shift_blind = committed.blind * beta;

        let committed_shift = Committed {
            values: beta_shift_values,
            commitment: beta_shift_commitment,
            blind: beta_shift_blind,
        };

        Self {
            beta: committed,
            beta_shift: committed_shift,
        }
    }

    pub(super) fn merge(alpha: C::Scalar, transcript0: Self, transcript1: Self) -> Self {
        let beta = Committed::fold(alpha, transcript0.beta, transcript1.beta);
        let beta_shift = Committed::fold(alpha, transcript0.beta_shift, transcript1.beta_shift);

        Self { beta, beta_shift }
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

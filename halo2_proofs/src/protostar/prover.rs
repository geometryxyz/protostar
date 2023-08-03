use std::{iter::zip, marker::PhantomData};

use ff::{Field, FromUniformBytes};
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    arithmetic::{lagrange_interpolate, powers},
    plonk::{Circuit, Error, FixedQuery},
    poly::{
        commitment::{CommitmentScheme, Prover},
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::{
    error_check::{Accumulator, AccumulatorInstance, AccumulatorWitness, GateEvaluationCache},
    gate::{self, Expr},
    keygen::ProvingKey,
    witness::{create_advice_transcript, create_instance_polys},
};

/// TODO(@adr1anh): This creates an accumulator actually
fn create_proof<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Scalar>,
    circuit: &ConcreteCircuit,
    instances: &[&[Scheme::Scalar]],
    mut rng: R,
    transcript: &mut T,
) -> Result<Accumulator<Scheme::Scalar>, Error>
where
    Scheme::Scalar: FromUniformBytes<64>,
{
    // Hash verification key into transcript
    // pk.vk.hash_into(transcript)?;

    // Create polynomials from instance and add to transcript
    let instance_polys =
        create_instance_polys::<Scheme, E, T>(&params, pk.cs(), instances, transcript)?;
    // Create advice columns and add to transcript
    let advice_transcript = create_advice_transcript::<Scheme, E, R, T, ConcreteCircuit>(
        params,
        pk.cs(),
        circuit,
        instances,
        rng,
        transcript,
    )?;

    // Protostar specific IOP stuff
    let n_sqrt = 1 << pk.log2_sqrt_num_rows();
    let beta = transcript.squeeze_challenge_scalar::<Scheme::Scalar>();
    let beta_sqrt = beta.pow_vartime([n_sqrt as u64]);
    // TODO(@adr1anh): Commit to beta vectors

    let y = transcript.squeeze_challenge_scalar::<Scheme::Scalar>();

    // TODO(@adr1anh): Add advice commitments
    let new_instance =
        AccumulatorInstance::new(pk, advice_transcript.challenges, instance_polys, *beta, *y);

    // TODO(@adr1anh): Provide beta only
    let new_witness = AccumulatorWitness::new(
        pk,
        advice_transcript.advice_polys,
        advice_transcript.advice_blinds,
        powers(*beta).take(n_sqrt).collect(),
        powers(beta_sqrt).take(n_sqrt).collect(),
    );

    Ok(Accumulator::new(new_instance, new_witness))
}

#[cfg(test)]
mod tests {
    use crate::{
        plonk::keygen_vk,
        poly::{
            self,
            commitment::ParamsProver,
            ipa::{commitment::IPACommitmentScheme, multiopen::ProverIPA},
        },
        protostar::shuffle::MyCircuit,
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use core::num;

    use crate::plonk::{sealed::Phase, ConstraintSystem, FirstPhase};
    use crate::{halo2curves::pasta::pallas, plonk::sealed::SealedPhase};

    use super::*;
    use crate::plonk::sealed;
    use crate::plonk::Expression;
    use rand_core::{OsRng, RngCore};

    #[test]
    fn test_expression_conversion() {
        let mut rng = OsRng;
        const W: usize = 4;
        const H: usize = 32;
        const K: u32 = 8;
        const N: usize = 64; // could be 33?
        let circuit1 = MyCircuit::<_, W, H>::rand(&mut rng);
        let circuit2 = MyCircuit::<_, W, H>::rand(&mut rng);
        let circuit3 = MyCircuit::<_, W, H>::rand(&mut rng);

        let params = poly::ipa::commitment::ParamsIPA::<_>::new(K);
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let pk = ProvingKey::new(N, &circuit1).unwrap();
        let r1 = create_proof::<IPACommitmentScheme<pallas::Affine>, _, _, _, _>(
            &params,
            &pk,
            &circuit1,
            &[],
            rng,
            &mut transcript,
        );
        let r2 = create_proof::<IPACommitmentScheme<pallas::Affine>, _, _, _, _>(
            &params,
            &pk,
            &circuit2,
            &[],
            rng,
            &mut transcript,
        );
        let r3 = create_proof::<IPACommitmentScheme<pallas::Affine>, _, _, _, _>(
            &params,
            &pk,
            &circuit3,
            &[],
            rng,
            &mut transcript,
        );
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());

        let mut acc = r1.unwrap();
        let acc2 = r2.unwrap();
        let acc3 = r3.unwrap();

        acc = acc.fold(&pk, acc2);
        acc = acc.fold(&pk, acc3);

        assert!(!acc.instance().ys.is_empty());
        // assert!(e.first().unwrap().is_zero_vartime());
        // assert!(e.last().unwrap().is_zero_vartime());

        // for e_j in e.iter() {
        //     transcript.write_scalar(*e_j).unwrap();
        // }
        // assert!(!e.is_empty());
    }
}

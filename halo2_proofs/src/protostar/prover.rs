use std::{iter::zip, marker::PhantomData};

use ff::{Field, FromUniformBytes};
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    arithmetic::{lagrange_interpolate, parallelize, powers},
    plonk::{Circuit, Error, FixedQuery},
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        empty_lagrange, LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::{
    error_check::Accumulator,
    gate::{self, Expr},
    keygen::ProvingKey,
    transcript::{
        advice::create_advice_transcript,
        compressed_verifier::{
            create_compressed_verifier_transcript, CompressedVerifierTranscript,
        },
        instance::create_instance_transcript,
        lookup::{self, create_lookup_transcript},
    },
};

/// Runs the IOP until the decision phase, and returns an `Accumulator` containing the entirety of the transcript.
/// The result can be folded into another `Accumulator`.
fn create_accumulator<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
    ConcreteCircuit: Circuit<C::Scalar>,
>(
    params: &P,
    pk: &ProvingKey<C>,
    circuit: &ConcreteCircuit,
    instances: &[&[C::Scalar]],
    mut rng: R,
    transcript: &mut T,
) -> Result<Accumulator<C>, Error> {
    // Hash verification key into transcript
    // pk.vk.hash_into(transcript)?;

    // Add public inputs/outputs to the transcript, and convert them to `Polynomial`s
    let instance_transcript = create_instance_transcript(params, pk.cs(), instances, transcript)?;

    // Run multi-phase IOP section to generate all `Advice` columns
    let advice_transcript = create_advice_transcript(
        params,
        pk,
        circuit,
        &instance_transcript.instance_polys,
        &mut rng,
        transcript,
    )?;

    // Extract the gate challenges from the advice transcript
    let challenges = advice_transcript.challenges();

    // Run the 2-round logUp IOP for all lookup arguments
    let lookup_transcript = create_lookup_transcript(
        params,
        pk,
        &advice_transcript.advice_polys,
        &instance_transcript.instance_polys,
        &challenges,
        rng,
        transcript,
    );

    // Generate random column(s) to multiply each constraint
    // so that we can compress them to a single constraint
    let compressed_verifier_transcript = create_compressed_verifier_transcript(params, transcript);

    // Challenge for the RLC of all constraints (all gates and all lookups)
    let y = *transcript.squeeze_challenge_scalar::<C::Scalar>();

    Ok(Accumulator::new(
        pk,
        instance_transcript,
        advice_transcript,
        lookup_transcript,
        compressed_verifier_transcript,
        y,
    ))
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

    use crate::plonk::{sealed::Phase, ConstraintSystem, FirstPhase};
    use crate::{halo2curves::pasta::pallas, plonk::sealed::SealedPhase};

    use super::*;
    use crate::plonk::sealed;
    use crate::plonk::Expression;
    use rand_core::{OsRng, RngCore};

    #[test]
    fn test_accumulation() {
        let mut rng = OsRng;
        const W: usize = 4;
        const H: usize = 32;
        const K: u32 = 8;
        let params = poly::ipa::commitment::ParamsIPA::<pallas::Affine>::new(K);

        let circuit1 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);
        let circuit2 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);
        let circuit3 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);

        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let pk = ProvingKey::new(&params, &circuit1).unwrap();
        let mut acc =
            create_accumulator(&params, &pk, &circuit1, &[], &mut rng, &mut transcript).unwrap();

        let acc2 =
            create_accumulator(&params, &pk, &circuit2, &[], &mut rng, &mut transcript).unwrap();
        acc.fold(&pk, acc2, &mut transcript);

        let acc3 =
            create_accumulator(&params, &pk, &circuit3, &[], &mut rng, &mut transcript).unwrap();
        acc.fold(&pk, acc3, &mut transcript);

        let acc4 =
            create_accumulator(&params, &pk, &circuit3, &[], &mut rng, &mut transcript).unwrap();
        acc.fold(&pk, acc4, &mut transcript);
        assert!(acc.decide(&params, &pk));
    }
}

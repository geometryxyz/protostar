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

use super::{accumulator, keygen::ProvingKey};

/// Runs the IOP until the decision phase, and returns an `Accumulator` containing the entirety of the transcript.
/// The result can be folded into another `Accumulator`.
pub fn create_accumulator<
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
) -> Result<accumulator::Accumulator<C>, Error> {
    // Hash verification key into transcript
    // pk.vk.hash_into(transcript)?;

    // Add public inputs/outputs to the transcript, and convert them to `Polynomial`s
    // and run multi-phase IOP section to generate all `Advice` columns
    let gate =
        accumulator::gate::Transcript::new(params, pk, circuit, instances, &mut rng, transcript)?;

    // Run the 2-round logUp IOP for all lookup arguments
    let lookups = accumulator::lookup::new(params, pk, &gate, &mut rng, transcript);

    // Generate random column(s) to multiply each constraint
    // so that we can compress them to a single constraint
    let beta = accumulator::compressed_verifier::Transcript::new(params, transcript);

    // Challenge for the RLC of all constraints (all gates and all lookups)
    let y = *transcript.squeeze_challenge_scalar::<C::Scalar>();

    let ys = powers(y).take(pk.num_folding_constraints()).collect();

    Ok(accumulator::Accumulator {
        gate,
        lookups,
        beta,
        ys,
        error: C::Scalar::ZERO,
    })
}

use std::{iter::zip, marker::PhantomData};

use ff::{Field, FromUniformBytes};
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    arithmetic::lagrange_interpolate,
    plonk::{Circuit, Error, FixedQuery},
    poly::{
        commitment::{CommitmentScheme, Prover},
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::{
    error_check::{AccumulatorInstance, AccumulatorWitness, GateEvaluationCache},
    gate::{self, Expr},
    keygen::{CircuitData, ProvingKey},
    witness::{create_advice_transcript, create_instance_polys},
};

struct WitnessTranscript<F: Field> {
    challenges: Vec<F>,
    instance: Vec<Polynomial<F, LagrangeCoeff>>,
    advice: Vec<Polynomial<F, LagrangeCoeff>>,
    advice_blinds: Vec<F>,
}

///
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
) -> Result<(), Error>
where
    Scheme::Scalar: FromUniformBytes<64>,
{
    // Hash verification key into transcript
    // pk.vk.hash_into(transcript)?;

    // Create polynomials from instance and add to transcript
    let instance_polys =
        create_instance_polys::<Scheme, E, T>(&params, &pk.circuit_data.cs, instances, transcript)?;
    // Create advice columns and add to transcript
    let advice_transcript = create_advice_transcript::<Scheme, E, R, T, ConcreteCircuit>(
        params,
        &pk.circuit_data.cs,
        circuit,
        instances,
        rng,
        transcript,
    )?;

    let n = pk.circuit_data.usable_rows.end as i32;

    let new_instance = AccumulatorInstance::new(pk, advice_transcript.challenges, instance_polys);
    let acc_instance = AccumulatorInstance::new_empty(pk);
    // let acc_instance = new_instance.clone();

    let new_witness = AccumulatorWitness::new(
        advice_transcript.advice_polys,
        advice_transcript.advice_blinds,
    );
    let acc_witness = AccumulatorWitness::new_empty(pk);
    // let acc_witness = new_witness.clone();

    let num_extra_evaluations = 3;
    let mut gate_caches: Vec<_> = pk
        .gates
        .iter()
        .map(|gate| {
            GateEvaluationCache::new(gate, &acc_instance, &new_instance, num_extra_evaluations)
        })
        .collect();

    // for poly in gate_caches
    //     .iter()
    //     .flat_map(|gate_chache| gate_chache.gate.polys.iter())
    // {
    //     println!("{:?}", poly);
    // }

    let num_evals = pk.max_degree + 1 + num_extra_evaluations;

    // DEBUG: Evaluation points used to do lagrange interpolation
    let points = {
        let mut points = Vec::with_capacity(num_evals);
        points.push(Scheme::Scalar::ZERO);
        for i in 1..num_evals {
            points.push(points[i - 1] + Scheme::Scalar::ONE);
        }
        points
    };

    // eval_sums[gate_idx][constraint_idx][eval_idx]
    let mut eval_sums: Vec<_> = gate_caches
        .iter()
        .map(|gate_cache| gate_cache.gate_eval.clone())
        .collect();

    for row_idx in pk.circuit_data.usable_rows.clone().into_iter() {
        for (gate_idx, gate_cache) in gate_caches.iter_mut().enumerate() {
            let evals =
                gate_cache.evaluate(row_idx, n, &pk.circuit_data, &acc_witness, &new_witness);
            if let Some(evals) = evals {
                for (poly_idx, poly_evals) in evals.iter().enumerate() {
                    // DEBUG: Check that the last `num_extra_evaluations` coeffs are zero
                    let poly_points = &points[..poly_evals.len()];
                    let eval_coeffs = lagrange_interpolate(poly_points, poly_evals);
                    debug_assert_ne!(eval_coeffs.len(), 0);

                    for (existing, new) in
                        zip(eval_sums[gate_idx][poly_idx].iter_mut(), poly_evals.iter())
                    {
                        *existing += new;
                    }
                }
            }
        }
    }

    let eval_coeffs: Vec<_> = eval_sums
        .iter()
        .flat_map(|gate_evals| {
            gate_evals.iter().map(|poly_evals| -> Vec<Scheme::Scalar> {
                let poly_points = &points[..poly_evals.len()];
                lagrange_interpolate(poly_points, poly_evals)
            })
        })
        .collect();
    assert_ne!(eval_coeffs.len(), 0);

    Ok(())
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
        let circuit = MyCircuit::<_, W, H>::rand(&mut rng);

        let params = poly::ipa::commitment::ParamsIPA::<_>::new(K);
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let cd = CircuitData::new(N, &circuit).unwrap();
        let pk = ProvingKey::new(&cd).unwrap();
        let r = create_proof::<IPACommitmentScheme<pallas::Affine>, _, _, _, _>(
            &params,
            &pk,
            &circuit,
            &[],
            rng,
            &mut transcript,
        );
        // let vk = keygen_vk(&params, &circuit).unwrap();
        // let data = create_advice_transcript::<IPACommitmentScheme<pallas::Affine>, _, _, _, _>(
        //     &params,
        //     &vk.cs(),
        //     &circuit,
        //     &[],
        //     rng,
        //     &mut transcript,
        // );
        assert!(r.is_ok());
    }
}

use std::marker::PhantomData;

use ff::{Field, FromUniformBytes};
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    plonk::{Circuit, Error, FixedQuery},
    poly::{
        commitment::{CommitmentScheme, Prover},
        LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::{
    error_check::{AccumulatorInstance, AccumulatorWitness, GateEvaluationCache},
    gate::Expr,
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

    // Create polynomials from instance, and
    let instance_polys =
        create_instance_polys::<Scheme, E, T>(&params, &pk.circuit_data.cs, instances, transcript)?;
    let advice_transcript = create_advice_transcript::<Scheme, E, R, T, ConcreteCircuit>(
        params,
        &pk.circuit_data.cs,
        circuit,
        instances,
        rng,
        transcript,
    )?;

    let fixed_polys = &pk.circuit_data.fixed;
    let selectors = &pk.circuit_data.selectors;

    let n = pk.circuit_data.usable_rows.end as i32;

    let new_instance = AccumulatorInstance::new(pk, advice_transcript.challenges, instance_polys);
    let acc_instance = new_instance.clone();
    //    AccumulatorInstance::new_empty(pk);

    let new_witness = AccumulatorWitness::new(
        advice_transcript.advice_polys,
        advice_transcript.advice_blinds,
    );
    let acc_witness = new_witness.clone();
    //     AccumulatorWitness::<Scheme::Scalar>::new_empty(n as usize, pk.circuit_data.n as usize);

    let mut gate_caches: Vec<_> = pk
        .gates
        .iter()
        .map(|gate| GateEvaluationCache::new(gate, &acc_instance, &new_instance, 1))
        .collect();

    for row_idx in pk.circuit_data.usable_rows.clone().into_iter() {
        for gate_cache in gate_caches.iter_mut() {
            let evals =
                gate_cache.evaluate(row_idx, n, &pk.circuit_data, &acc_witness, &new_witness);
            if let Some(evals) = evals {
                for eval in evals.iter() {
                    debug_assert!(eval[0].is_zero_vartime());
                    // debug_assert!(eval[1].is_zero_vartime());
                }
            }
        }
    }

    // for i in pk.circuit_data.usable_rows.clone().into_iter() {
    //     for gate in pk.circuit_data.cs.gates() {
    //         for poly in gate.polynomials() {
    //             let e: Expr<Scheme::Scalar> = poly.clone().into();
    //             let result = e.evaluate_lazy(
    //                 &|_slack| 1.into(),
    //                 &|constant| constant,
    //                 &|selector| {
    //                     if selectors[selector.0][i] {
    //                         Scheme::Scalar::ONE
    //                     } else {
    //                         Scheme::Scalar::ZERO
    //                     }
    //                 },
    //                 &|fixed| {
    //                     fixed_polys[fixed.column_index][get_rotation_idx(i, fixed.rotation(), n)]
    //                 },
    //                 &|advice| {
    //                     advice_polys[advice.column_index][get_rotation_idx(i, advice.rotation(), n)]
    //                 },
    //                 &|instance| {
    //                     instance_polys[instance.column_index]
    //                         [get_rotation_idx(i, instance.rotation(), n)]
    //                 },
    //                 &|challenge, power| challenges[challenge.index()].pow_vartime([power as u64]),
    //                 &|a| -a,
    //                 &|a, b| a + b,
    //                 &|a, b| a * b,
    //                 &|a, b| a * b,
    //                 &Scheme::Scalar::ZERO,
    //             );
    //             assert_eq!(result, Scheme::Scalar::ZERO);
    //         }
    //     }
    // }

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
        const N: usize = 1 << K;
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

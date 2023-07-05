use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        sealed::{self, Phase, SealedPhase},
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem, Error,
        FirstPhase, Fixed, FloorPlanner, Instance, ProvingKey, Selector, VerifyingKey,
    },
    poly::{
        self,
        commitment::{Blind, CommitmentScheme, Params, Prover},
        empty_lagrange_assigned, Basis, Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial,
        ProverQuery,
    },
    poly::{batch_invert_assigned, empty_lagrange},
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::{prime::PrimeCurveAffine, Curve};
use rand_core::RngCore;
use std::{
    collections::{BTreeSet, HashMap},
    ops::RangeTo,
};

///
struct WitnessCollection<'a, F: Field> {
    k: u32,
    current_phase: sealed::Phase,
    advice: &'a mut Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    challenges: &'a mut HashMap<usize, F>,
    instances: &'a [&'a [F]],
    usable_rows: RangeTo<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<'a, F: Field> SyncDeps for WitnessCollection<'a, F> {}

impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.instances
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| Value::known(*v))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Advice>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // let current_phase = self.current_phase.unwrap();
        // Ignore assignment of advice column in different phase than current one.
        if self.current_phase != column.column_type().phase {
            return Ok(());
        }

        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        *self
            .advice
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Fixed>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about advice columns here

        Ok(())
    }

    fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) -> Result<(), Error> {
        // We only care about advice columns here

        Ok(())
    }

    fn fill_from_row(
        &mut self,
        _: Column<Fixed>,
        _: usize,
        _: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.challenges
            .get(&challenge.index())
            .cloned()
            .map(Value::known)
            .unwrap_or_else(Value::unknown)
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

pub fn create_instance_polys<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    T: TranscriptWrite<Scheme::Curve, E>,
>(
    params: &Scheme::ParamsProver,
    cs: &ConstraintSystem<Scheme::Scalar>,
    instances: &[&[Scheme::Scalar]],
    transcript: &mut T,
) -> Result<Vec<Polynomial<Scheme::Scalar, LagrangeCoeff>>, Error> {
    if instances.len() != cs.num_instance_columns {
        return Err(Error::InvalidInstances);
    }
    let n = params.n() as usize;

    // TODO(@adr1anh): refactor into own function
    // generate polys for instance columns
    // NOTE(@adr1anh): In the case where the verifier does not query the instance,
    // we do not need to create a Lagrange polynomial of size n.
    let instance_polys = instances
        .iter()
        .map(|values| {
            let mut poly = empty_lagrange(n);

            if values.len() > (poly.len() - (cs.blinding_factors() + 1)) {
                return Err(Error::InstanceTooLarge);
            }
            for (poly, value) in poly.iter_mut().zip(values.iter()) {
                // The instance is part of the transcript
                // if !P::QUERY_INSTANCE {
                //     transcript.common_scalar(*value)?;
                // }
                transcript.common_scalar(*value)?;
                *poly = *value;
            }
            Ok(poly)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // // For large instances, we send a commitment to it and open it with PCS
    // if P::QUERY_INSTANCE {
    //     let instance_commitments_projective: Vec<_> = instance_polys
    //         .iter()
    //         .map(|poly| params.commit_lagrange(poly, Blind::default()))
    //         .collect();
    //     let mut instance_commitments =
    //         vec![Scheme::Curve::identity(); instance_commitments_projective.len()];
    //     <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
    //         &instance_commitments_projective,
    //         &mut instance_commitments,
    //     );
    //     let instance_commitments = instance_commitments;
    //     drop(instance_commitments_projective);

    //     for commitment in &instance_commitments {
    //         transcript.common_point(*commitment)?;
    //     }
    // }
    Ok(instance_polys)
}

/// Advice polynomials sent by the prover during the first phases of
/// the IOP protocol.
pub struct AdviceTranscript<F: Field> {
    pub challenges: Vec<F>,
    pub advice_polys: Vec<Polynomial<F, LagrangeCoeff>>,
    // blinding values for advice_polys, same length as advice_polys
    pub advice_blinds: Vec<Blind<F>>,
}

/// Runs the witness generation for the first phase of the protocol
/// TODO(@adr1anh): rename to generate advice
pub fn create_advice_transcript<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: Circuit<Scheme::Scalar>,
>(
    params: &Scheme::ParamsProver,
    cs: &ConstraintSystem<Scheme::Scalar>,
    circuit: &ConcreteCircuit,
    // raw instance columns
    instances: &[&[Scheme::Scalar]],
    mut rng: R,
    transcript: &mut T,
) -> Result<AdviceTranscript<Scheme::Scalar>, Error> {
    let n = params.n() as usize;

    let config = {
        let mut meta = ConstraintSystem::default();

        #[cfg(feature = "circuit-params")]
        let config = ConcreteCircuit::configure_with_params(&mut meta, circuit.params());
        #[cfg(not(feature = "circuit-params"))]
        let config = ConcreteCircuit::configure(&mut meta);
        config
    };

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &cs;

    // Synthesize the circuit over multiple iterations
    let (advice_polys, advice_blinds, challenges) = {
        let mut advice_assigned = vec![empty_lagrange_assigned(n); meta.num_advice_columns];
        let mut advice_polys = vec![empty_lagrange(n); meta.num_advice_columns];
        let mut advice_blinds = vec![Blind::default(); meta.num_advice_columns];
        let mut challenges = HashMap::<usize, Scheme::Scalar>::with_capacity(meta.num_challenges);

        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);

        // implements Assignment so that we can
        let mut witness = WitnessCollection {
            k: params.k(),
            current_phase: FirstPhase.to_sealed(),
            advice: &mut advice_assigned,
            instances,
            challenges: &mut challenges,
            // The prover will not be allowed to assign values to advice
            // cells that exist within inactive rows, which include some
            // number of blinding factors and an extra row for use in the
            // permutation argument.
            usable_rows: ..unusable_rows_start,
            _marker: std::marker::PhantomData,
        };

        // For each phase
        for current_phase in cs.phases() {
            witness.current_phase = current_phase;
            let column_indices = meta
                .advice_column_phase
                .iter()
                .enumerate()
                .filter_map(|(column_index, phase)| {
                    if current_phase == *phase {
                        Some(column_index)
                    } else {
                        None
                    }
                })
                .collect::<BTreeSet<_>>();

            // Synthesize the circuit to obtain the witness and other information.
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )?;

            let mut advice_values = batch_invert_assigned::<Scheme::Scalar>(
                witness
                    .advice
                    .iter()
                    .enumerate()
                    .filter_map(|(column_index, advice)| {
                        if column_indices.contains(&column_index) {
                            Some(advice.clone())
                        } else {
                            None
                        }
                    })
                    .collect(),
            );

            // Add blinding factors to advice columns
            for advice_values in &mut advice_values {
                for cell in &mut advice_values[unusable_rows_start..] {
                    *cell = Scheme::Scalar::random(&mut rng);
                }
            }

            // Compute commitments to advice column polynomials
            let blinds: Vec<_> = advice_values
                .iter()
                .map(|_| Blind(Scheme::Scalar::random(&mut rng)))
                .collect();
            let advice_commitments_projective: Vec<_> = advice_values
                .iter()
                .zip(blinds.iter())
                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                .collect();
            let mut advice_commitments =
                vec![Scheme::Curve::identity(); advice_commitments_projective.len()];
            <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                &advice_commitments_projective,
                &mut advice_commitments,
            );
            let advice_commitments = advice_commitments;
            drop(advice_commitments_projective);

            for commitment in &advice_commitments {
                transcript.write_point(*commitment)?;
            }
            for ((column_index, advice_values), blind) in
                column_indices.iter().zip(advice_values).zip(blinds)
            {
                advice_polys[*column_index] = advice_values;
                advice_blinds[*column_index] = blind;
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if current_phase == *phase {
                    let existing = witness
                        .challenges
                        .insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                }
            }
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice_polys, advice_blinds, challenges)
    };

    Ok(AdviceTranscript {
        advice_polys,
        advice_blinds,
        challenges,
    })
}

#[cfg(test)]

mod tests {
    use crate::{
        plonk::keygen_vk,
        poly::{
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
        let circuit = MyCircuit::<_, W, H>::rand(&mut rng);

        let params = poly::ipa::commitment::ParamsIPA::<_>::new(K);
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let vk = keygen_vk(&params, &circuit).unwrap();
        let data = create_advice_transcript::<IPACommitmentScheme<pallas::Affine>, _, _, _, _>(
            &params,
            &vk.cs(),
            &circuit,
            &[],
            &mut rng,
            &mut transcript,
        );
        assert!(data.is_ok());
    }
}

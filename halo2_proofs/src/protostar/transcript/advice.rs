use crate::{
    arithmetic::{eval_polynomial, powers, CurveAffine},
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        sealed::{self, Phase, SealedPhase},
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem, Error,
        FirstPhase, Fixed, FloorPlanner, Instance, Selector, VerifyingKey,
    },
    poly::{
        self,
        commitment::{Blind, CommitmentScheme, Params, Prover},
        empty_lagrange_assigned, Basis, Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial,
        ProverQuery,
    },
    poly::{batch_invert_assigned, empty_lagrange},
    protostar::keygen::ProvingKey,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::{prime::PrimeCurveAffine, Curve};
use rand_core::RngCore;
use std::{
    collections::{BTreeSet, HashMap},
    iter::zip,
    ops::RangeTo,
};

/// Advice polynomials sent by the prover during the first phases of
/// the IOP protocol.
pub struct AdviceTranscript<C: CurveAffine> {
    // Array of challenges and their powers
    // challenges[i][d] = challenge_i^{d+1}
    pub challenges: Vec<Vec<C::Scalar>>,
    pub advice_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub advice_commitments: Vec<C>,
    pub advice_blinds: Vec<Blind<C::Scalar>>,
}

/// Runs the witness generation for the first phase of the protocol
pub fn create_advice_transcript<
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
    instances: &Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    mut rng: R,
    transcript: &mut T,
) -> Result<AdviceTranscript<C>, Error> {
    let n = params.n() as usize;

    // TODO(@adr1anh): Can we cache the config in the `circuit_data`?
    // We don't apply selector optimization so it should remain the same as during the keygen.
    let config = {
        let mut meta = ConstraintSystem::default();

        #[cfg(feature = "circuit-params")]
        let config = ConcreteCircuit::configure_with_params(&mut meta, circuit.params());
        #[cfg(not(feature = "circuit-params"))]
        let config = ConcreteCircuit::configure(&mut meta);
        config
    };

    let meta = &pk.cs();

    // Synthesize the circuit over multiple iterations

    // TODO(@adr1anh): Use `circuit_data.num_advice_rows` to only allocate required number of rows
    let mut advice_assigned = vec![empty_lagrange_assigned(n); meta.num_advice_columns];
    let mut advice_polys = vec![empty_lagrange(n); meta.num_advice_columns];
    let mut advice_commitments = vec![C::identity(); meta.num_advice_columns];
    let mut advice_blinds = vec![Blind::default(); meta.num_advice_columns];
    let mut challenges = HashMap::<usize, C::Scalar>::with_capacity(meta.num_challenges);

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
    for current_phase in meta.phases() {
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

        let mut advice_values = batch_invert_assigned::<C::Scalar>(
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
                *cell = C::Scalar::random(&mut rng);
            }
        }

        // Compute commitments to advice column polynomials
        let blinds: Vec<_> = advice_values
            .iter()
            .map(|_| Blind(C::Scalar::random(&mut rng)))
            .collect();
        let advice_commitments_projective: Vec<_> = advice_values
            .iter()
            .zip(blinds.iter())
            .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
            .collect();
        let mut advice_commitments_affine =
            vec![C::identity(); advice_commitments_projective.len()];
        C::CurveExt::batch_normalize(
            &advice_commitments_projective,
            &mut advice_commitments_affine,
        );
        let advice_commitments_affine = advice_commitments_affine;
        drop(advice_commitments_projective);

        for commitment in &advice_commitments_affine {
            transcript.write_point(*commitment)?;
        }

        // Store advice columns in Assembly
        for (((column_index, advice_values), commitment), blind) in column_indices
            .iter()
            .zip(advice_values)
            .zip(advice_commitments_affine)
            .zip(blinds)
        {
            advice_polys[*column_index] = advice_values;
            advice_commitments[*column_index] = commitment;
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

    let challenge_degrees = pk.max_challenge_powers();
    let challenges = challenges
        .iter()
        .zip(challenge_degrees)
        .map(|(c, d)| powers(*c).skip(1).take(d).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    Ok(AdviceTranscript {
        challenges,
        advice_polys,
        advice_commitments,
        advice_blinds,
    })
}

impl<C: CurveAffine> AdviceTranscript<C> {
    pub fn challenges(&self) -> Vec<C::Scalar> {
        self.challenges.iter().map(|cs| cs[0]).collect()
    }

    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        self.challenges
            .iter()
            .flat_map(|challenges| challenges.iter())
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        self.challenges
            .iter_mut()
            .flat_map(|challenges| challenges.iter_mut())
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        self.advice_polys.iter()
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        self.advice_polys.iter_mut()
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        self.advice_commitments.iter()
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.advice_commitments.iter_mut()
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        self.advice_blinds.iter()
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        self.advice_blinds.iter_mut()
    }
}

/// Cache for storing the evaluated witness data during all phases of the advice generation.
struct WitnessCollection<'a, F: Field> {
    k: u32,
    current_phase: sealed::Phase,
    advice: &'a mut Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    challenges: &'a mut HashMap<usize, F>,
    instances: &'a Vec<Polynomial<F, LagrangeCoeff>>,
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

    fn query_instance(&mut self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        // TODO(@adr1anh): Compare with actual length of the instance column
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

        // TODO(@adr1anh): Compare with actual length of the advice column
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

        let params = poly::ipa::commitment::ParamsIPA::<pallas::Affine>::new(K);
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let pk = ProvingKey::new(&params, &circuit).unwrap();
        let data = create_advice_transcript(
            &params,
            &pk,
            &circuit,
            &Vec::new(),
            &mut rng,
            &mut transcript,
        );
        assert!(data.is_ok());
    }
}

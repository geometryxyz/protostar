use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    iter::zip,
    ops::RangeTo,
};

use ff::Field;
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    circuit::{layouter::SyncDeps, Value},
    plonk::{
        circuit::FloorPlanner,
        sealed::{self, SealedPhase},
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem, Error,
        FirstPhase, Fixed, Instance, Selector,
    },
    poly::{
        batch_invert_assigned,
        commitment::{Blind, Params},
        empty_lagrange, empty_lagrange_assigned, LagrangeCoeff, Polynomial,
    },
    protostar::{accumulator::committed::batch_commit, ProvingKey},
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::committed::Committed;

/// A gate transcript is the result of running the IOP for several rounds,
/// where in each round, the prover sends commitments to one or more advice columns
/// and the verifier responds with one or more challenges.
/// The order in which these are sent are defined by the ConstraintSystem.
/// Both parties agree on the instance, and then the prover sends the corresponding advice columns.
#[derive(PartialEq, Debug, Clone)]
pub struct Transcript<C: CurveAffine> {
    pub instance: Vec<Committed<C>>,
    pub advice: Vec<Committed<C>>,
    pub challenges: Vec<C::Scalar>,
}

impl<C: CurveAffine> Transcript<C> {
    pub fn new<
        'params,
        P: Params<'params, C>,
        ConcreteCircuit: Circuit<C::Scalar>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        params: &P,
        pk: &ProvingKey<C>,
        circuit: &ConcreteCircuit,
        instances: &[&[C::Scalar]],
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Self, Error> {
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

        let meta = &pk.cs;

        let instance: Vec<_> = {
            if instances.len() != meta.num_instance_columns {
                return Err(Error::InvalidInstances);
            }

            let instance_columns = instances
                .iter()
                .map(|values| {
                    // TODO(@adr1anh): Allocate only the required size for each column
                    let mut column = empty_lagrange(n);

                    if values.len() > (column.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    for (v, value) in zip(column.iter_mut(), values.iter()) {
                        *v = *value;
                    }
                    Ok(column)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // TODO(@adr1anh): Add support for query instance
            // if !P::QUERY_INSTANCE
            {
                // The instance is part of the transcript
                for &instance in instances {
                    for value in instance {
                        transcript.common_scalar(*value)?;
                    }
                }

                instance_columns
                    .into_iter()
                    .map(|column| Committed {
                        values: column,
                        commitment: C::identity(),
                        blind: Blind(C::Scalar::default()),
                    })
                    .collect()
            }
            // else {
            // // For large instances, we send a commitment to it and open it with PCS
            // batch_commit_transparent(params, instance_columns.into_iter(), transcript);
            // }
        };

        // Synthesize the circuit over multiple iterations
        let mut advice_assigned = vec![empty_lagrange_assigned(n); meta.num_advice_columns];
        let mut advice_committed = BTreeMap::<usize, Committed<C>>::new();
        let mut challenges = HashMap::<usize, C::Scalar>::with_capacity(meta.num_challenges);

        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);

        let instances = instance
            .iter()
            .map(|committed| &committed.values)
            .collect::<Vec<_>>();

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

            let committed = batch_commit(params, advice_values.into_iter(), &mut rng, transcript);

            for (column_index, committed) in column_indices.iter().zip(committed) {
                advice_committed.insert(*column_index, committed);
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

        Ok(Transcript {
            instance,
            advice: advice_committed.into_values().collect(),
            challenges,
        })
    }

    /// Computes the linear combination (1−α)⋅tx₀ + α⋅tx₁
    pub(super) fn merge(alpha: C::Scalar, transcript0: Self, transcript1: Self) -> Self {
        let advice = zip(
            transcript0.advice.into_iter(),
            transcript1.advice.into_iter(),
        )
        .map(|(committed0, committed1)| Committed::merge(alpha, committed0, committed1))
        .collect();
        let instance = zip(
            transcript0.instance.into_iter(),
            transcript1.instance.into_iter(),
        )
        .map(|(committed0, committed1)| Committed::merge(alpha, committed0, committed1))
        .collect();

        let challenges = zip(
            transcript0.challenges.into_iter(),
            transcript1.challenges.into_iter(),
        )
        .map(|(challenge0, challenge1)| challenge0 + (challenge1 - challenge0) * &alpha)
        .collect();
        Self {
            instance,
            advice,
            challenges,
        }
    }
}

/// Cache for storing the evaluated witness data during all phases of the advice generation.
struct WitnessCollection<'a, F: Field> {
    k: u32,
    current_phase: sealed::Phase,
    advice: &'a mut Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    challenges: &'a mut HashMap<usize, F>,
    instances: Vec<&'a Polynomial<F, LagrangeCoeff>>,
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

use std::iter::zip;

use group::Curve;
use halo2curves::{CurveAffine, CurveExt};

use crate::{
    plonk::{ConstraintSystem, Error},
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        empty_lagrange, LagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

#[derive(Debug, Clone, PartialEq)]
pub struct InstanceTranscript<C: CurveAffine> {
    pub instance_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub instance_commitments: Option<Vec<C>>,
    pub instance_blinds: Option<Vec<Blind<C::Scalar>>>,
    pub verifier_instance_commitments: Vec<C::Scalar>,
}

/// Generates `Polynomial`s from given `instance` values, and adds them to the transcript.
pub fn create_instance_transcript<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    cs: &ConstraintSystem<C::Scalar>,
    instances: &[&[C::Scalar]],
    transcript: &mut T,
) -> Result<InstanceTranscript<C>, Error> {
    // TODO(@adr1anh): Check that the lengths of each instance column is correct as well
    if instances.len() != cs.num_instance_columns {
        return Err(Error::InvalidInstances);
    }
    let n = params.n() as usize;

    // generate polys for instance columns
    // NOTE(@adr1anh): In the case where the verifier does not query the instance,
    // we do not need to create a Lagrange polynomial of size n.
    let mut verifier_instance_commitments: Vec<C::Scalar> = Vec::with_capacity(instances.len());

    let instance_polys = instances
        .iter()
        .map(|values| {
            // TODO(@adr1anh): Allocate only the required size for each column
            let mut poly = empty_lagrange(n);

            if values.len() > (poly.len() - (cs.blinding_factors() + 1)) {
                return Err(Error::InstanceTooLarge);
            }
            for (poly, value) in zip(poly.iter_mut(), values.iter()) {
                // The instance is part of the transcript
                // if !P::QUERY_INSTANCE {
                //     transcript.common_scalar(*value)?;
                // }
                transcript.common_scalar(*value)?;
                verifier_instance_commitments.push(*value);
                *poly = *value;
            }
            Ok(poly)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // TODO(@adr1anh): Split into two functions for handling both committed and raw instance columns
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
    Ok(InstanceTranscript {
        instance_polys,
        instance_commitments: None,
        instance_blinds: None,
        verifier_instance_commitments,
    })
}

impl<C: CurveAffine> InstanceTranscript<C> {
    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        std::iter::empty()
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        std::iter::empty()
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        self.instance_polys.iter()
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        self.instance_polys.iter_mut()
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        self.instance_commitments
            .iter()
            .flat_map(|commitments| commitments.iter())
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.instance_commitments
            .iter_mut()
            .flat_map(|commitments| commitments.iter_mut())
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        self.instance_blinds.iter().flat_map(|blinds| blinds.iter())
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        self.instance_blinds
            .iter_mut()
            .flat_map(|blinds| blinds.iter_mut())
    }
}

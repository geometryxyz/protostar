use std::iter::zip;

use group::Curve;
use halo2curves::CurveAffine;

use crate::{
    plonk::{ConstraintSystem, Error},
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        empty_lagrange, LagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::committed::{batch_commit_transparent, Committed};

#[derive(PartialEq, Debug, Clone)]
pub struct Transcript<C: CurveAffine> {
    pub committed: Vec<Committed<C>>,
}

impl<C: CurveAffine> Transcript<C> {
    pub fn new<'params, P: Params<'params, C>, E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        params: &P,
        cs: &ConstraintSystem<C::Scalar>,
        instances: &[&[C::Scalar]],
        transcript: &mut T,
    ) -> Result<Self, Error> {
        if instances.len() != cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
        let n = params.n() as usize;

        let instance_columns = instances
            .iter()
            .map(|values| {
                // TODO(@adr1anh): Allocate only the required size for each column
                let mut column = empty_lagrange(n);

                if values.len() > (column.len() - (cs.blinding_factors() + 1)) {
                    return Err(Error::InstanceTooLarge);
                }
                for (v, value) in zip(column.iter_mut(), values.iter()) {
                    *v = *value;
                }
                Ok(column)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let committed = {
            // if !P::QUERY_INSTANCE {
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
            // } else {
            // // For large instances, we send a commitment to it and open it with PCS
            // batch_commit_transparent(params, instance_columns.into_iter(), transcript);
            // }
        };

        Ok(Transcript { committed })
    }

    pub(super) fn merge(alpha: C::Scalar, transcript0: Self, transcript1: Self) -> Self {
        let committed = zip(
            transcript0.committed.into_iter(),
            transcript1.committed.into_iter(),
        )
        .map(|(committed0, committed1)| Committed::fold(alpha, committed0, committed1))
        .collect();

        Self { committed }
    }

    pub fn columns_ref(&self) -> Vec<&[C::Scalar]> {
        self.committed.iter().map(|c| c.values.as_ref()).collect()
    }
}

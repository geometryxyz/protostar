use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    poly::{Basis, LagrangeCoeff, Polynomial},
    protostar::{
        accumulator::{committed::Committed, Accumulator},
        ProvingKey,
    },
};

use super::{expression::QueryType, Data, LookupData};

/// Defines a QueriedExpression where leaves are references values from two accumulators.
pub struct CommittedRef<'a, C> {
    _marker: std::marker::PhantomData<&'a C>,
}

impl<'a, C: CurveAffine> QueryType for CommittedRef<'a, C> {
    type F = C::Scalar;
    type Challenge = &'a C::Scalar;
    type Fixed = &'a Committed<C>;
    type Witness = &'a Committed<C>;
}

impl<'a, C: CurveAffine> Data<CommittedRef<'a, C>> {
    pub fn new(pk: &'a ProvingKey<C>, acc: &'a Accumulator<C>) -> Self {
        let selectors: Vec<_> = pk.selectors.iter().collect();
        let fixed: Vec<_> = pk.fixed.iter().collect();

        let instance: Vec<_> = acc.gate.instance.iter().collect();

        let advice: Vec<_> = acc.gate.advice.iter().collect();

        let challenges: Vec<_> = acc.gate.challenges.iter().collect();

        let beta = &acc.beta.beta;

        let lookups: Vec<_> = acc
            .lookups
            .iter()
            .map(|lookup| {
                let m = &lookup.m;
                let g = &lookup.g;
                let h = &lookup.h;
                let thetas: Vec<_> = lookup.thetas.iter().collect();
                let r = &lookup.r;
                LookupData { m, g, h, thetas, r }
            })
            .collect();

        let ys: Vec<_> = acc.ys.iter().collect();

        Self {
            fixed,
            selectors,
            instance,
            advice,
            challenges,
            beta,
            lookups,
            ys,
        }
    }
}

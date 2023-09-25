use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    poly::{Basis, LagrangeCoeff, Polynomial},
    protostar::{accumulator::Accumulator, ProvingKey},
};

use super::{expression::QueryType, Data, LookupData};

/// Defines a QueriedExpression where leaves are references values from two accumulators.
pub struct PolynomialRef<'a, F, B> {
    _marker: std::marker::PhantomData<&'a (F, B)>,
}

impl<'a, F: Field, B: Basis> QueryType for PolynomialRef<'a, F, B> {
    type F = F;
    type Challenge = &'a F;
    type Fixed = &'a Polynomial<F, B>;
    type Witness = &'a Polynomial<F, B>;
}

impl<'a, F: Field> Data<PolynomialRef<'a, F, LagrangeCoeff>> {
    pub fn new<C>(pk: &'a ProvingKey<C>, acc: &'a Accumulator<C>) -> Self
    where
        C: CurveAffine<ScalarExt = F>,
    {
        let selectors: Vec<_> = pk.selectors_polys.iter().collect();
        let fixed: Vec<_> = pk.fixed_polys.iter().collect();

        let instance: Vec<_> = acc.instance.committed.iter().map(|i| &i.values).collect();

        let advice: Vec<_> = acc.advice.committed.iter().map(|a| &a.values).collect();

        let challenges: Vec<_> = acc.advice.challenges.iter().collect();

        let beta = &acc.beta.beta.values;

        let lookups: Vec<_> = acc
            .lookups
            .iter()
            .map(|lookup| {
                let m = &lookup.m.values;
                let g = &lookup.g.values;
                let h = &lookup.h.values;
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
            num_rows: pk.num_rows,
        }
    }
}



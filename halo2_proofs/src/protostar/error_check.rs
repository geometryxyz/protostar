use std::collections::BTreeMap;

use crate::poly::{LagrangeCoeff, Polynomial};
use halo2curves::CurveAffine;

// enum CircuitInstance<C:CurveAffine> {
//     Raw(Vec<Poly<C::Scalar, LagrangeCoeff>>),
//     Committed(Vec<C>),
// }

struct AccumulatorInstance<C: CurveAffine> {
    // challenges[i][p] = challenge[i][0]^{p-1}
    // p is the power of the challenge and >1.
    challenges: Vec<Vec<C::Scalar>>,
    instance_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    advice_commitments: Vec<C>,

    // powers of the constraint separation challenge
    y_challenges: Vec<C::Scalar>,
    // thetas: Vec<C::Scalar>,
    // aux_lookup_commitments: Vec<C>,
    // aux_table_online_commitments: Vec<C>,
    // aux_table_fixed_commitments: Vec<C>,
}

struct AccumulatorWitness<C: CurveAffine> {
    advice_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    // aux_lookup_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    // aux_table_polys: Vec<BTreeMap<usize, C::Scalar>>,
    // aux_cached_polys: Vec<BTreeMap<usize, C::Scalar>>,
}

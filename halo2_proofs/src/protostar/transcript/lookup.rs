use core::num;
use ff::{BatchInvert, Field, PrimeField};
use group::Curve;
use halo2curves::CurveAffine;
use rand_core::RngCore;
use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Mul, MulAssign},
};

use crate::{
    arithmetic::{parallelize, powers},
    plonk::{evaluation::evaluate, lookup::Argument, Expression},
    poly::{
        commitment::{Blind, Params},
        empty_lagrange, lagrange_from_vec, LagrangeCoeff, Polynomial,
    },
    protostar::keygen::ProvingKey,
    transcript::{EncodedChallenge, TranscriptWrite},
};

#[derive(Debug)]
pub struct LookupTranscipt<C: CurveAffine> {
    pub challanges_theta: Vec<C::Scalar>,

    pub m_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub m_commitments: Vec<C>,
    pub m_blinds: Vec<Blind<C::Scalar>>,

    pub challenge_r: C::Scalar,

    pub g_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub g_commitments: Vec<C>,
    pub g_blinds: Vec<Blind<C::Scalar>>,

    pub h_polys: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    pub h_commitments: Vec<C>,
    pub h_blinds: Vec<Blind<C::Scalar>>,
}

pub(crate) fn create_lookup_transcript<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    pk: &ProvingKey<C>,
    advice_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
    instance_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
    challenges: &[C::Scalar],
    mut rng: R,
    transcript: &mut T,
) -> Option<LookupTranscipt<C>> {
    let lookups = pk.cs().lookups();
    let num_lookups = lookups.len();
    if num_lookups == 0 {
        return None;
    }

    // TODO(@adr1anh): Fix soundness bug:
    // The challenge theta must be sampled after having committed to the m's
    // It should be sampled at the same time as r.
    // This is very annoying since we use theta to compress the table columns and efficiently lookup indices.
    // Proposed solution:
    // During preprocessing, we create a HashMap<Vec<C::Scalar>, usize> for each lookup argument,
    // and use it to lookup the indices.

    let theta = *transcript.squeeze_challenge_scalar::<C::Scalar>();

    let num_thetas = lookups
        .iter()
        .map(|lookup| {
            std::cmp::max(
                lookup.input_expressions().len(),
                lookup.table_expressions().len(),
            )
        })
        .max()
        .unwrap();
    let thetas: Vec<_> = powers(theta).skip(1).take(num_thetas).collect();

    // Closure to get values of expressions and compress them
    let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
        let compressed_expression = expressions
            .iter()
            .map(|expression| {
                lagrange_from_vec(evaluate(
                    expression,
                    params.n() as usize,
                    1,
                    &pk.fixed,
                    advice_values,
                    instance_values,
                    &challenges,
                ))
            })
            .enumerate()
            .fold(
                empty_lagrange(params.n() as usize),
                |acc, (i, expression)| acc * thetas[i] + &expression,
            );
        compressed_expression
    };

    let mut compressed_inputs = Vec::<_>::with_capacity(num_lookups);
    let mut compressed_tables = Vec::<_>::with_capacity(num_lookups);

    let mut m_polys = Vec::<_>::with_capacity(num_lookups);
    let mut m_blinds = Vec::<_>::with_capacity(num_lookups);
    let mut m_commitments_projective = Vec::<_>::with_capacity(num_lookups);

    for lookup in lookups {
        // Get values of input expressions involved in the lookup and compress them
        let compressed_input_expression = compress_expressions(lookup.input_expressions());

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(lookup.table_expressions());

        let blinding_factors = pk.cs().blinding_factors();

        // compute m(X)
        let table_index_value_mapping: BTreeMap<C::Scalar, usize> = compressed_table_expression
            .iter()
            .take(params.n() as usize - blinding_factors - 1)
            .enumerate()
            .map(|(i, &x)| (x, i))
            .collect();

        let mut m_poly = empty_lagrange(params.n() as usize);

        compressed_input_expression
            .iter()
            .take(params.n() as usize - blinding_factors - 1)
            .for_each(|fi| {
                let index = table_index_value_mapping.get(fi).unwrap_or_else(|| {
                    panic!("in lookup: {}, value: {:?} not in table", lookup.name(), fi)
                });
                m_poly[*index] += C::Scalar::ONE;
            });

        compressed_inputs.push(compressed_input_expression);
        compressed_tables.push(compressed_table_expression);

        // commit to m(X)
        let m_blind = Blind::new(&mut rng);
        let m_commitment_projective = params.commit_lagrange(&m_poly, m_blind);

        m_polys.push(m_poly);
        m_commitments_projective.push(m_commitment_projective);
        m_blinds.push(m_blind);
    }

    let mut m_commitments = vec![C::identity(); num_lookups];
    C::CurveExt::batch_normalize(&m_commitments_projective, &mut m_commitments);

    // write commitment of m(X) to transcript
    for m_commitment in m_commitments.iter() {
        let _ = transcript.write_point(*m_commitment);
    }

    let r = *transcript.squeeze_challenge_scalar::<C::Scalar>();

    let mut g_polys = Vec::<_>::with_capacity(num_lookups);
    let mut g_blinds = Vec::<_>::with_capacity(num_lookups);
    let mut g_commitments_projective = Vec::<_>::with_capacity(num_lookups);

    let mut h_polys = Vec::<_>::with_capacity(num_lookups);
    let mut h_blinds = Vec::<_>::with_capacity(num_lookups);
    let mut h_commitments_projective = Vec::<_>::with_capacity(num_lookups);

    for i in 0..num_lookups {
        let mut g_poly = empty_lagrange(params.n() as usize);

        parallelize(&mut g_poly, |g_poly, start| {
            for (g_i, fi) in g_poly.iter_mut().zip(compressed_inputs[i][start..].iter()) {
                *g_i = r + fi;
            }
        });
        g_poly.iter_mut().batch_invert();

        let mut h_poly = empty_lagrange(params.n() as usize);
        parallelize(&mut h_poly, |h_poly, start| {
            for (h_i, ti) in h_poly.iter_mut().zip(compressed_tables[i][start..].iter()) {
                *h_i = r + ti;
            }
        });
        h_poly.iter_mut().batch_invert();

        // commit to g(X)
        let g_blind = Blind::new(&mut rng);
        let g_commitment_projective = params.commit_lagrange(&g_poly, g_blind);

        g_polys.push(g_poly);
        g_commitments_projective.push(g_commitment_projective);
        g_blinds.push(g_blind);
        // commit to h(X)
        let h_blind = Blind::new(&mut rng);
        let h_commitment_projective = params.commit_lagrange(&h_poly, h_blind);

        h_polys.push(h_poly);
        h_commitments_projective.push(h_commitment_projective);
        h_blinds.push(h_blind);
    }

    let mut g_commitments = vec![C::identity(); num_lookups];
    C::CurveExt::batch_normalize(&g_commitments_projective, &mut g_commitments);

    // write commitment of g(X) to transcript
    for g_commitment in g_commitments.iter() {
        let _ = transcript.write_point(*g_commitment);
    }

    let mut h_commitments = vec![C::identity(); num_lookups];
    C::CurveExt::batch_normalize(&h_commitments_projective, &mut h_commitments);

    // write commitment of h(X) to transcript
    for h_commitment in h_commitments.iter() {
        let _ = transcript.write_point(*h_commitment);
    }

    Some(LookupTranscipt {
        challanges_theta: thetas,
        m_polys,
        m_commitments,
        m_blinds,
        challenge_r: r,
        g_polys,
        g_commitments,
        g_blinds,
        h_polys,
        h_commitments,
        h_blinds,
    })
}
impl<C: CurveAffine> LookupTranscipt<C> {
    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        self.challanges_theta
            .iter()
            .chain(std::iter::once(&self.challenge_r))
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        self.challanges_theta
            .iter_mut()
            .chain(std::iter::once(&mut self.challenge_r))
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        self.m_polys
            .iter()
            .chain(self.g_polys.iter())
            .chain(self.h_polys.iter())
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        self.m_polys
            .iter_mut()
            .chain(self.g_polys.iter_mut())
            .chain(self.h_polys.iter_mut())
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        self.m_commitments
            .iter()
            .chain(self.g_commitments.iter())
            .chain(self.h_commitments.iter())
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.m_commitments
            .iter_mut()
            .chain(self.g_commitments.iter_mut())
            .chain(self.h_commitments.iter_mut())
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        self.m_blinds
            .iter()
            .chain(self.g_blinds.iter())
            .chain(self.h_blinds.iter())
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        self.m_blinds
            .iter_mut()
            .chain(self.g_blinds.iter_mut())
            .chain(self.h_blinds.iter_mut())
    }
}

use core::num;
use ff::{BatchInvert, Field, FromUniformBytes, PrimeField};
use group::Curve;
use halo2curves::CurveAffine;
use rand_core::RngCore;
use std::{
    collections::{BTreeMap, HashMap},
    iter::zip,
    ops::{Mul, MulAssign},
};

use crate::{
    arithmetic::{parallelize, powers},
    plonk::{
        evaluation::evaluate,
        lookup::{self, Argument},
        Error, Expression,
    },
    poly::{
        commitment::{Blind, Params},
        empty_lagrange, lagrange_from_vec, LagrangeCoeff, Polynomial,
    },
    protostar::{error_check, keygen::ProvingKey},
    transcript::{EncodedChallenge, TranscriptWrite},
};

#[derive(Debug, Clone, PartialEq)]
pub struct LookupTranscipt<C: CurveAffine> {
    pub thetas: Option<Vec<C::Scalar>>,
    pub r: Option<C::Scalar>,
    pub singles_transcript: Vec<LookupTranscriptSingle<C>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LookupTranscriptSingle<C: CurveAffine> {
    pub theta: C::Scalar,

    pub m_poly: Polynomial<C::Scalar, LagrangeCoeff>,
    pub m_commitment: C,
    pub m_blind: Blind<C::Scalar>,

    pub g_poly: Polynomial<C::Scalar, LagrangeCoeff>,
    pub g_commitment: C,
    pub g_blind: Blind<C::Scalar>,

    pub h_poly: Polynomial<C::Scalar, LagrangeCoeff>,
    pub h_commitment: C,
    pub h_blind: Blind<C::Scalar>,
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
    challenges: &[Vec<C::Scalar>],
    advice: &[Polynomial<C::Scalar, LagrangeCoeff>],
    instance: &[Polynomial<C::Scalar, LagrangeCoeff>],
    mut rng: R,
    transcript: &mut T,
) -> LookupTranscipt<C> {
    let lookups = &pk.cs.lookups();
    let num_lookups = lookups.len();
    if num_lookups == 0 {
        return LookupTranscipt {
            thetas: None,
            r: None,
            singles_transcript: vec![],
        };
    }
    let num_rows = params.n() as usize - pk.cs.blinding_factors() - 1 as usize;

    let table_values_map: Vec<_> = lookups
        .iter()
        .map(|lookup| {
            build_lookup_index_table(
                lookup.table_expressions(),
                num_rows,
                challenges,
                &pk.selectors,
                &pk.fixed,
                instance,
                advice,
            )
        })
        .collect();

    let m_polys: Vec<_> = lookups
        .iter()
        .zip(table_values_map.iter())
        .map(|(lookup, table_index_map)| {
            build_m_poly(
                lookup.input_expressions(),
                table_index_map,
                num_rows,
                challenges,
                &pk.selectors,
                &pk.fixed,
                instance,
                advice,
            )
            .unwrap()
        })
        .collect();

    let (m_commitments_projective, m_blinds): (Vec<_>, Vec<_>) = m_polys
        .iter()
        .map(|m_poly| {
            let m_blind = Blind::new(&mut rng);
            (params.commit_lagrange(&m_poly, m_blind), m_blind)
        })
        .unzip();

    let mut m_commitments = vec![C::identity(); num_lookups];
    C::CurveExt::batch_normalize(&m_commitments_projective, &mut m_commitments);

    // write commitment of m(X) to transcript
    for m_commitment in m_commitments.iter() {
        let _ = transcript.write_point(*m_commitment);
    }

    let [theta, r] = [(); 2].map(|_| *transcript.squeeze_challenge_scalar::<C::Scalar>());

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

    let g_polys: Vec<_> = lookups
        .iter()
        .map(|lookup| {
            build_g_poly(
                lookup.input_expressions(),
                num_rows,
                challenges,
                &pk.selectors,
                &pk.fixed,
                instance,
                advice,
                &thetas,
                r,
            )
            .unwrap()
        })
        .collect();

    let (g_commitments_projective, g_blinds): (Vec<_>, Vec<_>) = g_polys
        .iter()
        .map(|g_poly| {
            let g_blind = Blind::default();
            (params.commit_lagrange(g_poly, g_blind), g_blind)
        })
        .unzip();

    let mut g_commitments = vec![C::identity(); num_lookups];
    C::CurveExt::batch_normalize(&g_commitments_projective, &mut g_commitments);
    // write commitment of g(X) to transcript
    for g_commitment in g_commitments.iter() {
        let _ = transcript.write_point(*g_commitment);
    }

    let h_polys: Vec<_> = lookups
        .iter()
        .zip(m_polys.iter())
        .map(|(lookup, m_poly)| {
            build_h_poly(
                lookup.table_expressions(),
                num_rows,
                challenges,
                &pk.selectors,
                &pk.fixed,
                instance,
                advice,
                m_poly,
                &thetas,
                r,
            )
            .unwrap()
        })
        .collect();

    let (h_commitments_projective, h_blinds): (Vec<_>, Vec<_>) = h_polys
        .iter()
        .map(|h_poly| {
            let h_blind = Blind::default();
            (params.commit_lagrange(&h_poly, h_blind), h_blind)
        })
        .unzip();

    let mut h_commitments = vec![C::identity(); num_lookups];
    C::CurveExt::batch_normalize(&h_commitments_projective, &mut h_commitments);
    // write commitment of g(X) to transcript
    for h_commitment in h_commitments.iter() {
        let _ = transcript.write_point(*h_commitment);
    }

    let mut singles_transcript = vec![LookupTranscriptSingle::default(); num_lookups];

    for (i, m_poly) in m_polys.into_iter().enumerate() {
        singles_transcript[i].m_poly = m_poly;
    }
    for (i, m_commitment) in m_commitments.into_iter().enumerate() {
        singles_transcript[i].m_commitment = m_commitment;
    }
    for (i, m_blind) in m_blinds.into_iter().enumerate() {
        singles_transcript[i].m_blind = m_blind;
    }
    for (i, h_poly) in h_polys.into_iter().enumerate() {
        singles_transcript[i].h_poly = h_poly;
    }
    for (i, h_commitment) in h_commitments.into_iter().enumerate() {
        singles_transcript[i].h_commitment = h_commitment;
    }
    for (i, h_blind) in h_blinds.into_iter().enumerate() {
        singles_transcript[i].h_blind = h_blind;
    }
    for (i, g_poly) in g_polys.into_iter().enumerate() {
        singles_transcript[i].g_poly = g_poly;
    }
    for (i, g_commitment) in g_commitments.into_iter().enumerate() {
        singles_transcript[i].g_commitment = g_commitment;
    }
    for (i, g_blind) in g_blinds.into_iter().enumerate() {
        singles_transcript[i].g_blind = g_blind;
    }

    LookupTranscipt {
        thetas: Some(thetas),
        r: Some(r),
        singles_transcript,
    }
}
impl<C: CurveAffine> LookupTranscipt<C> {
    pub fn challenges_iter(&self) -> impl Iterator<Item = &C::Scalar> {
        self.thetas
            .iter()
            .flat_map(|c| c.iter())
            .chain(self.r.iter().flat_map(std::iter::once))
    }

    pub fn challenges_iter_mut(&mut self) -> impl Iterator<Item = &mut C::Scalar> {
        self.thetas
            .iter_mut()
            .flat_map(|c| c.iter_mut())
            .chain(self.r.iter_mut().flat_map(std::iter::once))
    }

    pub fn polynomials_iter(&self) -> impl Iterator<Item = &Polynomial<C::Scalar, LagrangeCoeff>> {
        self.singles_transcript.iter().flat_map(|transcript| {
            std::iter::once(&transcript.m_poly)
                .chain(std::iter::once(&transcript.g_poly))
                .chain(std::iter::once(&transcript.h_poly))
        })
    }

    pub fn polynomials_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Polynomial<C::Scalar, LagrangeCoeff>> {
        self.singles_transcript.iter_mut().flat_map(|transcript| {
            std::iter::once(&mut transcript.m_poly)
                .chain(std::iter::once(&mut transcript.g_poly))
                .chain(std::iter::once(&mut transcript.h_poly))
        })
    }

    pub fn commitments_iter(&self) -> impl Iterator<Item = &C> {
        self.singles_transcript.iter().flat_map(|transcript| {
            std::iter::once(&transcript.m_commitment)
                .chain(std::iter::once(&transcript.g_commitment))
                .chain(std::iter::once(&transcript.h_commitment))
        })
    }

    pub fn commitments_iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.singles_transcript.iter_mut().flat_map(|transcript| {
            std::iter::once(&mut transcript.m_commitment)
                .chain(std::iter::once(&mut transcript.g_commitment))
                .chain(std::iter::once(&mut transcript.h_commitment))
        })
    }

    pub fn blinds_iter(&self) -> impl Iterator<Item = &Blind<C::Scalar>> {
        self.singles_transcript.iter().flat_map(|transcript| {
            std::iter::once(&transcript.m_blind)
                .chain(std::iter::once(&transcript.g_blind))
                .chain(std::iter::once(&transcript.h_blind))
        })
    }

    pub fn blinds_iter_mut(&mut self) -> impl Iterator<Item = &mut Blind<C::Scalar>> {
        self.singles_transcript.iter_mut().flat_map(|transcript| {
            std::iter::once(&mut transcript.m_blind)
                .chain(std::iter::once(&mut transcript.g_blind))
                .chain(std::iter::once(&mut transcript.h_blind))
        })
    }
}

fn build_lookup_index_table<F: PrimeField>(
    table_expressions: &[Expression<F>],
    num_rows: usize,
    challenges: &[Vec<F>],
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    instance: &[Polynomial<F, LagrangeCoeff>],
    advice: &[Polynomial<F, LagrangeCoeff>],
) -> HashMap<Vec<u8>, usize> {
    let mut table_evaluator =
        error_check::gate::GateEvaluator::new_single(table_expressions, challenges);

    let mut map: HashMap<Vec<u8>, usize> = HashMap::new();

    let mut row_evals = vec![F::ZERO; table_expressions.len()];
    let mut row_evals_repr: Vec<u8> = Vec::new();

    for row_idx in 0..num_rows {
        table_evaluator.evaluate_single(
            &mut row_evals,
            row_idx,
            selectors,
            fixed,
            instance,
            advice,
        );
        row_evals_repr.clear();
        for e in row_evals.iter() {
            row_evals_repr.extend(e.to_repr().as_ref().iter());
        }
        map.insert(row_evals_repr.clone(), row_idx);
    }
    map
}

fn build_m_poly<F: PrimeField>(
    lookup_expressions: &[Expression<F>],
    table_index_map: &HashMap<Vec<u8>, usize>,
    num_rows: usize,
    challenges: &[Vec<F>],
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    instance: &[Polynomial<F, LagrangeCoeff>],
    advice: &[Polynomial<F, LagrangeCoeff>],
) -> Result<Polynomial<F, LagrangeCoeff>, Error> {
    let mut lookup_evaluator =
        error_check::gate::GateEvaluator::new_single(lookup_expressions, challenges);

    let mut row_evals = vec![F::ZERO; lookup_expressions.len()];
    let mut row_evals_repr: Vec<u8> = Vec::new();

    let mut m_poly = empty_lagrange(num_rows);

    for row_idx in 0..num_rows {
        lookup_evaluator.evaluate_single(
            &mut row_evals,
            row_idx,
            selectors,
            fixed,
            instance,
            advice,
        );
        row_evals_repr.clear();
        for e in row_evals.iter() {
            row_evals_repr.extend(e.to_repr().as_ref().iter());
        }
        if let Some(index) = table_index_map.get(&row_evals_repr) {
            m_poly[*index] += F::ONE;
        } else {
            return Err(Error::BoundsFailure);
        }
    }
    Ok(m_poly)
}

fn build_g_poly<F: PrimeField>(
    lookup_expressions: &[Expression<F>],
    num_rows: usize,
    challenges: &[Vec<F>],
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    instance: &[Polynomial<F, LagrangeCoeff>],
    advice: &[Polynomial<F, LagrangeCoeff>],
    thetas: &[F],
    r: F,
) -> Result<Polynomial<F, LagrangeCoeff>, Error> {
    let mut lookup_evaluator =
        error_check::gate::GateEvaluator::new_single(lookup_expressions, challenges);

    let mut row_evals = vec![F::ZERO; lookup_expressions.len()];

    let mut g_poly = empty_lagrange(num_rows);

    for row_idx in 0..num_rows {
        lookup_evaluator.evaluate_single(
            &mut row_evals,
            row_idx,
            selectors,
            fixed,
            instance,
            advice,
        );
        g_poly[row_idx] =
            zip(row_evals.iter(), thetas.iter()).fold(r, |acc, (eval, theta)| acc + *eval * theta);
    }
    g_poly.iter_mut().batch_invert();
    Ok(g_poly)
}

fn build_h_poly<F: PrimeField>(
    table_expressions: &[Expression<F>],
    num_rows: usize,
    challenges: &[Vec<F>],
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    instance: &[Polynomial<F, LagrangeCoeff>],
    advice: &[Polynomial<F, LagrangeCoeff>],
    m_poly: &Polynomial<F, LagrangeCoeff>,
    thetas: &[F],
    r: F,
) -> Result<Polynomial<F, LagrangeCoeff>, Error> {
    let mut table_evaluator =
        error_check::gate::GateEvaluator::new_single(table_expressions, challenges);

    let mut row_evals = vec![F::ZERO; table_expressions.len()];

    let mut h_poly = empty_lagrange(num_rows);

    for row_idx in 0..num_rows {
        table_evaluator.evaluate_single(
            &mut row_evals,
            row_idx,
            selectors,
            fixed,
            instance,
            advice,
        );
        h_poly[row_idx] =
            zip(row_evals.iter(), thetas.iter()).fold(r, |acc, (eval, theta)| acc + *eval * theta);
    }
    h_poly.iter_mut().batch_invert();
    for row_idx in 0..num_rows {
        h_poly[row_idx] *= m_poly[row_idx];
    }
    Ok(h_poly)
}

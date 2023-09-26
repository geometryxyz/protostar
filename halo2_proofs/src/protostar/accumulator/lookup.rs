use core::num;
use std::{collections::HashMap, iter::zip, ops::Range};

use crate::{
    arithmetic::powers,
    plonk::Expression,
    poly::{commitment::Params, Rotation},
    protostar::ProvingKey,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::PrimeField;
use ff::{BatchInvert, Field};
use halo2curves::CurveAffine;
use rand_core::RngCore;

use super::committed::{batch_commit, commit, Committed};

#[derive(PartialEq, Debug, Clone)]
pub struct Transcript<C: CurveAffine> {
    pub m: Committed<C>,
    pub r: C::Scalar,
    pub thetas: Vec<C::Scalar>,
    pub g: Committed<C>,
    pub h: Committed<C>,
}

impl<C: CurveAffine> Transcript<C> {
    pub(super) fn merge(alpha: C::Scalar, transcript0: Self, transcript1: Self) -> Self {
        let m = Committed::merge(alpha, transcript0.m, transcript1.m);
        let g = Committed::merge(alpha, transcript0.g, transcript1.g);
        let h = Committed::merge(alpha, transcript0.h, transcript1.h);

        let r = (transcript1.r - transcript0.r) * alpha + transcript0.r;
        let thetas = zip(
            transcript0.thetas.into_iter(),
            transcript1.thetas.into_iter(),
        )
        .map(|(challenge0, challenge1)| (challenge1 - challenge0) * alpha + challenge0)
        .collect();
        Self { m, r, thetas, g, h }
    }
}

pub fn new<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    params: &P,
    pk: &ProvingKey<C>,
    gate_tx: &super::gate::Transcript<C>,
    mut rng: R,
    transcript: &mut T,
) -> Vec<Transcript<C>> {
    let selectors = pk
        .selectors
        .iter()
        .map(|c| c.values.as_ref())
        .collect::<Vec<_>>();
    let fixed = pk
        .fixed
        .iter()
        .map(|c| c.values.as_ref())
        .collect::<Vec<_>>();
    let instance = gate_tx
        .instance
        .iter()
        .map(|c| c.values.as_ref())
        .collect::<Vec<_>>();
    let challenges = &gate_tx.challenges;
    let advice = gate_tx
        .advice
        .iter()
        .map(|c| c.values.as_ref())
        .collect::<Vec<_>>();

    let m_columns = pk
        .cs
        .lookups
        .iter()
        .map(|lookup| {
            pk.domain.lagrange_from_vec(build_m(
                lookup,
                &pk.usable_rows,
                pk.num_rows,
                &selectors,
                &fixed,
                &instance,
                &advice,
                challenges,
            ))
        })
        .collect::<Vec<_>>();

    let m_committed = batch_commit(params, m_columns.into_iter(), &mut rng, transcript);

    let [r, theta] = [(); 2].map(|_| *transcript.squeeze_challenge_scalar::<C::Scalar>());

    pk.cs
        .lookups
        .iter()
        .zip(m_committed.into_iter())
        .map(|(lookup, m)| {
            let num_inputs = lookup.input_expressions.len();
            let thetas = powers(theta).skip(1).take(num_inputs).collect::<Vec<_>>();

            let g_column = pk.domain.lagrange_from_vec(build_g(
                lookup,
                &pk.usable_rows,
                pk.num_rows,
                &selectors,
                &fixed,
                &instance,
                &advice,
                challenges,
                &m.values,
                &thetas,
                r,
            ));

            let h_column = pk.domain.lagrange_from_vec(build_h(
                lookup,
                &pk.usable_rows,
                pk.num_rows,
                &selectors,
                &fixed,
                &instance,
                &advice,
                challenges,
                &thetas,
                r,
            ));

            let g = commit(params, g_column, &mut rng, transcript);
            let h = commit(params, h_column, &mut rng, transcript);

            Transcript { m, r, thetas, g, h }
        })
        .collect()
}

fn build_m<F: PrimeField>(
    lookup: &crate::plonk::lookup::Argument<F>,
    usable_rows: &Range<usize>,
    num_rows: usize,
    selectors: &[&[F]],
    fixed: &[&[F]],
    instance: &[&[F]],
    advice: &[&[F]],
    challenges: &[F],
) -> Vec<F> {
    let mut row_evals_repr: Vec<u8> = Vec::new();

    let mut map = HashMap::<Vec<u8>, usize>::new();

    for row_idx in usable_rows.clone() {
        row_evals_repr.clear();

        for expr in &lookup.table_expressions {
            let eval = evaluate(
                row_idx, num_rows, expr, selectors, fixed, instance, advice, challenges,
            );
            row_evals_repr.extend(eval.to_repr().as_ref().iter());
        }

        map.insert(row_evals_repr.clone(), row_idx);
    }

    let mut m = vec![F::ZERO; num_rows];

    for row_idx in usable_rows.clone() {
        row_evals_repr.clear();

        for expr in &lookup.input_expressions {
            let eval = evaluate(
                row_idx, num_rows, expr, selectors, fixed, instance, advice, challenges,
            );
            row_evals_repr.extend(eval.to_repr().as_ref().iter());
        }

        if let Some(index) = map.get(&row_evals_repr) {
            m[*index] += F::ONE;
        }
    }

    m
}

fn build_g<F: PrimeField>(
    lookup: &crate::plonk::lookup::Argument<F>,
    usable_rows: &Range<usize>,
    num_rows: usize,
    selectors: &[&[F]],
    fixed: &[&[F]],
    instance: &[&[F]],
    advice: &[&[F]],
    challenges: &[F],
    m: &[F],
    thetas: &[F],
    r: F,
) -> Vec<F> {
    let mut g = vec![F::ZERO; num_rows];

    for row_idx in usable_rows.clone() {
        g[row_idx] = evaluate_linear_combination(
            row_idx,
            num_rows,
            &lookup.table_expressions,
            thetas,
            selectors,
            fixed,
            instance,
            advice,
            challenges,
        ) + r;
    }

    g.iter_mut().batch_invert();

    for row_idx in usable_rows.clone() {
        g[row_idx] *= m[row_idx];
    }
    g
}

fn build_h<F: PrimeField>(
    lookup: &crate::plonk::lookup::Argument<F>,
    usable_rows: &Range<usize>,
    num_rows: usize,
    selectors: &[&[F]],
    fixed: &[&[F]],
    instance: &[&[F]],
    advice: &[&[F]],
    challenges: &[F],
    thetas: &[F],
    r: F,
) -> Vec<F> {
    let mut h = vec![F::ZERO; num_rows];

    for row_idx in usable_rows.clone() {
        h[row_idx] = evaluate_linear_combination(
            row_idx,
            num_rows,
            &lookup.input_expressions,
            thetas,
            selectors,
            fixed,
            instance,
            advice,
            challenges,
        ) + r;
    }

    h.iter_mut().batch_invert();
    h
}

fn evaluate_linear_combination<F: Field>(
    row_idx: usize,
    num_rows: usize,
    exprs: &[Expression<F>],
    coeffs: &[F],
    selectors: &[&[F]],
    fixed: &[&[F]],
    instance: &[&[F]],
    advice: &[&[F]],
    challenges: &[F],
) -> F {
    zip(exprs.iter(), coeffs.iter()).fold(F::ZERO, |acc, (expr, coeff)| {
        acc + evaluate(
            row_idx, num_rows, expr, selectors, fixed, instance, advice, challenges,
        ) * coeff
    })
}

fn evaluate<F: Field>(
    row_idx: usize,
    num_rows: usize,
    expr: &Expression<F>,
    selectors: &[&[F]],
    fixed: &[&[F]],
    instance: &[&[F]],
    advice: &[&[F]],
    challenges: &[F],
) -> F {
    let num_rows_i = num_rows as i32;
    expr.evaluate(
        &|constant| constant,
        &|selector_column| selectors[selector_column.index()][row_idx],
        &|query| fixed[query.column_index][get_rotation_idx(row_idx, query.rotation, num_rows_i)],
        &|query| advice[query.column_index][get_rotation_idx(row_idx, query.rotation, num_rows_i)],
        &|query| {
            instance[query.column_index][get_rotation_idx(row_idx, query.rotation, num_rows_i)]
        },
        &|challenge| challenges[challenge.index()],
        &|v| -v,
        &|v1, v2| v1 + v2,
        &|v1, v2| v1 * v2,
        &|v, c| v * c,
    )
}

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: Rotation, isize: i32) -> usize {
    (((idx as i32) + rot.0).rem_euclid(isize)) as usize
}

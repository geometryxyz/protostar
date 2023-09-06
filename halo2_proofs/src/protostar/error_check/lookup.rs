use std::{collections::BTreeSet, iter::zip};

use ff::Field;

use super::BETA_POLY_DEGREE;
use crate::{
    plonk::Expression,
    poly::{LagrangeCoeff, Polynomial},
    protostar::row_evaluator::{
        evaluated_poly::EvaluatedFrom2, PolyBooleanEvaluator, RowBooleanEvaluator,
    },
};

const G_POLY_DEGREE: usize = 1;
const THETA_POLY_DEGREE: usize = 1;
const H_POLY_DEGREE: usize = 1;

pub fn error_poly_lookup_inputs<F: Field>(
    num_rows: usize,
    selectors: &[BTreeSet<usize>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    challenges: [&[F]; 2],
    betas: [&Polynomial<F, LagrangeCoeff>; 2],
    advice: [&[Polynomial<F, LagrangeCoeff>]; 2],
    instance: [&[Polynomial<F, LagrangeCoeff>]; 2],
    input_expressions: &[Expression<F>],
    challenge_r: [&F; 2],
    challenge_thetas: [&[F]; 2],
    g: [&Polynomial<F, LagrangeCoeff>; 2],
) -> EvaluatedFrom2<F> {
    let num_inputs = input_expressions.len();
    let num_evals: usize = input_expressions
        .iter()
        .map(|poly| {
            poly.folding_degree() + BETA_POLY_DEGREE + G_POLY_DEGREE + THETA_POLY_DEGREE + 1
        })
        .max()
        .unwrap();

    let mut inputs_ev = RowBooleanEvaluator::new(
        input_expressions,
        challenges[0],
        challenges[1],
        vec![EvaluatedFrom2::new_empty(num_evals); num_inputs],
    );
    let mut beta_ev = PolyBooleanEvaluator::new(betas[0], betas[1], num_evals);
    let mut g_ev = PolyBooleanEvaluator::new(g[0], g[1], num_evals);

    let mut error_evals = EvaluatedFrom2::new_empty(num_evals);
    let mut row_evals = EvaluatedFrom2::new_empty(num_evals);

    let r_evals =
        EvaluatedFrom2::new_from_boolean_evals(*challenge_r[0], *challenge_r[1], num_evals);
    let thetas_evals: Vec<_> = zip(challenge_thetas[0].iter(), challenge_thetas[1].iter())
        .map(|(theta0, theta1)| EvaluatedFrom2::new_from_boolean_evals(*theta0, *theta1, num_evals))
        .collect();

    for row_idx in 0..num_rows {
        let beta_evals = beta_ev.evaluate_from_2(row_idx);
        let g_evals = g_ev.evaluate_from_2(row_idx);

        // Evaluate [w₁,ᵢ(D), …, wₖ,ᵢ(D)]
        let inputs_evals =
            inputs_ev.evaluate_all_from_2(row_idx, selectors, fixed, instance, advice);

        row_evals.set(&r_evals);

        // Evaluate r(D) + ∑ⱼ θⱼ(D)⋅wⱼ,ᵢ(D)
        for (input_evals, theta_evals) in zip(inputs_evals.iter(), thetas_evals.iter()) {
            row_evals.add_prod(input_evals, theta_evals);
        }

        // Compute eᵢ(D) = βᵢ(D)⋅(gᵢ(D)⋅[r(D) + ∑ⱼ θⱼ(D)⋅wⱼ,ᵢ(D)] - 1)
        row_evals *= g_evals;
        row_evals -= &F::ONE;
        row_evals *= beta_evals;

        // Evaluate e(D) += eᵢ(D)
        error_evals += &row_evals;
    }

    error_evals
}

pub fn error_poly_lookup_tables<F: Field>(
    num_rows: usize,
    selectors: &[BTreeSet<usize>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    challenges: [&[F]; 2],
    betas: [&Polynomial<F, LagrangeCoeff>; 2],
    advice: [&[Polynomial<F, LagrangeCoeff>]; 2],
    instance: [&[Polynomial<F, LagrangeCoeff>]; 2],
    table_expressions: &[Expression<F>],
    challenge_r: [&F; 2],
    challenge_thetas: [&[F]; 2],
    m: [&Polynomial<F, LagrangeCoeff>; 2],
    h: [&Polynomial<F, LagrangeCoeff>; 2],
) -> EvaluatedFrom2<F> {
    let num_tables = table_expressions.len();
    let num_evals = table_expressions
        .iter()
        .map(|poly| {
            poly.folding_degree() + BETA_POLY_DEGREE + H_POLY_DEGREE + THETA_POLY_DEGREE + 1
        })
        .max()
        .unwrap();

    let mut tables_ev = RowBooleanEvaluator::new(
        table_expressions,
        challenges[0],
        challenges[1],
        vec![EvaluatedFrom2::new_empty(num_evals); num_tables],
    );
    let mut beta_ev = PolyBooleanEvaluator::new(betas[0], betas[1], num_evals);
    let mut m_ev = PolyBooleanEvaluator::new(m[0], m[1], num_evals);
    let mut h_ev = PolyBooleanEvaluator::new(h[0], h[1], num_evals);

    let mut error_evals = EvaluatedFrom2::new_empty(num_evals);
    let mut row_evals = EvaluatedFrom2::new_empty(num_evals);

    let r_evals =
        EvaluatedFrom2::new_from_boolean_evals(*challenge_r[0], *challenge_r[1], num_evals);
    let thetas_evals: Vec<_> = zip(challenge_thetas[0].iter(), challenge_thetas[1].iter())
        .map(|(theta0, theta1)| EvaluatedFrom2::new_from_boolean_evals(*theta0, *theta1, num_evals))
        .collect();

    for row_idx in 0..num_rows {
        // Early exit if mᵢ(D) == 0
        if m[0][row_idx].is_zero_vartime() & m[1][row_idx].is_zero_vartime() {
            continue;
        }

        let beta_evals = beta_ev.evaluate_from_2(row_idx);
        let m_evals = m_ev.evaluate_from_2(row_idx);
        let h_evals = h_ev.evaluate_from_2(row_idx);

        // Evaluate [t₁,ᵢ(D), …, tₖ,ᵢ(D)]
        let tables_evals =
            tables_ev.evaluate_all_from_2(row_idx, selectors, fixed, instance, advice);

        // Evaluate r(D) + ∑ⱼ θⱼ(D)⋅tⱼ,ᵢ(D)
        row_evals.set(&r_evals);
        for (input_evals, theta_evals) in zip(tables_evals.iter(), thetas_evals.iter()) {
            row_evals.add_prod(input_evals, theta_evals);
        }

        // Compute eᵢ(D) = βᵢ(D)⋅(hᵢ(D)⋅[r(D) + ∑ⱼ θⱼ(D)⋅tⱼ,ᵢ(D)] - mᵢ(D))
        row_evals *= h_evals;
        row_evals -= m_evals;
        row_evals *= beta_evals;

        // Evaluate e(D) += eᵢ(D)
        error_evals += &row_evals;
    }

    error_evals
}

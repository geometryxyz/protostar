use std::iter::zip;

use ff::Field;

use crate::{
    plonk::Expression,
    poly::{LagrangeCoeff, Polynomial},
    protostar::row_evaluator::{boolean_evaluations, PolyBooleanEvaluator, RowBooleanEvaluator},
};

use super::{BETA_POLY_DEGREE, STARTING_EVAL_IDX};

pub fn error_poly_lookup_inputs<F: Field>(
    num_rows: usize,
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    challenges: [&[Vec<F>]; 2],
    betas: [&Polynomial<F, LagrangeCoeff>; 2],
    advice: [&[Polynomial<F, LagrangeCoeff>]; 2],
    instance: [&[Polynomial<F, LagrangeCoeff>]; 2],
    input_expressions: &[Expression<F>],
    challenge_r: [&F; 2],
    challenge_thetas: [&Vec<F>; 2],
    g: [&Polynomial<F, LagrangeCoeff>; 2],
) -> Vec<F> {
    const G_POLY_DEGREE: usize = 1;
    const THETA_POLY_DEGREE: usize = 1;

    let num_inputs = input_expressions.len();
    let inputs_num_evals: Vec<usize> = input_expressions
        .iter()
        .map(|poly| {
            poly.folding_degree() + BETA_POLY_DEGREE + G_POLY_DEGREE + THETA_POLY_DEGREE + 1
        })
        .collect();
    let num_evals = *inputs_num_evals.iter().max().unwrap();

    let mut inputs_ev = RowBooleanEvaluator::new(
        input_expressions,
        challenges[0],
        challenges[1],
        vec![vec![F::ZERO; num_evals]; num_inputs],
    );
    let mut beta_ev = PolyBooleanEvaluator::new(betas[0], betas[1], num_evals);
    let mut g_ev = PolyBooleanEvaluator::new(g[0], g[1], num_evals);

    let mut error_evals = vec![F::ZERO; num_evals];
    let mut row_evals = vec![F::ZERO; num_evals];

    let r_evals = boolean_evaluations(*challenge_r[0], *challenge_r[1], num_evals);
    let thetas_evals: Vec<_> = zip(challenge_thetas[0].iter(), challenge_thetas[1].iter())
        .map(|(theta0, theta1)| boolean_evaluations(*theta0, *theta1, num_evals))
        .collect();

    for row_idx in 0..num_rows {
        let beta_evals = beta_ev.evaluate(row_idx);
        let g_evals = g_ev.evaluate(row_idx);

        // Evaluate [w₁,ᵢ(D), …, wₖ,ᵢ(D)]
        let inputs_evals = inputs_ev.evaluate_all_from(
            STARTING_EVAL_IDX,
            row_idx,
            selectors,
            fixed,
            instance[0],
            instance[1],
            advice[0],
            advice[1],
        );

        row_evals.clear();
        row_evals.extend_from_slice(&r_evals);

        // Evaluate r(D) + ∑ⱼ θⱼ(D)⋅wⱼ,ᵢ(D)
        for (input_idx, input_evals) in inputs_evals.iter().enumerate() {
            let theta_evals = &thetas_evals[input_idx];
            for (eval_idx, row_eval) in row_evals.iter_mut().enumerate().skip(STARTING_EVAL_IDX) {
                let theta_eval = &theta_evals[eval_idx];
                let input_eval = &input_evals[eval_idx];
                *row_eval += *theta_eval * input_eval;
            }
        }

        // Compute eᵢ(D) = βᵢ(D)⋅(gᵢ(D)⋅[r(D) + ∑ⱼ θⱼ(D)⋅wⱼ,ᵢ(D)] - 1)
        for (eval_idx, row_eval) in row_evals.iter_mut().enumerate().skip(STARTING_EVAL_IDX) {
            *row_eval *= g_evals[eval_idx];
            *row_eval -= F::ONE;
            *row_eval *= beta_evals[eval_idx];
        }

        // Evaluate e(D) += eᵢ(D)
        for (eval_idx, error_eval) in error_evals.iter_mut().enumerate().skip(STARTING_EVAL_IDX) {
            *error_eval += row_evals[eval_idx];
        }
    }

    error_evals
}

pub fn error_poly_lookup_tables<F: Field>(
    num_rows: usize,
    selectors: &[Vec<bool>],
    fixed: &[Polynomial<F, LagrangeCoeff>],
    challenges: [&[Vec<F>]; 2],
    betas: [&Polynomial<F, LagrangeCoeff>; 2],
    advice: [&[Polynomial<F, LagrangeCoeff>]; 2],
    instance: [&[Polynomial<F, LagrangeCoeff>]; 2],
    table_expressions: &[Expression<F>],
    challenge_r: [&F; 2],
    challenge_thetas: [&Vec<F>; 2],
    m: [&Polynomial<F, LagrangeCoeff>; 2],
    h: [&Polynomial<F, LagrangeCoeff>; 2],
) -> Vec<F> {
    const H_POLY_DEGREE: usize = 1;
    const THETA_POLY_DEGREE: usize = 1;

    let num_tables = table_expressions.len();
    let tables_num_evals: Vec<usize> = table_expressions
        .iter()
        .map(|poly| {
            poly.folding_degree() + BETA_POLY_DEGREE + H_POLY_DEGREE + THETA_POLY_DEGREE + 1
        })
        .collect();
    let num_evals = *tables_num_evals.iter().max().unwrap();

    let mut tables_ev = RowBooleanEvaluator::new(
        table_expressions,
        challenges[0],
        challenges[1],
        vec![vec![F::ZERO; num_evals]; num_tables],
    );
    let mut beta_ev = PolyBooleanEvaluator::new(betas[0], betas[1], num_evals);
    let mut m_ev = PolyBooleanEvaluator::new(m[0], m[1], num_evals);
    let mut h_ev = PolyBooleanEvaluator::new(h[0], h[1], num_evals);

    let mut error_evals = vec![F::ZERO; num_evals];
    let mut row_evals = vec![F::ZERO; num_evals];

    let r_evals = boolean_evaluations(*challenge_r[0], *challenge_r[1], num_evals);
    let thetas_evals: Vec<_> = zip(challenge_thetas[0].iter(), challenge_thetas[1].iter())
        .map(|(theta0, theta1)| boolean_evaluations(*theta0, *theta1, num_evals))
        .collect();

    for row_idx in 0..num_rows {
        // Early exit if mᵢ(D) == 0
        if m[0][row_idx].is_zero_vartime() & m[1][row_idx].is_zero_vartime() {
            continue;
        }

        let beta_evals = beta_ev.evaluate(row_idx);
        let m_evals = m_ev.evaluate(row_idx);
        let h_evals = h_ev.evaluate(row_idx);

        // Evaluate [t₁,ᵢ(D), …, tₖ,ᵢ(D)]
        let tables_evals = tables_ev.evaluate_all_from(
            STARTING_EVAL_IDX,
            row_idx,
            selectors,
            fixed,
            instance[0],
            instance[1],
            advice[0],
            advice[1],
        );

        // Evaluate r(D) + ∑ⱼ θⱼ(D)⋅tⱼ,ᵢ(D)
        row_evals.clear();
        row_evals.extend_from_slice(&r_evals);
        for (input_idx, table_evals) in tables_evals.iter().enumerate() {
            let theta_evals = &thetas_evals[input_idx];
            for (eval_idx, row_eval) in row_evals.iter_mut().enumerate().skip(STARTING_EVAL_IDX) {
                let theta_eval = &theta_evals[eval_idx];
                let table_eval = &table_evals[eval_idx];
                *row_eval += *theta_eval * table_eval;
            }
        }

        // Compute eᵢ(D) = βᵢ(D)⋅(hᵢ(D)⋅[r(D) + ∑ⱼ θⱼ(D)⋅tⱼ,ᵢ(D)] - mᵢ(D))
        for (eval_idx, row_eval) in row_evals.iter_mut().enumerate().skip(STARTING_EVAL_IDX) {
            *row_eval *= h_evals[eval_idx];
            *row_eval -= m_evals[eval_idx];
            *row_eval *= beta_evals[eval_idx];
        }

        // Evaluate e(D) += eᵢ(D)
        for (eval_idx, error_eval) in error_evals.iter_mut().enumerate().skip(STARTING_EVAL_IDX) {
            *error_eval += row_evals[eval_idx];
        }
    }

    error_evals
}

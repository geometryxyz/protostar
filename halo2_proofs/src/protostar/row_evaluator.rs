use core::num;
use std::{collections::BTreeSet, iter::zip};

use ff::Field;

use crate::{
    plonk::Expression,
    poly::{LagrangeCoeff, Polynomial},
};

use self::{
    evaluated_poly::EvaluatedFrom2, queried_expression::QueriedExpression, queries::Queries,
    row::Row,
};

pub(crate) mod evaluated_poly;
pub(crate) mod interpolate;
mod queried_expression;
mod queries;
mod row;

/// Structure for efficiently evaluating a set of polynomials over many rows.
pub struct RowBooleanEvaluator<F: Field> {
    // Boolean evaluations of queried challenges
    challenges_evals: Vec<Vec<F>>,
    // List of polynomial expressions Gⱼ
    queried_polys: Vec<QueriedExpression<F>>,

    // Buffer for storing all row values
    row: Row<F>,

    // Buffer for storing and returning the evaluations of all polynomials
    // in the values stored in `row`.
    polys_evals_buffer: Vec<EvaluatedFrom2<F>>,
}

impl<F: Field> RowBooleanEvaluator<F> {
    /// Given a set of polynomials all belonging to the same gate,
    /// prepare the evaluator to compute up to `max_num_evals` polynomial
    /// evaluations of the given polynomials.
    pub fn new(
        polys: &[Expression<F>],
        challenges_acc: &[F],
        challenges_new: &[F],
        polys_evals_buffer: Vec<EvaluatedFrom2<F>>,
    ) -> Self {
        debug_assert_eq!(polys.len(), polys_evals_buffer.len());
        let queries = Queries::from_polys(&polys);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        // Compute the maximum
        let max_num_evals = polys_evals_buffer
            .iter()
            .map(|poly_evals| poly_evals.num_evals())
            .max()
            .unwrap();

        // Compute all boolean evaluations of the queried challenges for this gate
        let queried_challenges_acc = queries.queried_challenges(challenges_acc);
        let queried_challenges_new = queries.queried_challenges(challenges_new);
        let challenges_evals = boolean_evaluations_vec(
            queried_challenges_acc,
            queried_challenges_new,
            max_num_evals,
        );

        let row = Row::new(queries, max_num_evals);
        Self {
            challenges_evals,
            queried_polys,
            row,
            polys_evals_buffer,
        }
    }

    /// Evaluates all polynomials at X = 2, 3, ..., returning the evaluations as a slice.
    pub fn evaluate_all_from_2(
        &mut self,
        row_idx: usize,
        selectors: &[BTreeSet<usize>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: [&[Polynomial<F, LagrangeCoeff>]; 2],
        advice: [&[Polynomial<F, LagrangeCoeff>]; 2],
    ) -> &[EvaluatedFrom2<F>] {
        self.row
            .populate_all_evaluated(row_idx, selectors, fixed, instance, advice);

        for (poly_idx, poly) in self.queried_polys.iter().enumerate() {
            let poly_evals = &mut self.polys_evals_buffer[poly_idx];

            for (eval_idx, poly_eval) in poly_evals.evals.iter_mut().enumerate() {
                // The evaluations in `poly_eval` start from 2, so we increase eval_idx by 2
                // to compute the evaluation at the correct point.
                let eval_idx = eval_idx + 2;
                *poly_eval = self
                    .row
                    .evaluate_at(eval_idx, poly, &self.challenges_evals[eval_idx]);
            }
        }
        &self.polys_evals_buffer
    }
}

pub struct RowEvaluator<F: Field> {
    // Boolean evaluations of queried challenges
    challenges: Vec<F>,
    // List of polynomial expressions Gⱼ
    queried_polys: Vec<QueriedExpression<F>>,

    // buffer for storing all row values values
    row: Row<F>,

    // Buffer for storing the polynomial evaluations
    polys_evals_buffer: Vec<F>,
}

impl<F: Field> RowEvaluator<F> {
    // Prepares a `RowEvaluator` for the polynomial `polys`
    pub fn new(polys: &[Expression<F>], challenges: &[F]) -> Self {
        let queries = Queries::from_polys(&polys);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        let queried_challenges = queries.queried_challenges(challenges);

        let row = Row::new(queries, 1);
        Self {
            challenges: queried_challenges,
            queried_polys,
            row,
            polys_evals_buffer: vec![F::ZERO; polys.len()],
        }
    }

    /// Evaluates all polynomials at the given `row_idx`, returning a slice containing the evaluations.
    pub fn evaluate(
        &mut self,
        row_idx: usize,
        selectors: &[BTreeSet<usize>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
    ) -> &[F] {
        // Fetch the values for the row
        self.row
            .populate_all(row_idx, selectors, fixed, instance, advice);

        // Iterate over each polynomial constraint Gⱼ
        for (poly_idx, poly) in self.queried_polys.iter().enumerate() {
            self.polys_evals_buffer[poly_idx] = self.row.evaluate_at(0, poly, &self.challenges);
        }
        &self.polys_evals_buffer
    }
}

/// For a sequence of n linear polynomial [p₁(X), p₂(X), …, pₙ(X)], given as
///  `evals0` = [p₁(0), p₂(0)  , …, pₙ(0)], `evals1` = [p₁(1), p₂(1), …, pₙ(1)],
/// return the vector of all m = `num_evals` evaluations
/// [
///  [p₁(0)  , p₂(0)  , …, pₙ(0)  ],
///  [p₁(1)  , p₂(1)  , …, pₙ(1)  ],
///  …
///  [p₁(m-1), p₂(m-1), …, pₙ(m-1)],
/// ]
fn boolean_evaluations_vec<F: Field>(
    evals0: Vec<F>,
    evals1: Vec<F>,
    num_evals: usize,
) -> Vec<Vec<F>> {
    debug_assert!(2 <= num_evals);

    let n = evals0.len();
    debug_assert_eq!(evals1.len(), n);

    let diffs: Vec<F> = zip(evals0.iter(), evals1.iter())
        .map(|(eval0, eval1)| *eval1 - eval0)
        .collect();

    let mut result = vec![vec![F::ZERO; n]; num_evals];
    result[0] = evals0;
    result[1] = evals1;
    for i in 2..num_evals {
        for (eval_idx, diff) in diffs.iter().enumerate() {
            result[i][eval_idx] = result[i - 1][eval_idx] + diff;
        }
    }
    result
}

/// Given two vectors p₀, p₁, this structure can be used to efficiently evaluate
/// pᵢ(D) for D = {2, 3, …, `max_evals`}, where pᵢ(X) = (1−X)·₀,ᵢ + X·₁,ᵢ
pub struct PolyBooleanEvaluator<'a, F: Field> {
    poly_0: &'a Polynomial<F, LagrangeCoeff>,
    poly_1: &'a Polynomial<F, LagrangeCoeff>,
    evals: EvaluatedFrom2<F>,
}

impl<'a, F: Field> PolyBooleanEvaluator<'a, F> {
    /// Create a new evaluator for the domain D = {2, 3, …, `max_evals`}
    pub fn new(
        poly_0: &'a Polynomial<F, LagrangeCoeff>,
        poly_1: &'a Polynomial<F, LagrangeCoeff>,
        num_evals: usize,
    ) -> Self {
        Self {
            poly_0,
            poly_1,
            evals: EvaluatedFrom2::new_empty(num_evals),
        }
    }

    /// Return a reference to pᵢ(D) where i = `row_idx`.
    pub fn evaluate_from_2(&mut self, row_idx: usize) -> &EvaluatedFrom2<F> {
        self.evals
            .reset_with_boolean_evals(self.poly_0[row_idx], self.poly_1[row_idx]);
        &self.evals
    }
}

use core::num;
use std::iter::zip;

use ff::Field;

use crate::{
    plonk::Expression,
    poly::{LagrangeCoeff, Polynomial},
};

use self::{queried_expression::QueriedExpression, queries::Queries, row::Row};

mod queried_expression;
mod queries;
mod row;

/// Structure for efficiently evaluating a set of polynomials over many rows.
///
pub struct RowBooleanEvaluator<F: Field> {
    // Boolean evaluations of queried challenges
    challenges_evals: Vec<Vec<F>>,
    // List of polynomial expressions Gⱼ
    queried_polys: Vec<QueriedExpression<F>>,

    // Buffer for storing all row values
    row: Row<F>,

    // Buffer for storing and returning the evaluations of all polynomials
    // in the values stored in `row`.
    polys_evals_buffer: Vec<Vec<F>>,
}

impl<F: Field> RowBooleanEvaluator<F> {
    /// Given a set of polynomials all belonging to the same gate,
    /// prepare the evaluator to compute up to `max_num_evals` polynomial
    /// evaluations of the given polynomials.
    pub fn new(
        polys: &[Expression<F>],
        challenges_acc: &[Vec<F>],
        challenges_new: &[Vec<F>],
        polys_evals_buffer: Vec<Vec<F>>,
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
            .map(|poly_evals| poly_evals.len())
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

    /// Evaluates all polynomials in the gate, at X = `from_eval_idx`, ...,
    /// The total number of evaluations is defined by the legth of each list in
    /// `error_polys_evals`.
    /// That is, the polynomial constraint Gⱼ will have its evaluations stored in
    /// `error_polys_evals[j]`.
    pub fn evaluate_all_from(
        &mut self,
        from_eval_idx: usize,
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance_acc: &[Polynomial<F, LagrangeCoeff>],
        instance_new: &[Polynomial<F, LagrangeCoeff>],
        advice_acc: &[Polynomial<F, LagrangeCoeff>],
        advice_new: &[Polynomial<F, LagrangeCoeff>],
    ) -> &[Vec<F>] {
        self.row.populate_all_evaluated(
            row_idx,
            selectors,
            fixed,
            instance_acc,
            instance_new,
            advice_acc,
            advice_new,
        );

        for (poly_idx, poly) in self.queried_polys.iter().enumerate() {
            let poly_evals = &mut self.polys_evals_buffer[poly_idx];

            for (eval_idx, poly_eval) in poly_evals.iter_mut().enumerate().skip(from_eval_idx) {
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

    // buffer for storing all row values
    row: Row<F>,

    polys_evals_buffer: Vec<F>,
}

impl<F: Field> RowEvaluator<F> {
    // Gate evaluator for when only a single evaluation is required
    pub fn new(polys: &[Expression<F>], challenges: &[Vec<F>]) -> Self {
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

    /// Evaluates all polynomials in the gate at the given `row_idx`, storing the result in `evals`.
    pub fn evaluate(
        &mut self,
        row_idx: usize,
        selectors: &[Vec<bool>],
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

/// For a linear polynomial p(X) such that p(0) = eval0, p(1) = eval1,
/// return a vector [p(0), p(1), ..., p(num_evals-1)]
pub fn boolean_evaluations<F: Field>(eval0: F, eval1: F, num_evals: usize) -> Vec<F> {
    debug_assert!(2 <= num_evals);
    let mut result = vec![F::ZERO; num_evals];

    let diff = eval1 - eval0;

    result[0] = eval1;
    result[1] = eval1;
    for i in 2..num_evals {
        result[i] = result[i - 1] + diff;
    }
    result
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
pub fn boolean_evaluations_vec<F: Field>(
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

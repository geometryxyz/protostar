use std::{collections::BTreeSet, iter::zip};

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    plonk::{AdviceQuery, Challenge, Expression, FixedQuery, Gate, InstanceQuery, Selector},
    poly::{LagrangeCoeff, Polynomial, Rotation},
    protostar::keygen::ProvingKey,
};

use super::{
    boolean_evaluations_vec, boolean_evaluations_vec_skip_2,
    row::{QueriedExpression, Row, RowQueries},
    Accumulator, NUM_EXTRA_EVALUATIONS, NUM_SKIPPED_EVALUATIONS,
};

pub struct GateEvaluator<F: Field> {
    // Target number of evaluations for each polynomial.
    num_evals: Vec<usize>,
    max_num_evals: usize,

    // Boolean evaluations of queried challenges
    challenges_evals: Vec<Vec<F>>,
    // List of polynomial expressions Gⱼ
    queried_polys: Vec<QueriedExpression<F>>,

    // buffer for storing all row values
    row: Row<F>,
    errors_evals: Vec<Vec<F>>,
}

impl<F: Field> GateEvaluator<F> {
    pub fn new(
        polys: &[Expression<F>],
        challenges_acc: &[Vec<F>],
        challenges_new: &[Vec<F>],
        num_rows_i: i32,
    ) -> Self {
        let degrees: Vec<_> = polys.iter().map(|poly| poly.folding_degree()).collect();

        let queries = RowQueries::from_polys(&polys, num_rows_i);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        // Each `poly` Gⱼ(X) has degree dⱼ and dⱼ+1 coefficients,
        // therefore we need at least dⱼ+1 evaluations of Gⱼ(X) to recover
        // the coefficients.
        let num_evals: Vec<_> = degrees
            .iter()
            .map(|d| d + 1 + NUM_EXTRA_EVALUATIONS - NUM_SKIPPED_EVALUATIONS)
            .collect();
        // maximum number of evaluations over all Gⱼ(X)
        let max_num_evals = *num_evals.iter().max().unwrap();

        // Compute all boolean evaluations of the queried challenges for this gate
        let queried_challenges_acc = queries.queried_challenges(challenges_acc);
        let queried_challenges_new = queries.queried_challenges(challenges_new);

        let challenges_evals: Vec<_> =
            boolean_evaluations_vec_skip_2(queried_challenges_acc, queried_challenges_new)
                .take(max_num_evals)
                .collect();

        // for each polynomial, allocate a buffer for storing all the evaluations
        let errors_evals = num_evals.iter().map(|d| vec![F::ZERO; *d]).collect();

        let row = Row::new(queries);
        Self {
            num_evals,
            max_num_evals,
            challenges_evals,
            queried_polys,
            row,
            errors_evals,
        }
    }

    // Gate evaluator for when only a single evaluation is required
    pub fn new_single(polys: &[Expression<F>], challenges: &[Vec<F>], num_rows_i: i32) -> Self {
        let queries = RowQueries::from_polys(&polys, num_rows_i);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        let queried_challenges = queries.queried_challenges(challenges);

        let row = Row::new(queries);
        Self {
            num_evals: vec![1; polys.len()],
            max_num_evals: 1,
            challenges_evals: vec![queried_challenges],
            queried_polys,
            row,
            errors_evals: vec![vec![F::ZERO]; polys.len()],
        }
    }

    /// Evaluates the error polynomial for the populated row.
    /// Returns `None` if the common selector for the gate is false,
    /// otherwise returns a list of vectors containing the evaluations for
    /// each `poly` Gⱼ(X) in `gate`.
    pub fn evaluate_and_accumulate_errors(
        &mut self,
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance_acc: &[Polynomial<F, LagrangeCoeff>],
        instance_new: &[Polynomial<F, LagrangeCoeff>],
        advice_acc: &[Polynomial<F, LagrangeCoeff>],
        advice_new: &[Polynomial<F, LagrangeCoeff>],
    ) -> &[Vec<F>] {
        // Fill the row with data from both transcripts
        self.row.populate_selectors(row_idx, selectors);
        self.row.populate_fixed(row_idx, fixed);
        self.row
            .populate_advice_evals_skip_1(row_idx, advice_acc, advice_new);
        self.row
            .populate_instance_evals_skip_1(row_idx, instance_acc, instance_new);

        let max_num_evals = self.max_num_evals;

        // Iterate over all evaluations points X = 2, ..., max_num_evals - 2
        for eval_idx in 0..max_num_evals {
            // Compute the next linear evaluation of the interpolation of instance and advice values.
            self.row.populate_next_evals();

            // Iterate over each polynomial constraint Gⱼ, along with its required number of evaluations
            for (poly_idx, (poly, num_evals)) in
                zip(self.queried_polys.iter(), self.num_evals.iter()).enumerate()
            {
                // If the `eval_idx` X is larger than the required number of evaluations for the current poly,
                // we don't evaluate it and continue to the next poly.
                if eval_idx > *num_evals {
                    continue;
                }
                self.errors_evals[poly_idx][eval_idx] =
                    self.row.evaluate(poly, &self.challenges_evals[eval_idx]);
            }
        }
        &self.errors_evals
    }

    /// Evaluates the error polynomial for the populated row.
    /// Returns `None` if the common selector for the gate is false,
    /// otherwise returns a list of vectors containing the evaluations for
    /// each `poly` Gⱼ(X) in `gate`.
    pub fn evaluate_single(
        &mut self,
        evals: &mut [F],
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
    ) {
        self.row
            .populate_all(row_idx, selectors, fixed, instance, advice);

        // Iterate over each polynomial constraint Gⱼ, along with its required number of evaluations
        for (poly_idx, poly) in self.queried_polys.iter().enumerate() {
            evals[poly_idx] = self.row.evaluate(poly, &self.challenges_evals[0]);
        }
    }

    /// Returns a zero-initialized error polynomial for storing all evaluations of
    /// the polynomials for this gate.
    pub fn empty_error_polynomials(&self) -> Vec<Vec<F>> {
        self.num_evals
            .iter()
            .map(|num_eval| vec![F::ZERO; *num_eval])
            .collect()
    }
}

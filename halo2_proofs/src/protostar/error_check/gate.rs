use std::{collections::BTreeSet, iter::zip};

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    plonk::{AdviceQuery, Challenge, Expression, FixedQuery, Gate, InstanceQuery, Selector},
    poly::{LagrangeCoeff, Polynomial, Rotation},
    protostar::keygen::ProvingKey,
};

use super::{
    boolean_evaluations_vec,
    row::{QueriedExpression, Queries, Row},
    Accumulator, BETA_POLY_DEGREE,
};

pub struct GateEvaluator<F: Field> {
    // Boolean evaluations of queried challenges
    challenges_evals: Vec<Vec<F>>,
    // List of polynomial expressions Gⱼ
    queried_polys: Vec<QueriedExpression<F>>,

    // buffer for storing all row values
    row: Row<F>,
}

impl<F: Field> GateEvaluator<F> {
    /// Given a set of polynomials all belonging to the same gate,
    /// prepare the evaluator to compute up to `max_num_evals` polynomial
    /// evaluations of the given polynomials.
    pub fn new(
        polys: &[Expression<F>],
        challenges_acc: &[Vec<F>],
        challenges_new: &[Vec<F>],
        max_num_evals: usize,
    ) -> Self {
        let queries = Queries::from_polys(&polys);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        // Compute all boolean evaluations of the queried challenges for this gate
        let queried_challenges_acc = queries.queried_challenges(challenges_acc);
        let queried_challenges_new = queries.queried_challenges(challenges_new);
        let challenges_evals: Vec<_> =
            boolean_evaluations_vec(queried_challenges_acc, queried_challenges_new)
                .take(max_num_evals)
                .collect();

        let row = Row::new(queries, max_num_evals);
        Self {
            challenges_evals,
            queried_polys,
            row,
        }
    }

    // Gate evaluator for when only a single evaluation is required
    pub fn new_single(polys: &[Expression<F>], challenges: &[Vec<F>]) -> Self {
        let queries = Queries::from_polys(&polys);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        let queried_challenges = queries.queried_challenges(challenges);

        let row = Row::new(queries, 1);
        Self {
            challenges_evals: vec![queried_challenges],
            queried_polys,
            row,
        }
    }

    /// Evaluates all polynomials in the gate, at X = `from_eval_idx`, ...,
    /// The total number of evaluations is defined by the legth of each list in
    /// `error_polys_evals`.
    /// That is, the polynomial constraint Gⱼ will have its evaluations stored in
    /// `error_polys_evals[j]`.
    pub fn evaluate_all_from(
        &mut self,
        error_polys_evals: &mut [Vec<F>],
        from_eval_idx: usize,
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance_acc: &[Polynomial<F, LagrangeCoeff>],
        instance_new: &[Polynomial<F, LagrangeCoeff>],
        advice_acc: &[Polynomial<F, LagrangeCoeff>],
        advice_new: &[Polynomial<F, LagrangeCoeff>],
    ) {
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
            let error_poly_evals = &mut error_polys_evals[poly_idx];

            for (eval_idx, error_poly_eval) in
                error_poly_evals.iter_mut().enumerate().skip(from_eval_idx)
            {
                *error_poly_eval =
                    self.row
                        .evaluate_at(eval_idx, poly, &self.challenges_evals[eval_idx]);
            }
        }
    }

    /// Evaluates all polynomials in the gate at the given `row_idx`, storing the result in `evals`.
    pub fn evaluate_single(
        &mut self,
        evals: &mut [F],
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
    ) {
        // Fetch the values for the row
        self.row
            .populate_all(row_idx, selectors, fixed, instance, advice);

        // Iterate over each polynomial constraint Gⱼ
        for (poly_idx, poly) in self.queried_polys.iter().enumerate() {
            evals[poly_idx] = self.row.evaluate_at(0, poly, &self.challenges_evals[0]);
        }
    }
}

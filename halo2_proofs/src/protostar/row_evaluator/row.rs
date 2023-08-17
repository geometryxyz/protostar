use std::{collections::BTreeSet, iter::zip};

use ff::Field;

use crate::{
    plonk::Expression,
    poly::{LagrangeCoeff, Polynomial, Rotation},
};

use super::{queried_expression::QueriedExpression, queries::Queries};

/// A `Row` contains buffers for storing the values defined by the queries in `RowQueries`.
/// Values are populated, possibly interpolated, and then evaluated for a given `QueriedPolynomial`.
pub struct Row<F: Field> {
    selectors: Vec<F>,
    fixed: Vec<F>,
    instance_evals: Vec<Vec<F>>,
    instance_diff: Vec<F>,
    advice_evals: Vec<Vec<F>>,
    advice_diff: Vec<F>,
    queries: Queries,
    num_evals: usize,
}

impl<F: Field> Row<F> {
    /// Allocates buffers for fetching the values defined by `Queries`.
    /// After being populated, the advice and instance values will be interpolated in `num_evals` points.
    pub fn new(queries: Queries, num_evals: usize) -> Self {
        Self {
            selectors: vec![F::ZERO; queries.selectors.len()],
            fixed: vec![F::ZERO; queries.fixed.len()],
            instance_evals: vec![vec![F::ZERO; queries.instance.len()]; num_evals],
            advice_evals: vec![vec![F::ZERO; queries.advice.len()]; num_evals],
            instance_diff: vec![F::ZERO; queries.instance.len()],
            advice_diff: vec![F::ZERO; queries.advice.len()],
            queries,
            num_evals,
        }
    }

    /// Fills the local variables buffers with data from a single transcript
    pub fn populate_all(
        &mut self,
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
    ) {
        self.populate_selectors(row_idx, selectors);
        self.populate_fixed(row_idx, fixed);
        self.populate_instance(row_idx, 0, instance);
        self.populate_advice(row_idx, 0, advice);
    }

    /// Fills the local variables buffers with data from the accumulator and new transcript
    pub fn populate_all_evaluated(
        &mut self,
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: [&[Polynomial<F, LagrangeCoeff>]; 2],
        advice: [&[Polynomial<F, LagrangeCoeff>]; 2],
    ) {
        self.populate_selectors(row_idx, selectors);
        self.populate_fixed(row_idx, fixed);

        self.populate_advice(row_idx, 0, advice[0]);
        self.populate_advice(row_idx, 1, advice[1]);
        let num_advice = self.queries.advice.len();

        for i in 0..num_advice {
            self.advice_diff[i] = self.advice_evals[1][i] - self.advice_evals[0][i]
        }
        for eval_idx in 2..self.num_evals {
            for i in 0..num_advice {
                self.advice_evals[eval_idx][i] =
                    self.advice_evals[eval_idx - 1][i] + self.advice_diff[i];
            }
        }

        self.populate_instance(row_idx, 0, instance[0]);
        self.populate_instance(row_idx, 1, instance[1]);
        let num_instance = self.queries.instance.len();
        for i in 0..num_instance {
            self.instance_diff[i] = self.instance_evals[1][i] - self.instance_evals[0][i]
        }
        for eval_idx in 2..self.num_evals {
            for i in 0..num_instance {
                self.instance_evals[eval_idx][i] =
                    self.instance_evals[eval_idx - 1][i] + self.instance_diff[i];
            }
        }
    }

    /// Evaluate `poly` with the current values stored in the buffers.
    pub fn evaluate_at(&self, eval_idx: usize, poly: &QueriedExpression<F>, challenges: &[F]) -> F {
        // evaluate the j-th constraint G_j at X = eval_idx
        poly.evaluate(
            &|constant| constant,
            &|selector_idx| self.selectors[selector_idx],
            &|fixed_idx| self.fixed[fixed_idx],
            &|advice_idx| self.advice_evals[eval_idx][advice_idx],
            &|instance_idx| self.instance_evals[eval_idx][instance_idx],
            &|challenge_idx| challenges[challenge_idx],
            &|negated| -negated,
            &|sum_a, sum_b| sum_a + sum_b,
            &|prod_a, prod_b| prod_a * prod_b,
            &|scaled, v| scaled * v,
        )
    }

    /// Fetch the queried selectors.
    fn populate_selectors(&mut self, row_idx: usize, columns: &[Vec<bool>]) {
        for (row_value, column_idx) in self.selectors.iter_mut().zip(self.queries.selectors.iter())
        {
            *row_value = if columns[*column_idx][row_idx] {
                F::ONE
            } else {
                F::ZERO
            }
        }
    }

    /// Fetch the row values from queried fixed columns.
    fn populate_fixed(&mut self, row_idx: usize, columns: &[Polynomial<F, LagrangeCoeff>]) {
        Self::fill_row_with_rotations(&mut self.fixed, row_idx, &self.queries.fixed, columns)
    }

    /// Fetch the row values from queried instance columns.
    fn populate_instance(
        &mut self,
        row_idx: usize,
        eval_idx: usize,
        columns: &[Polynomial<F, LagrangeCoeff>],
    ) {
        Self::fill_row_with_rotations(
            &mut self.instance_evals[eval_idx],
            row_idx,
            &self.queries.instance,
            columns,
        )
    }

    /// Fetch the row values from queried advice columns.
    fn populate_advice(
        &mut self,
        row_idx: usize,
        eval_idx: usize,
        columns: &[Polynomial<F, LagrangeCoeff>],
    ) {
        Self::fill_row_with_rotations(
            &mut self.advice_evals[eval_idx],
            row_idx,
            &self.queries.advice,
            columns,
        )
    }

    fn fill_row_with_rotations(
        row: &mut [F],
        row_idx: usize,

        queries: &[(usize, Rotation)],
        columns: &[Polynomial<F, LagrangeCoeff>],
    ) {
        let row_len = row.len();
        debug_assert_eq!(queries.len(), row_len);

        for (row_value, (column_idx, rotation)) in row.iter_mut().zip(queries.iter()) {
            // ignore overflow since these should not occur in gates
            let row_idx = (row_idx as i32 + rotation.0) as usize;
            // let row_idx = (((row_idx as i32) + rotation.0).rem_euclid(num_rows_i)) as usize;
            *row_value = columns[*column_idx][row_idx]
        }
    }
}

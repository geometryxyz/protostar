use std::{
    iter::zip,
    ops::{AddAssign, Range},
};

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    arithmetic::{field_integers, lagrange_interpolate},
    poly::{LagrangeCoeff, Polynomial},
    protostar::{accumulator::Accumulator, ProvingKey},
};

use super::{
    expression::{IndexedExpression, QueriedExpression, QueryType},
    Data, LookupData,
};

/// Defines a QueriedExpression where leaves are references values from two accumulators.
pub struct Paired<'a, F> {
    _marker: std::marker::PhantomData<&'a F>,
}

impl<'a, F: Field> QueryType for Paired<'a, F> {
    type F = F;
    type Challenge = [&'a F; 2];
    type Fixed = &'a Polynomial<F, LagrangeCoeff>;
    type Witness = [&'a Polynomial<F, LagrangeCoeff>; 2];
}

impl<'a, F: Field> Paired<'a, F> {
    pub fn new_data<C>(
        pk: &'a ProvingKey<C>,
        acc0: &'a Accumulator<C>,
        acc1: &'a Accumulator<C>,
    ) -> Data<Self>
    where
        C: CurveAffine<ScalarExt = F>,
    {
        let selectors: Vec<_> = pk.selectors_polys.iter().collect();
        let fixed: Vec<_> = pk.fixed_polys.iter().collect();

        let instance: Vec<_> = zip(&acc0.instance.committed, &acc1.instance.committed)
            .map(|(i0, i1)| [&i0.values, &i1.values])
            .collect();

        let advice: Vec<_> = zip(&acc0.advice.committed, &acc1.advice.committed)
            .map(|(a0, a1)| [&a0.values, &a1.values])
            .collect();

        let challenges: Vec<_> = zip(&acc0.advice.challenges, &acc1.advice.challenges)
            .map(|(c0, c1)| [c0, c1])
            .collect();

        let beta = [&acc0.beta.beta.values, &acc1.beta.beta.values];

        let lookups: Vec<_> = zip(&acc0.lookups, &acc1.lookups)
            .map(|(lookup0, lookup1)| {
                let m = [&lookup0.m.values, &lookup1.m.values];
                let g = [&lookup0.g.values, &lookup1.g.values];
                let h = [&lookup0.h.values, &lookup1.h.values];
                let thetas: Vec<_> = zip(&lookup0.thetas, &lookup1.thetas)
                    .map(|(theta0, theta1)| [theta0, theta1])
                    .collect();
                let r = [&lookup0.r, &lookup1.r];
                LookupData { m, g, h, thetas, r }
            })
            .collect();

        let ys: Vec<_> = zip(&acc0.ys, &acc1.ys).map(|(y0, y1)| [y0, y1]).collect();

        Data::<Self> {
            fixed,
            selectors,
            instance,
            advice,
            challenges,
            beta,
            lookups,
            ys,
            num_rows: pk.num_rows,
        }
    }

    pub fn evaluate_compressed_polynomial(
        expr: QueriedExpression<Self>,
        rows: Range<usize>,
        num_rows: usize,
    ) -> Vec<F> {
        let indexed = IndexedExpression::<Paired<'a, F>>::new(expr);
        let num_evals = indexed.expr.degree() + 1;

        // Since challenges are the same for all rows, we only need to evaluate them once
        let challenges: Vec<_> = indexed
            .challenges
            .iter()
            .map(|queried_challenge| {
                EvaluatedError::new_from_boolean_evals(
                    *queried_challenge.value[0],
                    *queried_challenge.value[1],
                    num_evals,
                )
            })
            .collect();

        // Fixed and witness queries are different for each row, so we initialize them to 0
        let mut fixed = vec![F::ZERO; indexed.fixed.len()];
        let mut witness = vec![EvaluatedError::<F>::new(num_evals); indexed.witness.len()];

        let mut sum = EvaluatedError::<F>::new(num_evals);

        for row_index in rows {
            // Fetch fixed data
            for (fixed, query) in fixed.iter_mut().zip(indexed.fixed.iter()) {
                let row_idx = query.row_idx(row_index, num_rows);
                *fixed = query.column[row_idx];
            }

            // Fetch witness data and interpolate
            for (witness, query) in witness.iter_mut().zip(indexed.witness.iter()) {
                let row_idx = query.row_idx(row_index, num_rows);
                let eval0 = query.column[0][row_idx];
                let eval1 = query.column[1][row_idx];
                witness.evaluate(eval0, eval1);
            }

            // Evaluate the expression in the current row and
            for (eval_idx, eval) in sum.evals.iter_mut().enumerate() {
                *eval += indexed.expr.evaluate(
                    &|constant| constant,
                    &|challenge_idx| challenges[challenge_idx].evals[eval_idx],
                    &|fixed_idx| fixed[fixed_idx],
                    &|witness_idx| witness[witness_idx].evals[eval_idx],
                    &|negated| -negated,
                    &|a, b| a + b,
                    &|a, b| a * b,
                );
            }
        }

        sum.to_coefficients()
    }
}

///
#[derive(Clone)]
pub struct EvaluatedError<F> {
    pub evals: Vec<F>,
}

impl<F: Field> AddAssign<&EvaluatedError<F>> for EvaluatedError<F> {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs, rhs) in self.evals.iter_mut().zip(rhs.evals.iter()) {
            *lhs += rhs;
        }
    }
}

impl<F: Field> EvaluatedError<F> {
    pub fn new(num_evals: usize) -> Self {
        Self {
            evals: vec![F::ZERO; num_evals],
        }
    }

    pub fn new_from_boolean_evals(eval0: F, eval1: F, num_evals: usize) -> Self {
        let mut result = Self::new(num_evals);
        result.evaluate(eval0, eval1);
        result
    }

    pub fn evaluate(&mut self, eval0: F, eval1: F) {
        self.evals[0] = eval0;
        let diff = eval1 - eval0;
        let mut prev = eval0;
        for eval in self.evals.iter_mut().skip(1) {
            *eval = prev + diff;
            prev = *eval;
        }
    }

    pub fn to_coefficients(&self) -> Vec<F> {
        let points: Vec<_> = field_integers().take(self.evals.len()).collect();
        lagrange_interpolate(&points, &self.evals)
    }
}

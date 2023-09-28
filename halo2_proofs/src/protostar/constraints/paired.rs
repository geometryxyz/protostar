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
    /// Create a `Data` object from two accumulators, where each variable contains a pair of references to the
    /// some column from two different accumulators. This allows us to create a QueriedExpression where the leaves
    /// contain these two references.
    pub fn new_data<C>(
        pk: &'a ProvingKey<C>,
        acc0: &'a Accumulator<C>,
        acc1: &'a Accumulator<C>,
    ) -> Data<Self>
    where
        C: CurveAffine<ScalarExt = F>,
    {
        let selectors: Vec<_> = pk.selectors.iter().map(|c| &c.values).collect();
        let fixed: Vec<_> = pk.fixed.iter().map(|c| &c.values).collect();

        let instance: Vec<_> = zip(&acc0.gate.instance, &acc1.gate.instance)
            .map(|(i0, i1)| [&i0.values, &i1.values])
            .collect();

        let advice: Vec<_> = zip(&acc0.gate.advice, &acc1.gate.advice)
            .map(|(a0, a1)| [&a0.values, &a1.values])
            .collect();

        let challenges: Vec<_> = zip(&acc0.gate.challenges, &acc1.gate.challenges)
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

    /// Given an expression G where the variables are the linear polynomials interpolating between
    /// the challenges and witness columns from two accumulators,
    /// return the polynomial e(X) = ∑ᵢ βᵢ(X) G(fᵢ, wᵢ(X), rᵢ(X)).
    ///
    /// The strategy for evaluating e(X) is as follows:
    /// - Let D = {0,1,...,d} be the evaluation domain containing the first d + 1 integers, where d is the degree of e(X).
    /// - For each row i, we evaluate the expression eᵢ(X) = βᵢ(X) G(fᵢ, wᵢ(X), rᵢ(X)) over D,
    ///   and add it to the running sum for e(D) = ∑ᵢ eᵢ(D).
    ///   - The input variables βᵢ(X), wᵢ(X), rᵢ(X) are linear polynomials of the form pᵢ(X) = (1−X)⋅pᵢ,₀ + X⋅pᵢ,₁,
    ///     where pᵢ,₀ and pᵢ,₁ are values at the same position but from two different accumulators.
    ///   - For each variable fᵢ, compute fᵢ(D) by setting
    ///     - pᵢ(0) = pᵢ,₀
    ///     - pᵢ(1) = pᵢ,₁
    ///     - pᵢ(j) = pᵢ(j-1) + (pᵢ,₁ − pᵢ,₀) for j = 2, ..., d.
    ///   - Since challenge variables are the same for each row, we compute the evaluations only once.
    /// - Given the Expression for e(X), we evaluate it point-wise as eᵢ(j) = βᵢ(j) G(fᵢ, wᵢ(j), rᵢ(j)) for j in D.
    ///
    /// TODO: As an optimization, we can get away with evaluating the polynomial only at the points 2,...,d,
    /// since e(0) and e(1) are the existing errors from both accumulators. If we let D' = D \ {0,1}, then we can compute
    /// e(D') and reinsert the evaluations at 0 and 1 for the final result before the conversion to coefficients.
    pub fn evaluate_compressed_polynomial(
        expr: QueriedExpression<Self>,
        rows: Range<usize>,
        num_rows: usize,
    ) -> Vec<F> {
        // Convert the expression into an indexed one, where all leaves are extracted into vectors,
        // and replaced by the index in these vectors.
        // This allows us to separate the evaluation of the variables from the evaluation of the expression,
        // since the expression leaves will point to the indices in buffers where the evaluations are stored.
        let indexed = IndexedExpression::<Paired<'a, F>>::new(expr);
        // Evaluate the polynomial at the points 0,1,...,d, where d is the degree of expr,
        // since the polynomial e(X) has d+1 coefficients.
        let num_evals = indexed.expr.degree() + 1;

        // For two transcripts with respective challenge, c₀, c₁,
        // compute the evaluations of the polynomial c(X) = (1−X)⋅c₀ + X⋅c₁
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

        // For each variable of the expression, we allocate buffers for storing their evaluations at each row.
        // - Fixed variables are considered as constants, so we only need to fetch the value from the proving key
        //   and consider fᵢ(j) = fᵢ for all j
        // - Witness variables are interpolated from the values at the two accumulators,
        //   and the evaluations are stored in a buffer.
        let mut fixed = vec![F::ZERO; indexed.fixed.len()];
        let mut witness = vec![EvaluatedError::<F>::new(num_evals); indexed.witness.len()];

        // Running sum for e(D) = ∑ᵢ eᵢ(D)
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

            // Evaluate the expression in the current row and add it to e(D)
            for (eval_idx, eval) in sum.evals.iter_mut().enumerate() {
                // For each `eval_idx` j = 0, 1, ..., d, evaluate the expression eᵢ(j) = βᵢ(j) G(fᵢ, wᵢ(j), rᵢ(j))
                *eval += indexed.expr.evaluate(
                    &|&constant| constant,
                    &|&challenge_idx| challenges[challenge_idx].evals[eval_idx],
                    &|&fixed_idx| fixed[fixed_idx],
                    &|&witness_idx| witness[witness_idx].evals[eval_idx],
                    &|&negated| -negated,
                    &|a, b| a + b,
                    &|a, b| a * b,
                );
            }
        }

        // Convert the evaluations into the coefficients of the polynomial
        sum.to_coefficients()
    }
}

/// Represents a polynomial evaluated over the integers 0,1,...,d.
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
    /// Create a set of `num_evals` evaluations of the zero polynomial
    pub fn new(num_evals: usize) -> Self {
        Self {
            evals: vec![F::ZERO; num_evals],
        }
    }

    /// Returns the evaluations of the linear polynomial (1-X)eval0 + Xeval1.
    pub fn new_from_boolean_evals(eval0: F, eval1: F, num_evals: usize) -> Self {
        let mut result = Self::new(num_evals);
        result.evaluate(eval0, eval1);
        result
    }

    /// Overwrites the current evaluations and replaces it with the evaluations of the linear polynomial (1-X)eval0 + Xeval1.
    pub fn evaluate(&mut self, eval0: F, eval1: F) {
        let mut curr = eval0;
        let diff = eval1 - eval0;
        for eval in self.evals.iter_mut() {
            *eval = curr;
            curr += diff;
        }
    }

    /// Convert the n evalations into the coefficients of the polynomial.
    pub fn to_coefficients(&self) -> Vec<F> {
        let points: Vec<_> = field_integers().take(self.evals.len()).collect();
        lagrange_interpolate(&points, &self.evals)
    }
}

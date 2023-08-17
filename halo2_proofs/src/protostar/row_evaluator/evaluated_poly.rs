use std::{
    iter::zip,
    ops::{
        AddAssign, Deref, DerefMut, Index, IndexMut, MulAssign, RangeFrom, RangeFull, SubAssign,
    },
};

use ff::Field;

/// Represents the evaluations p(D) of a polynomial p(X) over D = {2, 3, , …, n-1}.
#[derive(Debug, Clone)]
pub struct EvaluatedFrom2<F: Field> {
    pub(super) evals: Vec<F>,
}

impl<F: Field> EvaluatedFrom2<F> {
    pub fn new_empty(num_evals: usize) -> Self {
        debug_assert!(2 <= num_evals);
        Self {
            evals: vec![F::ZERO; num_evals - 2],
        }
    }

    pub fn new_from_boolean_evals(eval0: F, eval1: F, num_evals: usize) -> Self {
        let mut result = Self::new_empty(num_evals);
        result.reset_with_boolean_evals(eval0, eval1);
        result
    }

    pub fn reset_with_boolean_evals(&mut self, eval0: F, eval1: F) {
        let diff = eval1 - eval0;
        let mut prev = eval1;
        for eval in self.evals.iter_mut() {
            *eval = prev + diff;
            prev = *eval;
        }
    }

    pub fn set(&mut self, other: &Self) {
        for (lhs, rhs) in zip(self.evals.iter_mut(), other.evals.iter()) {
            *lhs = *rhs;
        }
    }

    pub fn add_prod(&mut self, a: &Self, b: &Self) {
        for (s, (a, b)) in self
            .evals
            .iter_mut()
            .zip(zip(a.evals.iter(), b.evals.iter()))
        {
            *s += *a * b;
        }
    }

    pub fn num_evals(&self) -> usize {
        self.evals.len() + 2
    }

    pub fn to_evaluated(&self, eval0: F, eval1: F) -> Vec<F> {
        let mut evals = self.evals.clone();
        evals.reserve(2);
        evals.insert(0, eval1);
        evals.insert(0, eval0);
        evals
    }
}

impl<F: Field> AddAssign<&EvaluatedFrom2<F>> for EvaluatedFrom2<F> {
    fn add_assign(&mut self, rhs: &Self) {
        for (lhs_eval, rhs_eval) in zip(self.evals.iter_mut(), rhs.evals.iter()) {
            *lhs_eval += rhs_eval;
        }
    }
}

impl<F: Field> SubAssign<&EvaluatedFrom2<F>> for EvaluatedFrom2<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        for (lhs_eval, rhs_eval) in zip(self.evals.iter_mut(), rhs.evals.iter()) {
            *lhs_eval -= rhs_eval;
        }
    }
}

impl<F: Field> MulAssign<&EvaluatedFrom2<F>> for EvaluatedFrom2<F> {
    fn mul_assign(&mut self, rhs: &Self) {
        for (lhs_eval, rhs_eval) in zip(self.evals.iter_mut(), rhs.evals.iter()) {
            *lhs_eval *= rhs_eval;
        }
    }
}

impl<F: Field> AddAssign<&F> for EvaluatedFrom2<F> {
    fn add_assign(&mut self, rhs: &F) {
        for lhs_eval in self.evals.iter_mut() {
            *lhs_eval += rhs;
        }
    }
}

impl<F: Field> MulAssign<&F> for EvaluatedFrom2<F> {
    fn mul_assign(&mut self, rhs: &F) {
        for lhs_eval in self.evals.iter_mut() {
            *lhs_eval *= rhs;
        }
    }
}

impl<F: Field> SubAssign<&F> for EvaluatedFrom2<F> {
    fn sub_assign(&mut self, rhs: &F) {
        for lhs_eval in self.evals.iter_mut() {
            *lhs_eval -= rhs;
        }
    }
}

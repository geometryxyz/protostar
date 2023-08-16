use std::collections::BTreeSet;

use ff::Field;

use crate::{plonk::Expression, poly::Rotation};

use super::queried_expression::QueriedExpression;

// For a given `Expression`, stores all queried variables.
pub struct Queries {
    pub selectors: Vec<usize>,
    pub fixed: Vec<(usize, Rotation)>,
    pub instance: Vec<(usize, Rotation)>,
    pub advice: Vec<(usize, Rotation)>,
    pub challenges: Vec<(usize, usize)>,
}

impl Queries {
    // Computes the lists of queried variables for a given list of `Expression`s.
    pub fn from_polys<F: Field>(polys: &[Expression<F>]) -> Self {
        let mut queried_selectors = BTreeSet::<usize>::new();
        let mut queried_fixed = BTreeSet::<(usize, Rotation)>::new();
        let mut queried_challenges = BTreeSet::<(usize, usize)>::new();
        let mut queried_instance = BTreeSet::<(usize, Rotation)>::new();
        let mut queried_advice = BTreeSet::<(usize, Rotation)>::new();

        // Collect all common queries for the set of polynomials in `gate`
        for poly in polys {
            poly.traverse(&mut |e| match e {
                Expression::Selector(v) => {
                    queried_selectors.insert(v.index());
                }
                Expression::Fixed(v) => {
                    queried_fixed.insert((v.column_index(), v.rotation()));
                }
                Expression::Challenge(v) => {
                    queried_challenges.insert((v.index(), v.power() - 1));
                }
                Expression::Instance(v) => {
                    queried_instance.insert((v.column_index(), v.rotation()));
                }
                Expression::Advice(v) => {
                    queried_advice.insert((v.column_index(), v.rotation()));
                }
                _ => {}
            });
        }
        // Convert the sets of queries into sorted vectors
        Self {
            selectors: queried_selectors.into_iter().collect(),
            fixed: queried_fixed.into_iter().collect(),
            instance: queried_instance.into_iter().collect(),
            advice: queried_advice.into_iter().collect(),
            challenges: queried_challenges.into_iter().collect(),
        }
    }

    /// Given lists of all leaves of the original `Expression`,
    /// create a `QueriedExpression` whose nodes point to indices of variables in `self`.
    pub fn queried_expression<F: Field>(&self, poly: &Expression<F>) -> QueriedExpression<F> {
        fn get_idx<T: PartialEq>(container: &[T], elem: T) -> usize {
            container.iter().position(|x| *x == elem).unwrap()
        }

        poly.evaluate(
            &|v| QueriedExpression::Constant(v),
            &|query| QueriedExpression::Selector(get_idx(&self.selectors, query.index())),
            &|query| {
                QueriedExpression::Fixed(get_idx(
                    &self.fixed,
                    (query.column_index(), query.rotation()),
                ))
            },
            &|query| {
                QueriedExpression::Advice(get_idx(
                    &self.advice,
                    (query.column_index(), query.rotation()),
                ))
            },
            &|query| {
                QueriedExpression::Instance(get_idx(
                    &self.instance,
                    (query.column_index(), query.rotation()),
                ))
            },
            &|query| {
                QueriedExpression::Challenge(get_idx(
                    &self.challenges,
                    (query.index(), query.power() - 1),
                ))
            },
            &|e| QueriedExpression::Negated(e.into()),
            &|e1, e2| QueriedExpression::Sum(e1.into(), e2.into()),
            &|e1, e2| QueriedExpression::Product(e1.into(), e2.into()),
            &|e, v| QueriedExpression::Scaled(e.into(), v),
        )
    }

    // Given a list of challenges with their powers, returns a list of all challenges
    pub fn queried_challenges<F: Field>(&self, challenges: &[Vec<F>]) -> Vec<F> {
        self.challenges
            .iter()
            .map(|(index, row)| challenges[*index][*row])
            .collect()
    }
}

use ff::Field;

use crate::plonk::{AdviceQuery, Challenge, Expression, FixedQuery, InstanceQuery, Selector};

#[derive(Clone)]
struct Slack;

/// Low-degree expression representing an identity that must hold over the committed columns.
#[derive(Clone)]
pub enum HomogenizedExpression<F> {
    Slack(),
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    Selector(Selector),
    /// This is a fixed column queried at a certain relative location
    Fixed(FixedQuery),
    /// This is an advice (witness) column queried at a certain relative location
    Advice(AdviceQuery),
    /// This is an instance (external) column queried at a certain relative location
    Instance(InstanceQuery),
    /// This is a challenge, where the second parameter represents the power of a Challenge
    ChallengePower(Challenge, usize),
    /// This is a negated polynomial
    Negated(Box<HomogenizedExpression<F>>),
    /// This is the sum of two polynomials
    Sum(Box<HomogenizedExpression<F>>, Box<HomogenizedExpression<F>>),
    /// This is the product of two polynomials
    Product(Box<HomogenizedExpression<F>>, Box<HomogenizedExpression<F>>),
    /// This is a scaled polynomial
    Scaled(Box<HomogenizedExpression<F>>, F),
}

impl<F: Field> Expression<F> {
    fn homogenize(self) -> HomogenizedExpression<F> {
        match self {
            Expression::Constant(v) => HomogenizedExpression::Constant(v),
            Expression::Selector(v) => HomogenizedExpression::Selector(v),
            Expression::Fixed(v) => HomogenizedExpression::Fixed(v),
            Expression::Advice(v) => HomogenizedExpression::Advice(v),
            Expression::Instance(v) => HomogenizedExpression::Instance(v),
            Expression::Challenge(c) => HomogenizedExpression::ChallengePower(v, 1),
            Expression::Negated(e) => HomogenizedExpression::Negated(Box::new(e.homogenize())),
            Expression::Sum(e1, e2) => {
                HomogenizedExpression::Sum(Box::new(e1.homogenize()), Box::new(e2.homogenize()))
            }
            Expression::Product(e1, e2) => {
                HomogenizedExpression::Product(Box::new(e1.homogenize()), Box::new(e2.homogenize()))
            }
            Expression::Scaled(e, v) => HomogenizedExpression::Scaled(Box::new(e.homogenize()), v),
        }
    }
}

impl<F: Field> HomogenizedExpression<F> {
    fn distribute_challenge_powers(self, challenge: &Challenge, power: usize) -> Self {
        match self {
            HomogenizedExpression::ChallengePower(other_challenge, d) => {
                if other_challenge == *challenge {
                    if d == 1 {
                        panic!("Challenge power cannot be other than 1")
                    }
                    HomogenizedExpression::ChallengePower(*challenge, 1 + power)
                } else {
                }
            }
            HomogenizedExpression::Negated(e) => HomogenizedExpression::Negated(Box::new(
                e.distribute_challenge_powers(challenge, power),
            )),
            HomogenizedExpression::Sum(e1, e2) => HomogenizedExpression::Sum(
                Box::new(e1.distribute_challenge_powers(challenge, power)),
                Box::new(e2.distribute_challenge_powers(challenge, power)),
            ),
            HomogenizedExpression::Product(e1, e2) => {
                if power == 0 {
                    HomogenizedExpression::Product(
                        Box::new(e1.distribute_challenge_powers(challenge, power)),
                        Box::new(e2.distribute_challenge_powers(challenge, power)),
                    )
                } else {
                    HomogenizedExpression::Product(
                        Box::new(e1.distribute_challenge_powers(challenge, power)),
                        Box::new(e2.distribute_challenge_powers(challenge, power)),
                    )
                }
            }
            HomogenizedExpression::Scaled(v, f) => HomogenizedExpression::Scaled(
                Box::new(v.distribute_challenge_powers(challenge, power)),
                f,
            ),
            v => {
                if power == 0 {
                    v
                } else {
                    HomogenizedExpression::Product(
                        Box::new(v),
                        Box::new(HomogenizedExpression::ChallengePower(*challenge, power)),
                    )
                }
            }
        }
    }

    fn flatten_challenge_powers(self, challenges: &[Challenge]) -> Self {
        let mut expr = self;
        for challenge in challenges {
            (expr, _) = expr.flatten_challenge_powers_inner(challenge);
        }
        expr
    }

    fn homogenize(self) -> Self {
        HomogenizedExpression::Slack()
    }
}

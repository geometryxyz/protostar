use std::collections::BTreeSet;

use ff::Field;

use crate::{
    plonk::{AdviceQuery, Challenge, Expression, FirstPhase, FixedQuery, InstanceQuery, Selector},
    poly::Rotation,
};

use super::keygen::CircuitData;

/// A Protostar gate is augments the structure of a plonk::Gate to allow for more efficient evaluation
#[derive(Clone)]
pub struct Gate<F: Field> {
    // polynomial  expressions where nodes point to indices of elements of `queried_*` vectors
    // top-level selector has been extracted into `simple_selectors` if the gate was obtained via `WithSelector`
    pub(super) polys: Vec<Expr<F>>,
    // simple selector for toggling all polys
    pub(super) simple_selector: Option<Selector>,
    // homogeneous degree of each poly
    pub(super) degrees: Vec<usize>,

    // queries for this expression inside the full table.
    pub(super) queried_selectors: Vec<Selector>,
    pub(super) queried_fixed: Vec<FixedQuery>,
    pub(super) queried_challenges: Vec<Challenge>,
    pub(super) queried_instance: Vec<InstanceQuery>,
    pub(super) queried_advice: Vec<AdviceQuery>,
}

fn extract_simple_selector<F: Field>(
    original_polys: &[Expression<F>],
) -> (Vec<Expression<F>>, Option<Selector>) {
    let (mut polys, simple_selectors): (Vec<_>, Vec<_>) = original_polys
        .iter()
        .map(|poly| {
            let (simple_selector, poly) = match poly {
                // If the whole polynomial is multiplied by a simple selector, return it along with the expression it selects
                Expression::Product(e1, e2) => match (&**e1, &**e2) {
                    (Expression::Selector(s), e) | (e, Expression::Selector(s)) => (Some(*s), e),
                    _ => (None, poly),
                },
                _ => (None, poly),
            };
            (poly.clone(), simple_selector)
        })
        .unzip();

    // Check if all simple selectors are the same and if so select it
    let potential_selector = match simple_selectors.as_slice() {
        [head, tail @ ..] => {
            if let Some(s) = *head {
                tail.iter().all(|x| x.is_some_and(|x| s == x)).then(|| s)
            } else {
                None
            }
        }
        [] => None,
    };

    // if we haven't found a common simple selector, then we just use the previous polys
    if potential_selector.is_none() {
        polys.clear();
        polys.extend_from_slice(original_polys);
    }

    (polys, potential_selector)
}

impl<F: Field> From<&crate::plonk::Gate<F>> for Gate<F> {
    fn from(cs_gate: &crate::plonk::Gate<F>) -> Gate<F> {
        let mut selectors = BTreeSet::<Selector>::new();
        let mut fixed = BTreeSet::<FixedQuery>::new();
        let mut challenges = BTreeSet::<Challenge>::new();
        let mut instance = BTreeSet::<InstanceQuery>::new();
        let mut advice = BTreeSet::<AdviceQuery>::new();

        // Recover common simple `Selector` from the `Gate`, along with the branch `Expression`s  it selects
        let (polys, simple_selector) = extract_simple_selector(cs_gate.polynomials());

        // Collect all common queries for the set of polynomials in `gate`
        for poly in polys.iter() {
            poly.traverse(&mut |e| match e {
                Expression::Selector(v) => {
                    selectors.insert(*v);
                }
                Expression::Fixed(v) => {
                    fixed.insert(*v);
                }
                Expression::Challenge(v) => {
                    challenges.insert(*v);
                }
                Expression::Instance(v) => {
                    instance.insert(*v);
                }
                Expression::Advice(v) => {
                    advice.insert(*v);
                }
                _ => {}
            });
        }

        // Convert the sets of queries into sorted vectors
        let selectors: Vec<_> = selectors.into_iter().collect();
        let fixed: Vec<_> = fixed.into_iter().collect();
        let challenges: Vec<_> = challenges.into_iter().collect();
        let instance: Vec<_> = instance.into_iter().collect();
        let advice: Vec<_> = advice.into_iter().collect();

        let num_challenges = challenges.len();

        // get homogenized and degree-flattened expressions
        let (polys, degrees): (Vec<_>, Vec<_>) = polys
            .iter()
            // convert Expression to Expr
            .map(|e| e.to_expr(&selectors, &fixed, &challenges, &instance, &advice))
            // merge products of challenges into challenge powers
            .map(|e| {
                (0..num_challenges)
                    .into_iter()
                    .fold(e, |acc, c| acc.distribute_challenge(c, 0))
            })
            // homogenize the expression by introducing slack variable, returning the degree too
            .map(|e| e.homogenize())
            .unzip();
        // let max_degree = *degrees.iter().max().unwrap();

        Gate {
            polys,
            simple_selector,
            degrees,
            queried_selectors: selectors,
            queried_fixed: fixed,
            queried_challenges: challenges,
            queried_instance: instance,
            queried_advice: advice,
        }
    }
}

/// Low-degree expression representing an identity that must hold over the committed columns.
#[derive(Debug, Clone)]
pub enum Expr<F> {
    /// This is a slack variable whose purpose is to make the equation homogeneous.
    Slack(usize),
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
    // TODO(@adr1anh): replace with Selector(Box<Expr<F>>, Selector),
    Selector(usize),
    /// This is a fixed column queried at a certain relative location
    Fixed(usize),
    /// This is an advice (witness) column queried at a certain relative location
    Advice(usize),
    /// This is an instance (external) column queried at a certain relative location
    Instance(usize),
    /// This is a challenge, where the second parameter represents the power of a Challenge.
    /// Two challenges with different powers are treated as separate variables.
    Challenge(usize, usize),
    /// This is a negated polynomial
    Negated(Box<Expr<F>>),
    /// This is the sum of two polynomials
    Sum(Box<Expr<F>>, Box<Expr<F>>),
    /// This is the product of two polynomials
    Product(Box<Expr<F>>, Box<Expr<F>>),
    /// This is a scaled polynomial
    Scaled(Box<Expr<F>>, F),
}

impl<F: Field> Expression<F> {
    pub fn to_expr(
        &self,
        selectors: &Vec<Selector>,
        fixed: &Vec<FixedQuery>,
        challenges: &Vec<Challenge>,
        instance: &Vec<InstanceQuery>,
        advice: &Vec<AdviceQuery>,
    ) -> Expr<F> {
        fn get_idx<T: PartialEq>(container: &[T], elem: &T) -> usize {
            container.iter().position(|x| x == elem).unwrap()
        }

        let recurse = |e: &Expression<F>| -> Box<Expr<F>> {
            Box::new(e.to_expr(selectors, fixed, challenges, instance, advice))
        };

        match self {
            Expression::Constant(v) => Expr::Constant(*v),
            Expression::Selector(v) => Expr::Selector(get_idx(selectors, v)),
            Expression::Fixed(v) => Expr::Fixed(get_idx(fixed, v)),
            Expression::Advice(v) => Expr::Advice(get_idx(advice, v)),
            Expression::Instance(v) => Expr::Instance(get_idx(instance, v)),
            Expression::Challenge(v) => Expr::Challenge(get_idx(challenges, v), 1),
            Expression::Negated(e) => Expr::Negated(recurse(e)),
            Expression::Sum(e1, e2) => Expr::Sum(recurse(e1), recurse(e2)),
            Expression::Product(e1, e2) => Expr::Product(recurse(e1), recurse(e2)),
            Expression::Scaled(e, v) => Expr::Scaled(recurse(e), *v),
        }
    }
}

impl<F: Field> Expr<F> {
    fn flatten_challenges(self, challenges: &[usize]) -> Self {
        // for each challenge, flatten the tree and turn products of the challenge
        // with powers of the challenge
        challenges
            .iter()
            .fold(self, |acc, c| acc.distribute_challenge(*c, 0))
    }

    /// Multiply self by challenge^power, merge multiplications of challenges into powers of it
    pub fn distribute_challenge(self, challenge: usize, power: usize) -> Expr<F> {
        match self {
            Expr::Negated(e) => Expr::Negated(e.distribute_challenge(challenge, power).into()),
            Expr::Scaled(v, f) => Expr::Scaled(v.distribute_challenge(challenge, power).into(), f),
            Expr::Sum(e1, e2) => Expr::Sum(
                e1.distribute_challenge(challenge, power).into(),
                e2.distribute_challenge(challenge, power).into(),
            ),
            Expr::Product(e1, e2) => {
                match (*e1, *e2) {
                    (Expr::Challenge(c, d), e) | (e, Expr::Challenge(c, d)) => {
                        if c == challenge {
                            // If either branch is the challenge with power d,
                            // replace current node with the other branch and
                            // multiply that branch by (c,d+power)
                            e.distribute_challenge(challenge, power + d)
                        } else {
                            // Recreate the node, storing challenge on the left,
                            // and recurse on the other
                            Expr::Product(
                                Expr::Challenge(c, d).into(),
                                e.distribute_challenge(challenge, power).into(),
                            )
                        }
                    }
                    // Neither banches are challenges -> recurse
                    (e1, e2) => Expr::Product(
                        e1.distribute_challenge(challenge, power).into(),
                        e2.distribute_challenge(challenge, power).into(),
                    ),
                }
            }
            // Update the current challenge to the power
            Expr::Challenge(c, d) => {
                if c == challenge {
                    // if same challenge, then existing power must be 1
                    debug_assert!(d == 1, "degree cannot be different than 1");
                    Expr::Challenge(c, 1 + power)
                } else {
                    // other challenge stays the same
                    Expr::Challenge(c, d)
                }
            }
            // handles leaves
            // replaces v with (challenge^power * v)
            v => {
                if power == 0 {
                    v
                } else {
                    Expr::Product(Expr::Challenge(challenge, power).into(), v.into())
                }
            }
        }
    }

    // Compute the degree where powers of challenges count as 1
    fn degree(&self) -> usize {
        match self {
            Expr::Slack(d) => *d,
            Expr::Constant(_) => 0,
            Expr::Selector(_) => 0,
            Expr::Fixed(_) => 0,
            Expr::Advice(_) => 1,
            Expr::Instance(_) => 1,
            Expr::Challenge(_, _) => 1,
            Expr::Negated(e) => e.degree(),
            Expr::Sum(e1, e2) => std::cmp::max(e1.degree(), e2.degree()),
            Expr::Product(e1, e2) => e1.degree() + e2.degree(),
            Expr::Scaled(e, _) => e.degree(),
        }
    }

    // Homogenizes self using powers of Expr::Slack, also returning the new degree.
    // Assumes that the expression has not been homogenized yet
    pub fn homogenize(self) -> (Expr<F>, usize) {
        match self {
            Expr::Slack(_) => panic!("Should not contain existing slack variable"),
            Expr::Negated(e) => {
                let (e, d) = e.homogenize();
                (Expr::Negated(e.into()), d)
            }
            Expr::Sum(e1, e2) => {
                let (mut e1, d1) = e1.homogenize();
                let (mut e2, d2) = e2.homogenize();
                let d = std::cmp::max(d1, d2);

                e1 = if d1 < d {
                    Expr::Product(Expr::Slack(d - d1).into(), e1.into())
                } else {
                    e1
                };
                e2 = if d2 < d {
                    Expr::Product(Expr::Slack(d - d2).into(), e2.into())
                } else {
                    e2
                };
                (Expr::Sum(e1.into(), e2.into()), d)
            }
            Expr::Product(e1, e2) => {
                let (e1, d1) = e1.homogenize();
                let (e2, d2) = e2.homogenize();
                // otherwise increase the degree of e_prod to degree
                (Expr::Product(e1.into(), e2.into()), d1 + d2)
            }
            Expr::Scaled(e, v) => {
                let (e, d) = e.homogenize();
                (Expr::Scaled(e.into(), v), d)
            }
            v => {
                let d = v.degree();
                (v, d)
            }
        }
    }

    /// If self is homogeneous, return the degree, else None
    fn homogeneous_degree(&self) -> Option<usize> {
        match self {
            Expr::Negated(e) => e.homogeneous_degree(),
            Expr::Sum(e1, e2) => {
                // Get homogenous degree of children,
                // and return their degree if they are the same
                if let (Some(d1), Some(d2)) = (e1.homogeneous_degree(), e2.homogeneous_degree()) {
                    if d1 == d2 {
                        return Some(d1);
                    }
                }
                None
            }
            Expr::Product(e1, e2) => {
                // Get homogenous degree of children,
                // and return their sum
                if let (Some(d1), Some(d2)) = (e1.homogeneous_degree(), e2.homogeneous_degree()) {
                    return Some(d1 + d2);
                }
                None
            }
            Expr::Scaled(e, _) => e.homogeneous_degree(),
            v => Some(v.degree()),
        }
    }

    /// Evaluate the expression using closures for each node types.
    pub fn evaluate<T>(
        &self,
        slack: &impl Fn(usize) -> T,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(usize) -> T,
        fixed_column: &impl Fn(usize) -> T,
        advice_column: &impl Fn(usize) -> T,
        instance_column: &impl Fn(usize) -> T,
        challenge: &impl Fn(usize, usize) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            Expr::Slack(d) => slack(*d),
            Expr::Constant(scalar) => constant(*scalar),
            Expr::Selector(selector) => selector_column(*selector),
            Expr::Fixed(query) => fixed_column(*query),
            Expr::Advice(query) => advice_column(*query),
            Expr::Instance(query) => instance_column(*query),
            Expr::Challenge(value, power) => challenge(*value, *power),
            Expr::Negated(a) => {
                let a = a.evaluate(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                negated(a)
            }
            Expr::Sum(a, b) => {
                let a = a.evaluate(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                sum(a, b)
            }
            Expr::Product(a, b) => {
                let a = a.evaluate(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                let b = b.evaluate(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                product(a, b)
            }
            Expr::Scaled(a, f) => {
                let a = a.evaluate(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                );
                scaled(a, *f)
            }
        }
    }

    /// Approximate the computational complexity of this expression.
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Slack(_) => 1,
            Expr::Constant(_) => 0,
            // Selectors should always be evaluated first
            Expr::Selector(_) => 0,
            Expr::Fixed(_) => 1,
            Expr::Advice(_) => 1,
            Expr::Instance(_) => 1,
            Expr::Challenge(_, _) => 1,
            Expr::Negated(poly) => poly.complexity() + 5,
            Expr::Sum(a, b) => a.complexity() + b.complexity() + 15,
            Expr::Product(a, b) => a.complexity() + b.complexity() + 30,
            Expr::Scaled(poly, _) => poly.complexity() + 30,
        }
    }

    /// Evaluate the polynomial lazily using the provided closures to perform the
    /// operations.
    pub fn evaluate_lazy<T: PartialEq>(
        &self,
        slack: &impl Fn(usize) -> T,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(usize) -> T,
        fixed_column: &impl Fn(usize) -> T,
        advice_column: &impl Fn(usize) -> T,
        instance_column: &impl Fn(usize) -> T,
        challenge: &impl Fn(usize, usize) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
        zero: &T,
    ) -> T {
        match self {
            Expr::Slack(power) => slack(*power),
            Expr::Constant(scalar) => constant(*scalar),
            Expr::Selector(selector) => selector_column(*selector),
            Expr::Fixed(query) => fixed_column(*query),
            Expr::Advice(query) => advice_column(*query),
            Expr::Instance(query) => instance_column(*query),
            Expr::Challenge(value, power) => challenge(*value, *power),
            Expr::Negated(a) => {
                let a = a.evaluate_lazy(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                negated(a)
            }
            Expr::Sum(a, b) => {
                let a = a.evaluate_lazy(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                let b = b.evaluate_lazy(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                sum(a, b)
            }
            Expr::Product(a, b) => {
                let (a, b) = if a.complexity() <= b.complexity() {
                    (a, b)
                } else {
                    (b, a)
                };
                let a = a.evaluate_lazy(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );

                if a == *zero {
                    a
                } else {
                    let b = b.evaluate_lazy(
                        slack,
                        constant,
                        selector_column,
                        fixed_column,
                        advice_column,
                        instance_column,
                        challenge,
                        negated,
                        sum,
                        product,
                        scaled,
                        zero,
                    );
                    product(a, b)
                }
            }
            Expr::Scaled(a, f) => {
                let a = a.evaluate_lazy(
                    slack,
                    constant,
                    selector_column,
                    fixed_column,
                    advice_column,
                    instance_column,
                    challenge,
                    negated,
                    sum,
                    product,
                    scaled,
                    zero,
                );
                scaled(a, *f)
            }
        }
    }

    fn traverse(&self, f: &mut impl FnMut(&Expr<F>)) {
        match self {
            Expr::Negated(e) => e.traverse(f),
            Expr::Sum(e1, e2) => {
                e1.traverse(f);
                e2.traverse(f);
            }
            Expr::Product(e1, e2) => {
                e1.traverse(f);
                e2.traverse(f);
            }
            Expr::Scaled(e, _) => e.traverse(f),
            v => f(v),
        }
    }
}

// impl<F: std::fmt::Debug + Field> std::fmt::Debug for Expr<F> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Expr::Slack(d) => f.debug_tuple("Slack").field(d).finish(),
//             Expr::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
//             Expr::Selector(selector) => f.debug_tuple("Selector").field(selector).finish(),
//             // Skip enum variant and print query struct directly to maintain backwards compatibility.
//             Expr::Fixed(query) => {
//                 let mut debug_struct = f.debug_struct("Fixed");
//                 match query.index {
//                     None => debug_struct.field("query_index", &query.index),
//                     Some(idx) => debug_struct.field("query_index", &idx),
//                 };
//                 debug_struct
//                     .field("column_index", &query.column_index)
//                     .field("rotation", &query.rotation)
//                     .finish()
//             }
//             Expr::Advice(query) => {
//                 let mut debug_struct = f.debug_struct("Advice");
//                 match query.index {
//                     None => debug_struct.field("query_index", &query.index),
//                     Some(idx) => debug_struct.field("query_index", &idx),
//                 };
//                 debug_struct
//                     .field("column_index", &query.column_index)
//                     .field("rotation", &query.rotation);
//                 // Only show advice's phase if it's not in first phase.
//                 // if query.phase != FirstPhase.to_sealed() {
//                 //     debug_struct.field("phase", &query.phase);
//                 // }
//                 debug_struct.finish()
//             }
//             Expr::Instance(query) => {
//                 let mut debug_struct = f.debug_struct("Instance");
//                 match query.index {
//                     None => debug_struct.field("query_index", &query.index),
//                     Some(idx) => debug_struct.field("query_index", &idx),
//                 };
//                 debug_struct
//                     .field("column_index", &query.column_index)
//                     .field("rotation", &query.rotation)
//                     .finish()
//             }
//             Expr::Challenge(_, power) => f.debug_tuple("Challenge").field(power).finish(),
//             Expr::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
//             Expr::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
//             Expr::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
//             Expr::Scaled(poly, scalar) => {
//                 f.debug_tuple("Scaled").field(poly).field(scalar).finish()
//             }
//         }
//     }
// }

// #[cfg(test)]

// mod tests {
//     use core::num;

//     use crate::plonk::{sealed::Phase, ConstraintSystem, FirstPhase};
//     use crate::{halo2curves::pasta::Fp, plonk::sealed::SealedPhase};

//     use super::*;
//     use crate::plonk::sealed;
//     use crate::plonk::Expression;
//     use rand_core::{OsRng, RngCore};

//     #[test]
//     fn test_expression_conversion() {
//         let rng = OsRng;

//         let num_challenges = 4;
//         let degree = 5;

//         let challenges: Vec<_> = {
//             let mut cs = ConstraintSystem::<Fp>::default();
//             let _ = cs.advice_column();
//             let _ = cs.advice_column_in(FirstPhase);
//             (0..num_challenges)
//                 .map(|_| cs.challenge_usable_after(FirstPhase))
//                 .collect()
//         };
//         let x = challenges[0];
//         let x_var: Expression<Fp> = Expression::Challenge(x);

//         let coefficients: Vec<_> = (0..degree - 1)
//             .map(|_| Expression::Constant(Fp::random(rng)))
//             .collect();

//         // Expression for the polynomial computed via Horner's method
//         // F(X) = a0 + X(a1 + X(a2 + X(...(a{n-2} + Xa{n-1})))))
//         let horner_poly = coefficients
//             .iter()
//             .rev()
//             .skip(1)
//             .fold(coefficients.last().unwrap().clone(), |acc, a_i| {
//                 x_var.clone() * acc + a_i.clone()
//             });

//         // Distribute the challenge with power 0
//         let flattened_poly = Expr::from(horner_poly).distribute_challenge(&x, 0);
//         let d_flattened = flattened_poly.degree();
//         assert_eq!(d_flattened, 1);

//         let (homogenous_poly, d_homogeneous) = flattened_poly.homogenize();
//         assert_eq!(d_homogeneous, 1);

//         let opt_d = homogenous_poly.homogeneous_degree();
//         // Check if homogeneous
//         assert!(opt_d.is_some());
//         assert!(opt_d.unwrap() == d_homogeneous);
//     }
// }

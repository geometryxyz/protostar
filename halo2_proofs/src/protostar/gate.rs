use std::collections::BTreeSet;

use ff::Field;

use crate::{
    plonk::{AdviceQuery, Challenge, Expression, FirstPhase, FixedQuery, InstanceQuery, Selector},
    poly::Rotation,
};

/// A Protostar `Gate` augments the structure of a `plonk::Gate` to allow for more efficient evaluation.
/// Stores the different polynomial expressions Gⱼ for the gate.
/// Each Gⱼ is represented as tree where nodes point to indices of elements of the `queried_*` vectors.
/// If all original polynomial expressions were multiplied at the top-level by a common simple `Selector`,
/// this latter leaf is extracted from each Gⱼ and applied only once to all sub-polynomials.
/// In general, this undoes the transformation done by `Constraints::with_selector`.
#[derive(Debug, Clone)]
pub struct Gate<F: Field> {
    // List of polynomial expressions Gⱼ
    pub polys: Vec<Expr<F>>,
    // Simple `Selector` which multiplies all Gⱼ
    pub simple_selector: Option<Selector>,
    // Degrees of each Gⱼ
    pub degrees: Vec<usize>,

    // List of all columns queried by the polynomial expressions Gⱼ
    pub queried_selectors: Vec<Selector>,
    pub queried_fixed: Vec<FixedQuery>,
    pub queried_challenges: Vec<Challenge>,
    pub queried_instance: Vec<InstanceQuery>,
    pub queried_advice: Vec<AdviceQuery>,
}

impl<F: Field> From<&crate::plonk::Gate<F>> for Gate<F> {
    /// Create an augmented Protostar gate from a `plonk::Gate`.
    /// - Extract the common top-level `Selector` if it exists
    /// - Extract all queries, and replace leaves with indices to the queries stored in the gate
    /// - Flatten challenges so that a product of the same challenge is replaced by a power of that challenge
    fn from(cs_gate: &crate::plonk::Gate<F>) -> Gate<F> {
        let mut selectors = BTreeSet::<Selector>::new();
        let mut fixed = BTreeSet::<FixedQuery>::new();
        let mut challenges = BTreeSet::<Challenge>::new();
        let mut instance = BTreeSet::<InstanceQuery>::new();
        let mut advice = BTreeSet::<AdviceQuery>::new();

        // Recover common simple `Selector` from the `Gate`, along with the branch `Expression`s  it selects
        let (polys, simple_selector) = cs_gate.extract_simple_selector();

        // Merge products of challenges
        let polys: Vec<_> = polys
            .into_iter()
            .map(|poly| poly.merge_challenge_products())
            .collect();

        let degrees: Vec<_> = polys.iter().map(|poly| poly.folding_degree()).collect();

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

        // get homogenized and degree-flattened expressions
        let polys: Vec<_> = polys
            .iter()
            // convert Expression to Expr, replacing each query node by its index in the given vectors
            .map(|e| e.to_expr(&selectors, &fixed, &challenges, &instance, &advice))
            .collect();

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
#[derive(Clone)]
pub enum Expr<F> {
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
    /// This is a challenge
    Challenge(usize),
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
    /// Given lists of all leaves of the original `Expression`, create an `Expr` where the leaves
    /// correspond to indices of the variable in the lists.
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
            Expression::Challenge(v) => Expr::Challenge(get_idx(challenges, v)),
            Expression::Negated(e) => Expr::Negated(recurse(e)),
            Expression::Sum(e1, e2) => Expr::Sum(recurse(e1), recurse(e2)),
            Expression::Product(e1, e2) => Expr::Product(recurse(e1), recurse(e2)),
            Expression::Scaled(e, v) => Expr::Scaled(recurse(e), *v),
        }
    }
}

impl<F: Field> Expr<F> {
    /// Evaluate the expression using closures for each node types.
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        selector_column: &impl Fn(usize) -> T,
        fixed_column: &impl Fn(usize) -> T,
        advice_column: &impl Fn(usize) -> T,
        instance_column: &impl Fn(usize) -> T,
        challenge: &impl Fn(usize) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        match self {
            Expr::Constant(scalar) => constant(*scalar),
            Expr::Selector(selector) => selector_column(*selector),
            Expr::Fixed(query) => fixed_column(*query),
            Expr::Advice(query) => advice_column(*query),
            Expr::Instance(query) => instance_column(*query),
            Expr::Challenge(value) => challenge(*value),
            Expr::Negated(a) => {
                let a = a.evaluate(
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
}

impl<F: std::fmt::Debug + Field> std::fmt::Debug for Expr<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            Expr::Selector(selector) => f.debug_tuple("Selector").field(selector).finish(),
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            Expr::Fixed(query) => f.debug_tuple("Fixed").field(query).finish(),
            Expr::Advice(query) => f.debug_tuple("Advice").field(query).finish(),
            Expr::Instance(query) => f.debug_tuple("Instance").field(query).finish(),
            Expr::Challenge(c) => f.debug_tuple("Challenge").field(c).finish(),
            Expr::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            Expr::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            Expr::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            Expr::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

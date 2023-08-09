use std::{collections::BTreeSet, iter::zip};

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    plonk::{AdviceQuery, Challenge, Expression, FixedQuery, Gate, InstanceQuery, Selector},
    poly::Rotation,
    protostar::keygen::ProvingKey,
};

use super::{boolean_evaluations_vec, Accumulator};

/// When evaluating a `Gate` over all rows, we fetch the input data from the previous accumulator
/// and the new witness and store it in this struct.
/// This struct can be cloned when paralellizing the evaluation over chunks of rows,
/// so that each chunk operates over independent data.
/// // /// A Protostar `Gate` augments the structure of a `plonk::Gate` to allow for more efficient evaluation.
/// Stores the different polynomial expressions Gⱼ for the gate.
/// Each Gⱼ is represented as tree where nodes point to indices of elements of the `queried_*` vectors.
/// If all original polynomial expressions were multiplied at the top-level by a common simple `Selector`,
/// this latter leaf is extracted from each Gⱼ and applied only once to all sub-polynomials.
/// In general, this undoes the transformation done by `Constraints::with_selector`.
/// Create an augmented Protostar gate from a `plonk::Gate`.
/// - Extract the common top-level `Selector` if it exists
/// - Extract all queries, and replace leaves with indices to the queries stored in the gate
/// - Flatten challenges so that a product of the same challenge is replaced by a power of that challenge
/// TODO(@adr1anh): Cleanup this comment mess

pub struct GateEvaluator<F: Field> {
    // Number of evaluations requried to recover each `poly` in `gate`.
    // Equal to `poly.degree() + 1 + num_extra_evaluations`,
    // which is the number of coefficients of the polynomial,
    // plus any additional evaluations required if we want to multiply by another variable
    // later on.
    num_evals: Vec<usize>,
    // Maximum of `num_evals`
    max_num_evals: usize,

    // List of polynomial expressions Gⱼ
    polys: Vec<Expr<F>>,

    // List of all columns queried by the polynomial expressions Gⱼ
    // Simple `Selector` which multiplies all Gⱼ
    queried_simple_selector: Option<Selector>,
    queried_selectors: Vec<Selector>,
    queried_fixed: Vec<FixedQuery>,
    queried_instance: Vec<InstanceQuery>,
    queried_advice: Vec<AdviceQuery>,

    // Evaluations of the challenges queried by the gate
    challenge_evals: Vec<Vec<F>>,

    // Cache for storing the fixed, accumulated, and witness values for evaluating this
    // `gate` at a single row.
    //
    // If all `polys` are multiplied by the same `Selector`, store its value here
    // and allow the evaluation of the gate to be skipped if it is `false`
    simple_selector: Option<bool>,
    // Contains the remaining selectors inside `polys` defined by `gate.queried_selectors`
    selectors: Vec<F>,
    // Values for the row from `circuit_data` at the indices defined by `gate.queried_fixed`
    fixed: Vec<F>,

    // Values for the row from `acc` at the indices defined by `gate.queried_instance`
    instance_acc: Vec<F>,
    // Values of the difference between the rows `new` and `acc` at the indices defined by `gate.queried_instance`
    instance_diff: Vec<F>,
    // Values for the row from `acc` at the indices defined by `gate.queried_advice`
    advice_acc: Vec<F>,
    // Values of the difference between the rows `new` and `acc` at the indices defined by `gate.queried_advice`
    advice_diff: Vec<F>,

    // For each `poly` at index `j` in `gate`,
    // store the evaluations of the error polynomial `e_j(X)`
    // at X = 0, 1, ..., num_evals[j] - 1 in `gate_eval[j][X]`
    pub gate_eval: Vec<Vec<F>>,
}

impl<F: Field> GateEvaluator<F> {
    /// Initialize a chache for a given `Gate` for combining an existing accumulator with
    /// the data from a new transcript.
    /// Evaluations of the linear combination of the slack and challenge variables
    /// are recomputed and stored locally to ensure no data races between different caches.
    /// The `evaluate` method can compute additional evaluations defined by `num_extra_evaluations`,
    /// so that the resulting evaluations can be multiplied afterwards, for example by the challenge `y`.
    pub fn new(
        gate: &Gate<F>,
        acc_challenges: &[Vec<F>],
        new_challenges: &[Vec<F>],
        num_extra_evaluations: usize,
    ) -> Self {
        // Recover common simple `Selector` from the `Gate`, along with the branch `Expression`s  it selects
        let (polys, queried_simple_selector) = gate.extract_simple_selector();

        // Merge products of challenges
        let polys: Vec<_> = polys
            .into_iter()
            .map(|poly| poly.merge_challenge_products())
            .collect();

        let degrees: Vec<_> = polys.iter().map(|poly| poly.folding_degree()).collect();

        let mut queried_selectors = BTreeSet::<Selector>::new();
        let mut queried_fixed = BTreeSet::<FixedQuery>::new();
        let mut queried_challenges = BTreeSet::<Challenge>::new();
        let mut queried_instance = BTreeSet::<InstanceQuery>::new();
        let mut queried_advice = BTreeSet::<AdviceQuery>::new();

        // Collect all common queries for the set of polynomials in `gate`
        for poly in polys.iter() {
            poly.traverse(&mut |e| match e {
                Expression::Selector(v) => {
                    queried_selectors.insert(*v);
                }
                Expression::Fixed(v) => {
                    queried_fixed.insert(*v);
                }
                Expression::Challenge(v) => {
                    queried_challenges.insert(*v);
                }
                Expression::Instance(v) => {
                    queried_instance.insert(*v);
                }
                Expression::Advice(v) => {
                    queried_advice.insert(*v);
                }
                _ => {}
            });
        }
        // Convert the sets of queries into sorted vectors
        let queried_selectors: Vec<_> = queried_selectors.into_iter().collect();
        let queried_fixed: Vec<_> = queried_fixed.into_iter().collect();
        let queried_challenges: Vec<_> = queried_challenges.into_iter().collect();
        let queried_instance: Vec<_> = queried_instance.into_iter().collect();
        let queried_advice: Vec<_> = queried_advice.into_iter().collect();
        // allocate buffers for storing gate valuess
        let selectors = vec![F::ZERO; queried_selectors.len()];
        let fixed = vec![F::ZERO; queried_fixed.len()];
        let instance_acc = vec![F::ZERO; queried_instance.len()];
        let instance_diff = vec![F::ZERO; queried_instance.len()];
        let advice_acc = vec![F::ZERO; queried_advice.len()];
        let advice_diff = vec![F::ZERO; queried_advice.len()];

        // get homogenized and degree-flattened expressions
        let polys: Vec<_> = polys
            .iter()
            // convert Expression to Expr, replacing each query node by its index in the given vectors
            .map(|e| {
                e.to_expr(
                    &queried_selectors,
                    &queried_fixed,
                    &queried_challenges,
                    &queried_instance,
                    &queried_advice,
                )
            })
            .collect();

        // Each `poly` Gⱼ(X) has degree dⱼ and dⱼ+1 coefficients,
        // therefore we need at least dⱼ+1 evaluations of Gⱼ(X) to recover
        // the coefficients.
        let num_evals: Vec<_> = degrees
            .iter()
            .map(|d| d + 1 + num_extra_evaluations)
            .collect();
        // maximum number of evaluations over all Gⱼ(X)
        let max_num_evals = *num_evals.iter().max().unwrap();

        let acc_queried_challenges: Vec<_> = queried_challenges
            .iter()
            .map(|c| acc_challenges[c.index()][c.power() - 1])
            .collect();
        let new_queried_challenges: Vec<_> = queried_challenges
            .iter()
            .map(|c| new_challenges[c.index()][c.power() - 1])
            .collect();

        let challenge_evals: Vec<_> =
            boolean_evaluations_vec(&acc_queried_challenges, &new_queried_challenges)
                .take(max_num_evals)
                .collect();

        // for each polynomial, allocate a buffer for storing all the evaluations
        let gate_eval = num_evals.iter().map(|d| vec![F::ZERO; *d]).collect();

        Self {
            num_evals,
            max_num_evals,
            polys,
            queried_simple_selector,
            queried_selectors,
            queried_fixed,
            queried_instance,
            queried_advice,
            challenge_evals,
            simple_selector: None,
            selectors,
            fixed,
            instance_acc,
            instance_diff,
            advice_acc,
            advice_diff,
            gate_eval,
        }
    }

    /// Fills the local variables buffers with data from the accumulator and new transcript
    fn populate<C>(
        &mut self,
        row: usize,
        isize: i32,
        pk: &ProvingKey<C>,
        acc: &Accumulator<C>,
        new: &Accumulator<C>,
    ) where
        C: CurveAffine<ScalarExt = F>,
    {
        /// Return the index in the polynomial of size `isize` after rotation `rot`.
        fn get_rotation_idx(idx: usize, rot: Rotation, isize: i32) -> usize {
            (((idx as i32) + rot.0).rem_euclid(isize)) as usize
        }

        // Check if the gate is guarded by a simple selector and whether it is active
        if let Some(simple_selector) = self.queried_simple_selector {
            let selector_column = simple_selector.0;
            let value = pk.selectors[selector_column][row];
            self.simple_selector = Some(value);
            // do nothing and return
            if !value {
                return;
            }
        }

        // Fill selectors
        for (i, selector) in self.queried_selectors.iter().enumerate() {
            let selector_column = selector.0;
            let value = pk.selectors[selector_column][row];
            self.selectors[i] = if value { F::ONE } else { F::ZERO };
        }

        // Fill fixed
        for (i, fixed) in self.queried_fixed.iter().enumerate() {
            let fixed_column = fixed.column_index();
            let fixed_row = get_rotation_idx(row, fixed.rotation(), isize);
            self.fixed[i] = pk.fixed[fixed_column][fixed_row];
        }

        // Fill instance
        for (i, instance) in self.queried_instance.iter().enumerate() {
            let instance_column = instance.column_index();
            let instance_row = get_rotation_idx(row, instance.rotation(), isize);
            self.instance_acc[i] =
                acc.instance_transcript.instance_polys[instance_column][instance_row];
            self.instance_diff[i] = new.instance_transcript.instance_polys[instance_column]
                [instance_row]
                - self.instance_acc[i];
        }

        // Fill advice
        for (i, advice) in self.queried_advice.iter().enumerate() {
            let advice_column = advice.column_index();
            let advice_row = get_rotation_idx(row, advice.rotation(), isize);
            self.advice_acc[i] = acc.advice_transcript.advice_polys[advice_column][advice_row];
            self.advice_diff[i] =
                new.advice_transcript.advice_polys[advice_column][advice_row] - self.advice_acc[i];
        }
    }

    /// Evaluates the error polynomial for the populated row.
    /// Returns `None` if the common selector for the gate is false,
    /// otherwise returns a list of vectors containing the evaluations for
    /// each `poly` Gⱼ(X) in `gate`.
    pub fn evaluate<C>(
        &mut self,
        row: usize,
        isize: i32,
        pk: &ProvingKey<C>,
        acc: &Accumulator<C>,
        new: &Accumulator<C>,
    ) -> Option<&[Vec<F>]>
    where
        C: CurveAffine<ScalarExt = F>,
    {
        // Fill the buffers with data from the fixed circuit_data, the previous accumulator
        // and the new witness.
        self.populate(row, isize, pk, acc, new);

        // exit early if there is a simple selector and it is off
        if let Some(simple_selector_value) = self.simple_selector {
            if !simple_selector_value {
                return None;
            }
        }
        let max_num_evals = self.max_num_evals;
        // Use the `*_acc` buffers to store the linear combination evaluations.
        // In the first iteration with X=0, it is already equal to `*_acc`
        let instance_tmp = &mut self.instance_acc;
        let advice_tmp = &mut self.advice_acc;

        // Iterate over all evaluations points X = 0, ..., max_num_evals-1
        for eval_idx in 0..max_num_evals {
            // After the first iteration, add the contents of the new instance and advice to the tmp buffer
            if eval_idx > 0 {
                for (i_tmp, i_diff) in zip(instance_tmp.iter_mut(), self.instance_diff.iter()) {
                    *i_tmp += i_diff;
                }
                for (a_tmp, a_diff) in zip(advice_tmp.iter_mut(), self.advice_diff.iter()) {
                    *a_tmp += a_diff;
                }
            }
            // Iterate over each polynomial constraint Gⱼ, along with its required number of evaluations
            for (poly_idx, (poly, num_evals)) in
                zip(self.polys.iter(), self.num_evals.iter()).enumerate()
            {
                // If the eval_idx X is larger than the required number of evaluations for the current poly,
                // we don't evaluate it and continue to the next poly.
                if eval_idx > *num_evals {
                    continue;
                }
                // evaluate the j-th constraint G_j at X = eval_idx
                let e = poly.evaluate(
                    &|constant| constant,
                    &|selector_idx| self.selectors[selector_idx],
                    &|fixed_idx| self.fixed[fixed_idx],
                    &|advice_idx| advice_tmp[advice_idx],
                    &|instance_idx| instance_tmp[instance_idx],
                    &|challenge_idx| self.challenge_evals[eval_idx][challenge_idx],
                    &|negated| -negated,
                    &|sum_a, sum_b| sum_a + sum_b,
                    &|prod_a, prod_b| prod_a * prod_b,
                    &|scaled, v| scaled * v,
                );
                self.gate_eval[poly_idx][eval_idx] = e;
            }
        }
        Some(&self.gate_eval)
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

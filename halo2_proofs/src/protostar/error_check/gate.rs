use std::{collections::BTreeSet, iter::zip};

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    plonk::{AdviceQuery, Challenge, Expression, FixedQuery, Gate, InstanceQuery, Selector},
    poly::{LagrangeCoeff, Polynomial, Rotation},
    protostar::keygen::ProvingKey,
};

use super::{
    boolean_evaluations_vec, boolean_evaluations_vec_skip_2, Accumulator, MIN_GATE_DEGREE,
    NUM_EXTRA_EVALUATIONS, NUM_SKIPPED_EVALUATIONS,
};

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

    simple_selector_col: Option<usize>,
    challenges_evals: Vec<Vec<F>>,
    // List of polynomial expressions Gⱼ
    queried_polys: Vec<QueriedExpression<F>>,

    row: Row<F>,

    errors_evals: Vec<Vec<F>>,
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
        challenges_acc: &[Vec<F>],
        challenges_new: &[Vec<F>],
        num_rows_i: i32,
    ) -> Self {
        // Recover common simple `Selector` from the `Gate`, along with the branch `Expression`s  it selects
        let (polys, queried_simple_selector) = extract_common_simple_selector(gate.polynomials());

        // Merge products of challenges and compute degrees, and filter out the polynomials whose degree is ≤1
        let (degrees, polys): (Vec<_>, Vec<_>) = polys
            .into_iter()
            .map(|poly| {
                let poly = poly.merge_challenge_products();
                (poly.folding_degree(), poly)
            })
            .filter(|(d, _)| *d > MIN_GATE_DEGREE)
            .unzip();

        let queries = RowQueries::from_polys(&polys, num_rows_i);

        let queried_polys: Vec<_> = polys
            .iter()
            .map(|poly| queries.queried_expression(poly))
            .collect();

        // Each `poly` Gⱼ(X) has degree dⱼ and dⱼ+1 coefficients,
        // therefore we need at least dⱼ+1 evaluations of Gⱼ(X) to recover
        // the coefficients.
        let num_evals: Vec<_> = degrees
            .iter()
            .map(|d| d + 1 + NUM_EXTRA_EVALUATIONS - NUM_SKIPPED_EVALUATIONS)
            .collect();
        // maximum number of evaluations over all Gⱼ(X)
        let max_num_evals = *num_evals.iter().max().unwrap();

        let queried_challenges_acc = queries.queried_challenges(challenges_acc);
        let queried_challenges_new = queries.queried_challenges(challenges_new);

        let challenges_evals: Vec<_> =
            boolean_evaluations_vec_skip_2(queried_challenges_acc, queried_challenges_new)
                .take(max_num_evals)
                .collect();

        // for each polynomial, allocate a buffer for storing all the evaluations
        let errors_evals = num_evals.iter().map(|d| vec![F::ZERO; *d]).collect();

        let simple_selector_col = queried_simple_selector.map(|selector| selector.index());

        let row = Row::new(queries);
        Self {
            num_evals,
            max_num_evals,
            simple_selector_col,
            challenges_evals,
            queried_polys,
            row,
            errors_evals,
        }
    }

    /// Evaluates the error polynomial for the populated row.
    /// Returns `None` if the common selector for the gate is false,
    /// otherwise returns a list of vectors containing the evaluations for
    /// each `poly` Gⱼ(X) in `gate`.
    pub fn evaluate<C>(
        &mut self,
        row_idx: usize,
        pk: &ProvingKey<C>,
        acc: &Accumulator<C>,
        new: &Accumulator<C>,
    ) -> Option<&[Vec<F>]>
    where
        C: CurveAffine<ScalarExt = F>,
    {
        if let Some(selector_col) = self.simple_selector_col {
            if !pk.selectors[selector_col][row_idx] {
                return None;
            }
        }
        self.row.populate_selectors(row_idx, &pk.selectors);
        self.row.populate_fixed(row_idx, &pk.fixed);
        self.row.populate_advice_evals_skip_1(
            row_idx,
            &acc.advice_transcript.advice_polys,
            &new.advice_transcript.advice_polys,
        );
        self.row.populate_instance_evals_skip_1(
            row_idx,
            &acc.instance_transcript.instance_polys,
            &new.instance_transcript.instance_polys,
        );

        let max_num_evals = self.max_num_evals;

        // Iterate over all evaluations points X = 0, ..., max_num_evals-1
        for eval_idx in 0..max_num_evals {
            self.row.populate_next_evals();

            // Iterate over each polynomial constraint Gⱼ, along with its required number of evaluations
            for (poly_idx, (poly, num_evals)) in
                zip(self.queried_polys.iter(), self.num_evals.iter()).enumerate()
            {
                // If the eval_idx X is larger than the required number of evaluations for the current poly,
                // we don't evaluate it and continue to the next poly.
                if eval_idx > *num_evals {
                    continue;
                }
                self.errors_evals[poly_idx][eval_idx] =
                    self.row.evaluate(poly, &self.challenges_evals[eval_idx]);
            }
        }
        Some(&self.errors_evals)
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

struct RowQueries {
    selectors: Vec<usize>,
    fixed: Vec<(usize, Rotation)>,
    instance: Vec<(usize, Rotation)>,
    advice: Vec<(usize, Rotation)>,
    challenges: Vec<(usize, usize)>,
    num_rows_i: i32,
}

impl RowQueries {
    fn from_polys<F: Field>(polys: &[Expression<F>], num_rows_i: i32) -> Self {
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
            num_rows_i,
        }
    }

    /// Given lists of all leaves of the original `Expression`, create an `Expr` where the leaves
    /// correspond to indices of the variable in the lists.
    pub fn queried_expression<F: Field>(&self, poly: &Expression<F>) -> QueriedExpression<F> {
        fn get_idx<T: PartialEq>(container: &Vec<T>, elem: T) -> usize {
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

    fn queried_challenges<F: Field>(&self, challenges: &[Vec<F>]) -> Vec<F> {
        self.challenges
            .iter()
            .map(|query| challenges[query.0][query.1])
            .collect()
    }

    fn iter_with_rotations<'c, 's: 'c, F: Field>(
        row_idx: usize,
        num_rows_i: i32,
        queries: &'s [(usize, Rotation)],
        columns: &'c [Polynomial<F, LagrangeCoeff>],
    ) -> impl Iterator<Item = F> + 'c {
        queries.iter().map(move |(column_idx, rotation)| {
            let row_idx = (((row_idx as i32) + rotation.0).rem_euclid(num_rows_i)) as usize;
            columns[*column_idx][row_idx]
        })
    }

    fn iter_selector_row<'c, 's: 'c, F: Field>(
        &'s self,
        row_idx: usize,
        columns: &'c [Vec<bool>],
    ) -> impl Iterator<Item = F> + 'c {
        self.selectors.iter().map(move |column_idx| {
            if columns[*column_idx][row_idx] {
                F::ONE
            } else {
                F::ZERO
            }
        })
    }

    fn iter_fixed_row<'c, 's: 'c, F: Field>(
        &'s self,
        row_idx: usize,
        columns: &'c [Polynomial<F, LagrangeCoeff>],
    ) -> impl Iterator<Item = F> + 'c {
        Self::iter_with_rotations(row_idx, self.num_rows_i, &self.fixed, columns)
    }

    fn iter_advice_row<'c, 's: 'c, F: Field>(
        &'s self,
        row_idx: usize,
        columns: &'c [Polynomial<F, LagrangeCoeff>],
    ) -> impl Iterator<Item = F> + 'c {
        Self::iter_with_rotations(row_idx, self.num_rows_i, &self.advice, columns)
    }

    fn iter_instance_row<'c, 's: 'c, F: Field>(
        &'s self,
        row_idx: usize,
        columns: &'c [Polynomial<F, LagrangeCoeff>],
    ) -> impl Iterator<Item = F> + 'c {
        Self::iter_with_rotations(row_idx, self.num_rows_i, &self.advice, columns)
    }
}

struct Row<F: Field> {
    selectors: Vec<F>,
    fixed: Vec<F>,
    instance: Vec<F>,
    advice: Vec<F>,
    instance_diff: Vec<F>,
    advice_diff: Vec<F>,
    queries: RowQueries,
}

impl<F: Field> Row<F> {
    fn new(queries: RowQueries) -> Self {
        Self {
            selectors: vec![F::ZERO; queries.selectors.len()],
            fixed: vec![F::ZERO; queries.fixed.len()],
            instance: vec![F::ZERO; queries.instance.len()],
            advice: vec![F::ZERO; queries.advice.len()],
            instance_diff: vec![F::ZERO; queries.instance.len()],
            advice_diff: vec![F::ZERO; queries.advice.len()],
            queries,
        }
    }

    fn populate_selectors(&mut self, row_idx: usize, selectors: &[Vec<bool>]) {
        self.selectors.clear();
        self.selectors
            .extend(self.queries.iter_selector_row::<F>(row_idx, selectors));
    }

    fn populate_fixed(&mut self, row_idx: usize, fixed: &[Polynomial<F, LagrangeCoeff>]) {
        self.fixed.clear();
        self.fixed
            .extend(self.queries.iter_fixed_row(row_idx, fixed));
    }

    fn populate_instance(&mut self, row_idx: usize, instance: &[Polynomial<F, LagrangeCoeff>]) {
        self.instance.clear();
        self.instance
            .extend(self.queries.iter_instance_row(row_idx, instance));
    }

    fn populate_advice(&mut self, row_idx: usize, advice: &[Polynomial<F, LagrangeCoeff>]) {
        self.advice.clear();
        self.advice
            .extend(self.queries.iter_advice_row(row_idx, advice));
    }

    fn populate_instance_evals_skip_1(
        &mut self,
        row_idx: usize,
        instance_evals0: &[Polynomial<F, LagrangeCoeff>],
        instance_evals1: &[Polynomial<F, LagrangeCoeff>],
    ) {
        for ((curr, diff), (eval0, eval1)) in zip(
            zip(self.instance.iter_mut(), self.instance_diff.iter_mut()),
            zip(
                self.queries.iter_instance_row(row_idx, instance_evals0),
                self.queries.iter_instance_row(row_idx, instance_evals1),
            ),
        ) {
            *curr = eval1;
            *diff = eval1 - eval0;
        }
    }

    fn populate_advice_evals_skip_1(
        &mut self,
        row_idx: usize,
        advice_evals0: &[Polynomial<F, LagrangeCoeff>],
        advice_evals1: &[Polynomial<F, LagrangeCoeff>],
    ) {
        for ((curr, diff), (eval0, eval1)) in zip(
            zip(self.advice.iter_mut(), self.advice_diff.iter_mut()),
            zip(
                self.queries.iter_advice_row(row_idx, advice_evals0),
                self.queries.iter_advice_row(row_idx, advice_evals1),
            ),
        ) {
            *curr = eval1;
            *diff = eval1 - eval0;
        }
    }

    fn populate_next_evals(&mut self) {
        for (curr, diff) in zip(self.instance.iter_mut(), self.instance_diff.iter()) {
            *curr += diff;
        }
        for (curr, diff) in zip(self.advice.iter_mut(), self.advice_diff.iter()) {
            *curr += diff;
        }
    }

    /// Fills the local variables buffers with data from the accumulator and new transcript
    fn populate_all(
        &mut self,
        row_idx: usize,
        selectors: &[Vec<bool>],
        fixed: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
    ) {
        self.populate_selectors(row_idx, selectors);
        self.populate_fixed(row_idx, fixed);
        self.populate_instance(row_idx, instance);
        self.populate_advice(row_idx, advice);
    }

    fn evaluate(&self, poly: &QueriedExpression<F>, challenges: &[F]) -> F {
        // evaluate the j-th constraint G_j at X = eval_idx
        poly.evaluate(
            &|constant| constant,
            &|selector_idx| self.selectors[selector_idx],
            &|fixed_idx| self.fixed[fixed_idx],
            &|advice_idx| self.advice[advice_idx],
            &|instance_idx| self.instance[instance_idx],
            &|challenge_idx| challenges[challenge_idx],
            &|negated| -negated,
            &|sum_a, sum_b| sum_a + sum_b,
            &|prod_a, prod_b| prod_a * prod_b,
            &|scaled, v| scaled * v,
        )
    }
}

/// Undo `Constraints::WithSelector` and return the common top-level `Selector` along with the expressions it selects.
/// If no simple `Selector` is found, returns the original list of polynomials.
pub fn extract_common_simple_selector<F: Field>(
    polys: &[Expression<F>],
) -> (Vec<Expression<F>>, Option<Selector>) {
    let (extracted_polys, simple_selectors): (Vec<_>, Vec<_>) = polys
        .iter()
        .map(|poly| {
            // Check whether the top node is a multiplication by a selector
            let (simple_selector, poly) = match poly {
                // If the whole polynomial is multiplied by a simple selector,
                // return it along with the expression it selects
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
        (polys.to_vec(), None)
    } else {
        (extracted_polys, potential_selector)
    }
}

/// Low-degree expression representing an identity that must hold over the committed columns.
#[derive(Clone)]
pub enum QueriedExpression<F> {
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
    Negated(Box<QueriedExpression<F>>),
    /// This is the sum of two polynomials
    Sum(Box<QueriedExpression<F>>, Box<QueriedExpression<F>>),
    /// This is the product of two polynomials
    Product(Box<QueriedExpression<F>>, Box<QueriedExpression<F>>),
    /// This is a scaled polynomial
    Scaled(Box<QueriedExpression<F>>, F),
}

impl<F: Field> QueriedExpression<F> {
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
            QueriedExpression::Constant(scalar) => constant(*scalar),
            QueriedExpression::Selector(selector) => selector_column(*selector),
            QueriedExpression::Fixed(query) => fixed_column(*query),
            QueriedExpression::Advice(query) => advice_column(*query),
            QueriedExpression::Instance(query) => instance_column(*query),
            QueriedExpression::Challenge(value) => challenge(*value),
            QueriedExpression::Negated(a) => {
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
            QueriedExpression::Sum(a, b) => {
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
            QueriedExpression::Product(a, b) => {
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
            QueriedExpression::Scaled(a, f) => {
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

impl<F: std::fmt::Debug + Field> std::fmt::Debug for QueriedExpression<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueriedExpression::Constant(scalar) => f.debug_tuple("Constant").field(scalar).finish(),
            QueriedExpression::Selector(selector) => {
                f.debug_tuple("Selector").field(selector).finish()
            }
            // Skip enum variant and print query struct directly to maintain backwards compatibility.
            QueriedExpression::Fixed(query) => f.debug_tuple("Fixed").field(query).finish(),
            QueriedExpression::Advice(query) => f.debug_tuple("Advice").field(query).finish(),
            QueriedExpression::Instance(query) => f.debug_tuple("Instance").field(query).finish(),
            QueriedExpression::Challenge(c) => f.debug_tuple("Challenge").field(c).finish(),
            QueriedExpression::Negated(poly) => f.debug_tuple("Negated").field(poly).finish(),
            QueriedExpression::Sum(a, b) => f.debug_tuple("Sum").field(a).field(b).finish(),
            QueriedExpression::Product(a, b) => f.debug_tuple("Product").field(a).field(b).finish(),
            QueriedExpression::Scaled(poly, scalar) => {
                f.debug_tuple("Scaled").field(poly).field(scalar).finish()
            }
        }
    }
}

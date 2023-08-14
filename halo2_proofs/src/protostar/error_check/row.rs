use std::{collections::BTreeSet, iter::zip};

use ff::Field;

use crate::{
    plonk::Expression,
    poly::{LagrangeCoeff, Polynomial, Rotation},
};

// For a given `Expression`, stores all queried variables.
pub struct RowQueries {
    selectors: Vec<usize>,
    fixed: Vec<(usize, Rotation)>,
    instance: Vec<(usize, Rotation)>,
    advice: Vec<(usize, Rotation)>,
    challenges: Vec<(usize, usize)>,
}

impl RowQueries {
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
/// A `Row` contains buffers for storing the values defined by the queries in `RowQueries`.
/// Values are populated, possibly interpolated, and then evaluated for a given `QueriedPolynomial`.
pub struct Row<F: Field> {
    selectors: Vec<F>,
    fixed: Vec<F>,
    instance: Vec<F>,
    advice: Vec<F>,
    instance_diff: Vec<F>,
    advice_diff: Vec<F>,
    queries: RowQueries,
}

impl<F: Field> Row<F> {
    /// Create new buffers of the same size as the query sets in `queries`.
    pub fn new(queries: RowQueries) -> Self {
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

    /// Fetch the queried selectors.
    pub fn populate_selectors(&mut self, row_idx: usize, columns: &[Vec<bool>]) {
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
    pub fn populate_fixed(&mut self, row_idx: usize, columns: &[Polynomial<F, LagrangeCoeff>]) {
        Self::fill_row_with_rotations(&mut self.fixed, row_idx, &self.queries.fixed, columns)
    }

    /// Fetch the row values from queried instance columns.
    pub fn populate_instance(&mut self, row_idx: usize, columns: &[Polynomial<F, LagrangeCoeff>]) {
        Self::fill_row_with_rotations(&mut self.instance, row_idx, &self.queries.instance, columns)
    }

    /// Fetch the row values from queried advice columns.
    pub fn populate_advice(&mut self, row_idx: usize, columns: &[Polynomial<F, LagrangeCoeff>]) {
        Self::fill_row_with_rotations(&mut self.advice, row_idx, &self.queries.advice, columns)
    }

    /// Fetch the row values from two queried sets of instance columns.
    /// Each pair is interpreted as boolean low-degree extension.
    /// `self.instance` holds the evaluation at 1, and
    /// `self.instance_diff` contains the linear coefficient of the polynomial.
    /// Calling `self.populate_next_evals` will set `self.instance` to be the evaluation at 2.
    pub fn populate_instance_evals_skip_1(
        &mut self,
        row_idx: usize,
        columns_evals0: &[Polynomial<F, LagrangeCoeff>],
        columns_evals1: &[Polynomial<F, LagrangeCoeff>],
    ) {
        // instance = evals1
        Self::fill_row_with_rotations(
            &mut self.instance,
            row_idx,
            &self.queries.instance,
            columns_evals1,
        );
        // diff = evals0
        Self::fill_row_with_rotations(
            &mut self.instance_diff,
            row_idx,
            &self.queries.instance,
            columns_evals0,
        );
        // diff = eval1 - evals0
        for (eval0, eval1) in self.instance_diff.iter_mut().zip(self.instance.iter()) {
            let diff = *eval1 - *eval0;
            *eval0 = diff;
        }
    }

    /// Fetch the row values from two queried sets of advice columns.
    /// Each pair is interpreted as boolean low-degree extension.
    /// `self.advice` holds the evaluation at 1, and
    /// `self.advice_diff` contains the linear coefficient of the polynomial.
    /// Calling `self.populate_next_evals` will set `self.advice` to be the evaluation at 2.
    pub fn populate_advice_evals_skip_1(
        &mut self,
        row_idx: usize,
        columns_evals0: &[Polynomial<F, LagrangeCoeff>],
        columns_evals1: &[Polynomial<F, LagrangeCoeff>],
    ) {
        // advice = evals1
        Self::fill_row_with_rotations(
            &mut self.advice,
            row_idx,
            &self.queries.advice,
            columns_evals1,
        );
        // diff = evals0
        Self::fill_row_with_rotations(
            &mut self.advice_diff,
            row_idx,
            &self.queries.advice,
            columns_evals0,
        );
        // diff = eval1 - evals0
        for (eval0, eval1) in self.advice_diff.iter_mut().zip(self.advice.iter()) {
            let diff = *eval1 - *eval0;
            *eval0 = diff;
        }
    }

    /// Updates both `self.instance` and `self.advice` to the next evaluation over the integers
    /// of the boolean low-degree extensions they correspond to.
    pub fn populate_next_evals(&mut self) {
        for (curr, diff) in zip(self.instance.iter_mut(), self.instance_diff.iter()) {
            *curr += diff;
        }
        for (curr, diff) in zip(self.advice.iter_mut(), self.advice_diff.iter()) {
            *curr += diff;
        }
    }

    /// Fills the local variables buffers with data from the accumulator and new transcript
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
        self.populate_instance(row_idx, instance);
        self.populate_advice(row_idx, advice);
    }

    /// Evaluate `poly` with the current values stored in the buffers.
    pub fn evaluate(&self, poly: &QueriedExpression<F>, challenges: &[F]) -> F {
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

    fn fill_row_with_rotations(
        row: &mut [F],
        row_idx: usize,

        queries: &[(usize, Rotation)],
        columns: &[Polynomial<F, LagrangeCoeff>],
    ) {
        debug_assert_eq!(row.len(), queries.len());
        for (row_value, (column_idx, rotation)) in row.iter_mut().zip(queries.iter()) {
            // ignore overflow since these should not occur in gates
            let row_idx = (row_idx as i32 + rotation.0) as usize;
            // let row_idx = (((row_idx as i32) + rotation.0).rem_euclid(num_rows_i)) as usize;
            *row_value = columns[*column_idx][row_idx]
        }
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

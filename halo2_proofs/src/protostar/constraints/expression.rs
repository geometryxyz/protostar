use std::ops::{Add, Mul, Neg, Sub};

use ff::Field;

/// Mirror of a `plonk::Expression` where nodes have been moved to a `Queries` structure, and replaced with their indices therein.
#[derive(Clone)]
pub enum Expression<F, CV, FV, WV> {
    Constant(F),
    Challenge(CV),
    Fixed(FV),
    Witness(WV),
    Negated(Box<Expression<F, CV, FV, WV>>),
    Sum(
        Box<Expression<F, CV, FV, WV>>,
        Box<Expression<F, CV, FV, WV>>,
    ),
    Product(
        Box<Expression<F, CV, FV, WV>>,
        Box<Expression<F, CV, FV, WV>>,
    ),
}

impl<F, CV, FV, WV> Expression<F, CV, FV, WV> {
    /// Returns the degree of the Expression, considering Challenge and Witness leaves as variables.
    pub fn degree(&self) -> usize {
        match self {
            Self::Constant(_) => 0,
            Self::Challenge(_) => 1,
            Self::Fixed(_) => 0,
            Self::Witness(_) => 1,
            Self::Negated(a) => a.degree(),
            Self::Sum(a, b) => std::cmp::max(a.degree(), b.degree()),
            Self::Product(a, b) => a.degree() + b.degree(),
        }
    }

    /// Applies `f` to all leaves
    pub fn traverse(&self, f: &mut impl FnMut(&Self)) {
        match self {
            Self::Negated(e) => e.traverse(f),
            Self::Sum(e1, e2) => {
                e1.traverse(f);
                e2.traverse(f);
            }
            Self::Product(e1, e2) => {
                e1.traverse(f);
                e2.traverse(f);
            }
            v => f(v),
        }
    }

    /// Evaluate the polynomial using the provided closures to perform the operations.
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        challenge: &impl Fn(CV) -> T,
        fixed: &impl Fn(FV) -> T,
        witness: &impl Fn(WV) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
    ) -> T
    where
        F: Copy,
        CV: Copy,
        FV: Copy,
        WV: Copy,
    {
        match self {
            Self::Constant(v) => constant(*v),
            Self::Challenge(v) => challenge(*v),
            Self::Fixed(v) => fixed(*v),
            Self::Witness(v) => witness(*v),
            Self::Negated(a) => {
                let a = a.evaluate(constant, challenge, fixed, witness, negated, sum, product);
                negated(a)
            }
            Self::Sum(a, b) => {
                let a = a.evaluate(constant, challenge, fixed, witness, negated, sum, product);
                let b = b.evaluate(constant, challenge, fixed, witness, negated, sum, product);
                sum(a, b)
            }
            Self::Product(a, b) => {
                let a = a.evaluate(constant, challenge, fixed, witness, negated, sum, product);
                let b = b.evaluate(constant, challenge, fixed, witness, negated, sum, product);
                product(a, b)
            }
        }
    }

    /// Evaluate the polynomial using the provided mutable closures to perform the operations.
    pub fn evaluate_mut<T>(
        &self,
        constant: &mut impl FnMut(F) -> T,
        challenge: &mut impl FnMut(CV) -> T,
        fixed: &mut impl FnMut(FV) -> T,
        witness: &mut impl FnMut(WV) -> T,
        negated: &mut impl FnMut(T) -> T,
        sum: &mut impl FnMut(T, T) -> T,
        product: &mut impl FnMut(T, T) -> T,
    ) -> T
    where
        F: Copy,
        CV: Copy,
        FV: Copy,
        WV: Copy,
    {
        match self {
            Self::Constant(v) => constant(*v),
            Self::Challenge(v) => challenge(*v),
            Self::Fixed(v) => fixed(*v),
            Self::Witness(v) => witness(*v),
            Self::Negated(a) => {
                let a = a.evaluate_mut(constant, challenge, fixed, witness, negated, sum, product);
                negated(a)
            }
            Self::Sum(a, b) => {
                let a = a.evaluate_mut(constant, challenge, fixed, witness, negated, sum, product);
                let b = b.evaluate_mut(constant, challenge, fixed, witness, negated, sum, product);
                sum(a, b)
            }
            Self::Product(a, b) => {
                let a = a.evaluate_mut(constant, challenge, fixed, witness, negated, sum, product);
                let b = b.evaluate_mut(constant, challenge, fixed, witness, negated, sum, product);
                product(a, b)
            }
        }
    }
}

impl<F, CV, FV, WV> Neg for Expression<F, CV, FV, WV> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Expression::Negated(Box::new(self))
    }
}

impl<F, CV, FV, WV> Add for Expression<F, CV, FV, WV> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Expression::Sum(Box::new(self), Box::new(rhs))
    }
}

impl<F, CV, FV, WV> Sub for Expression<F, CV, FV, WV> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Expression::Sum(Box::new(self), Box::new(-rhs))
    }
}

impl<F, CV, FV, WV> Mul for Expression<F, CV, FV, WV> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Expression::Product(Box::new(self), Box::new(rhs))
    }
}

/// Reference to a specific column variable with a given rotation
#[derive(Clone, Copy, PartialEq)]
pub struct ColumnQuery<T> {
    pub column: T,
    pub rotation: i32,
}

impl<T> ColumnQuery<T> {
    pub fn row_idx(&self, idx: usize, num_rows: usize) -> usize {
        let idx = self.rotation + idx as i32;
        idx.rem_euclid(num_rows as i32) as usize
    }
}

/// Reference to a challenge variable
#[derive(Clone, Copy, PartialEq)]
pub struct ChallengeQuery<T> {
    pub value: T,
}

/// An `Expression` where leaves are references to the underlying data.
pub type QueriedExpression<T> = Expression<
    <T as QueryType>::F,
    ChallengeQuery<<T as QueryType>::Challenge>,
    ColumnQuery<<T as QueryType>::Fixed>,
    ColumnQuery<<T as QueryType>::Witness>,
>;

/// Container type for data contained in a QueriedExpression
pub trait QueryType {
    // Field
    type F: Field;
    // ChallengeVariable
    type Challenge: Copy + PartialEq;
    // FixedVariable
    type Fixed: Copy + PartialEq;
    // WitnessVariable
    type Witness: Copy + PartialEq;

    /// Convert a plonk::Expression into a QueriedExpression, where leaves contain references to the underlying data.
    fn from_expression(
        expr: &crate::plonk::Expression<Self::F>,
        selectors: &[Self::Fixed],
        fixed: &[Self::Fixed],
        instance: &[Self::Witness],
        advice: &[Self::Witness],
        challenges: &[Self::Challenge],
    ) -> QueriedExpression<Self> {
        expr.evaluate(
            &QueriedExpression::<Self>::Constant,
            &|selector_column| {
                QueriedExpression::<Self>::Fixed(ColumnQuery {
                    column: selectors[selector_column.index()],
                    rotation: 0,
                })
            },
            &|fixed_column| {
                QueriedExpression::<Self>::Fixed(ColumnQuery {
                    column: fixed[fixed_column.column_index()],
                    rotation: fixed_column.rotation().0,
                })
            },
            &|advice_column| {
                QueriedExpression::<Self>::Witness(ColumnQuery {
                    column: advice[advice_column.column_index()],

                    rotation: advice_column.rotation().0,
                })
            },
            &|instance_column| {
                QueriedExpression::<Self>::Witness(ColumnQuery {
                    column: instance[instance_column.column_index()],
                    rotation: instance_column.rotation().0,
                })
            },
            &|challenge| {
                QueriedExpression::<Self>::Challenge(ChallengeQuery {
                    value: challenges[challenge.index()],
                })
            },
            &|negated| QueriedExpression::<Self>::Negated(negated.into()),
            &|a, b| QueriedExpression::<Self>::Sum(a.into(), b.into()),
            &|a, b| QueriedExpression::<Self>::Product(a.into(), b.into()),
            &|a, b| {
                QueriedExpression::<Self>::Product(
                    a.into(),
                    QueriedExpression::<Self>::Constant(b).into(),
                )
            },
        )
    }

    /// Create a Constant QueriedExpression
    fn new_constant(value: Self::F) -> QueriedExpression<Self> {
        QueriedExpression::<Self>::Constant(value)
    }

    /// Create a Witness QueriedExpression with 0 rotation
    fn new_witness(value: Self::Witness) -> QueriedExpression<Self> {
        QueriedExpression::<Self>::Witness(ColumnQuery {
            column: value,
            rotation: 0,
        })
    }

    /// Create a Fixed QueriedExpression with 0 rotation
    fn new_fixed(value: Self::Fixed) -> QueriedExpression<Self> {
        QueriedExpression::<Self>::Fixed(ColumnQuery {
            column: value,
            rotation: 0,
        })
    }

    /// Create a Challenge QueriedExpression
    fn new_challenge(value: Self::Challenge) -> QueriedExpression<Self> {
        QueriedExpression::<Self>::Challenge(ChallengeQuery { value })
    }

    fn linear_combination(
        lhs: &[QueriedExpression<Self>],
        rhs: &[QueriedExpression<Self>],
    ) -> QueriedExpression<Self> {
        assert_eq!(lhs.len(), rhs.len());
        let products = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| lhs.clone() * rhs.clone());

        Self::sum(products)
    }

    fn sum(values: impl IntoIterator<Item = QueriedExpression<Self>>) -> QueriedExpression<Self> {
        let mut i = values.into_iter();
        let first = i.next().unwrap();
        i.fold(first, |acc, next| acc + next)
    }
}

/// Derived from a QueriedExpression, where the leaves are collected into separate structures,
/// and the expression leaves point to indices in the query sets.
pub struct IndexedExpression<T: QueryType> {
    pub expr: Expression<T::F, usize, usize, usize>,
    pub challenges: Vec<ChallengeQuery<T::Challenge>>,
    pub fixed: Vec<ColumnQuery<T::Fixed>>,
    pub witness: Vec<ColumnQuery<T::Witness>>,
}

impl<T: QueryType> IndexedExpression<T> {
    pub fn new(expr: QueriedExpression<T>) -> Self {
        fn find_or_insert<T: PartialEq + Copy>(container: &mut Vec<T>, elem: &T) -> usize {
            if let Some(idx) = container.iter().position(|x| *x == *elem) {
                idx
            } else {
                container.push(*elem);
                container.len() - 1
            }
        }

        // Collect all unique leaves of expr
        let mut challenges = Vec::new();
        let mut fixed = Vec::new();
        let mut witness = Vec::new();

        // Create an expression where the leaves are indices into `queried_*`
        let expr = expr.evaluate_mut(
            &mut |constant| Expression::Constant(constant),
            &mut |c| Expression::Challenge(find_or_insert(&mut challenges, &c)),
            &mut |f| Expression::Fixed(find_or_insert(&mut fixed, &f)),
            &mut |w| Expression::Witness(find_or_insert(&mut witness, &w)),
            &mut |e| Expression::Negated(e.into()),
            &mut |e1, e2| Expression::Sum(e1.into(), e2.into()),
            &mut |e1, e2| Expression::Product(e1.into(), e2.into()),
        );

        Self {
            expr,
            challenges,
            fixed,
            witness,
        }
    }
}

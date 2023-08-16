use ff::Field;

/// Mirror of a `plonk::Expression` where nodes have been moved to a `Queries` structure, and replaced with their indices therein.
#[derive(Clone)]
pub enum QueriedExpression<F> {
    /// This is a constant polynomial
    Constant(F),
    /// This is a virtual selector
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

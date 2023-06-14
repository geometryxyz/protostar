struct Accumulator {}
struct Transcript {}

const D: usize = 5;
const NUM_ADVICE_VARS: usize = 5;
const NUM_FIXED_COLS: usize = 5;
const NUM_CHALLENGES: usize = 5;

// Represents the data required to evaluate e_i(X)
struct Row {
    // not sure we need the index here
    idx: usize,
    // beta_0 * beta_1 = \beta^i
    beta_0: u64,
    beta_1: u64,

    // We want to populate this not only with the current row, but also all the rotations.
    // I.e., the constraint G may depend on rows i+1, i-1 etc.
    advice: [u64; NUM_ADVICE_VARS],
}

struct ChallengeRow {
    // slack = [mu, mu+1, ..., mu+D]
    slack: u64,
    // Evaluation of acc.challenge + X*tx.challenges for X = 0,1,..,D
    challenges_eval: [u64; NUM_CHALLENGES],
}

// index of active gate, and the Lagrange normalization factor for q(q-1)...(q-l)
struct SelectorActivation(usize, u64);

struct FixedRow {
    // index of the only active
    // represents the active row.
    active_constraint: usize,

    // constants for this row
    fixed: [u64; NUM_FIXED_COLS],
}

struct ErrorRow {
    // Evaluations
    evaluations: [u64; D],
}

// represents G, and evaluates it
fn eval_constraint(constants: FixedRow, row: Row, challenges: &ChallengeRow) -> u64 {
    0
}

// at a given row i, compute acc.row[i] + X * tx.row[i]
fn eval_error_at_row(
    constants: FixedRow,
    acc_row: &Row,
    tx_row: &Row,
    challenges: &[ChallengeRow; D],
    degree: usize,
) -> ErrorRow {
    let mut error_row: [u64; D];

    // Here, acc and tx rows would just contain info about where the actual values in acc and tx are located
    // We can do something like row.get(i) to actually grab the data from memory.
    // something like in ValueSource::get.
    // If the row constants contains information about which gate is active,
    // then we only need to grab the data that is relevant for this row
    // (if the selector for a gate is 0, then we don't compute anything so we don't need the data either)
    let mut tmp_row = acc_row;

    // compute e_i(0)
    error_row[0] = eval_constraint(constants, tmp_row, &challenges[0]);
    for l in 1..D {
        // need to impl AddAssign
        // tmp_row += acc_row;

        // compute e_i(l), using the
        error_row[l] = eval_constraint(constants, tmp_row, &challenges[l]);
    }
    ErrorRow {
        evaluations: error_row,
    }
}

fn prove(accumulator: Accumulator, transcript: Transcript) {
    // Ensure iop_transcript is well formed/rederive challenges

    // Sample `alpha`/ValueSource::Y() for random-linear combination of gates?
    // Not necessary with proper selector structure.

    // / // prepare evaluations of challenges ()
    // / challenges: [FF;d]
    // / challenges[0] = acc.challenges
    // / for l : [1..d] {
    // /     challenges[l] = challenges[l-1] + tx.challenges
    // / }

    // / interpolate each row over 1,...,d-1
    // / for row_i : transcript.rows {
    // /     acc_i = accumulator.row
    // /     for l : d-1 {
    // /         acc_i += row_i
    // /
    // /     }
    // / }
}

use std::rc::Rc;

#[derive(Clone)]
enum Node {
    Var(String),
    Sum(Vec<Rc<Node>>),
    Product(Vec<Rc<Node>>),
    Power(Rc<Node>, i32),
}

fn degree(node: &Rc<Node>) -> i32 {
    match &**node {
        Node::Var(_) => 1,
        Node::Sum(children) => children.iter().map(degree).max().unwrap(),
        Node::Product(children) => children.iter().map(degree).sum(),
        Node::Power(base, exponent) => degree(base) * exponent,
    }
}

fn homogenize(node: Rc<Node>, total_degree: i32, new_var: &str) -> Rc<Node> {
    match &*node {
        Node::Var(_) => node.clone(),
        Node::Sum(children) => {
            let mut new_children = vec![];
            for child in children {
                let child_degree = degree(child);
                if child_degree < total_degree {
                    new_children.push(Rc::new(Node::Product(vec![
                        homogenize(child.clone(), total_degree, new_var),
                        Rc::new(Node::Power(
                            Rc::new(Node::Var(new_var.to_string())),
                            total_degree - child_degree,
                        )),
                    ])));
                } else {
                    new_children.push(homogenize(child.clone(), total_degree, new_var));
                }
            }
            Rc::new(Node::Sum(new_children))
        }
        Node::Product(children) => Rc::new(Node::Product(
            children
                .iter()
                .map(|child| homogenize(child.clone(), total_degree, new_var))
                .collect(),
        )),
        Node::Power(_, _) => panic!("Power nodes are not supported in the original polynomial"),
    }
}

fn homogenize_ast(root: Rc<Node>, new_var: &str) -> Rc<Node> {
    let total_degree = degree(&root);
    homogenize(root, total_degree, new_var)
}

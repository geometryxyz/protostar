use std::iter::zip;

use ff::Field;
use halo2curves::CurveAffine;

use crate::{
    plonk::{self, lookup},
    poly::{Basis, LagrangeCoeff, Polynomial},
};

use self::expression::{QueriedExpression, QueryType};

use super::{accumulator::Accumulator, ProvingKey};

pub(crate) mod expression;
pub(crate) mod paired;
pub(crate) mod polynomial;

/// Used to store references of the variables referenced by an expression. 
/// TODO: Merge with Verifier and Prover accumulator
pub struct Data<T: QueryType> {
    fixed: Vec<T::Fixed>,
    selectors: Vec<T::Fixed>,
    instance: Vec<T::Witness>,
    advice: Vec<T::Witness>,
    challenges: Vec<T::Challenge>,
    beta: T::Witness,
    lookups: Vec<LookupData<T>>,
    ys: Vec<T::Challenge>,
}

pub struct LookupData<T: QueryType> {
    m: T::Witness,
    g: T::Witness,
    h: T::Witness,
    thetas: Vec<T::Challenge>,
    r: T::Challenge,
}

impl<T: QueryType> Data<T> {
    pub fn all_constraints(
        &self,
        gates: &[plonk::circuit::Gate<T::F>],
        lookups: &[plonk::lookup::Argument<T::F>],
    ) -> Vec<QueriedExpression<T>> {
        let gate_constraints: Vec<_> = gates
            .iter()
            .flat_map(|gate| {
                gate.polynomials().iter().map(|expr| {
                    T::from_expression(
                        expr,
                        &self.selectors,
                        &self.fixed,
                        &self.instance,
                        &self.advice,
                        &self.challenges,
                    )
                })
            })
            .collect();

        let lookup_constraints: Vec<_> = lookups
            .iter()
            .zip(self.lookups.iter())
            .flat_map(|(arg, data)| {
                // Get expressions for inputs input_0, ..., input_k
                let inputs = arg
                    .input_expressions()
                    .iter()
                    .map(|e| {
                        T::from_expression(
                            e,
                            &self.selectors,
                            &self.fixed,
                            &self.instance,
                            &self.advice,
                            &self.challenges,
                        )
                    })
                    .collect::<Vec<_>>();

                // Get expressions for tables table_0, ..., table_k
                let tables = arg
                    .table_expressions()
                    .iter()
                    .map(|e| {
                        T::from_expression(
                            e,
                            &self.selectors,
                            &self.fixed,
                            &self.instance,
                            &self.advice,
                            &self.challenges,
                        )
                    })
                    .collect::<Vec<_>>();

                // Get expressions for variables r, m, g, h
                let r = T::new_challenge(data.r);
                let m = T::new_witness(data.m);
                let g = T::new_witness(data.g);
                let h = T::new_witness(data.h);

                // Get expressions for variables theta_0, ..., theta_k
                let thetas = data
                    .thetas
                    .iter()
                    .map(|theta| T::new_challenge(*theta))
                    .collect::<Vec<_>>();

                let one = T::new_constant(T::F::ONE);

                // h * (r + theta_1 * input_1 + ... + theta_k * input_k )
                let input_constraint =
                    h * zip(inputs, thetas.iter()).fold(r.clone(), |acc, (input, theta)| {
                        acc + (input * theta.clone())
                    }) - one;

                let table_constraint = g * zip(tables, thetas.iter())
                    .fold(r, |acc, (table, theta)| acc + (table * theta.clone()))
                    - m;
                [input_constraint, table_constraint].into_iter()
            })
            .collect();

        [&gate_constraints[..], &lookup_constraints[..]].concat()
    }

    pub fn full_constraint(
        &self,
        gates: &[plonk::circuit::Gate<T::F>],
        lookups: &[plonk::lookup::Argument<T::F>],
    ) -> QueriedExpression<T> {
        let beta = T::new_witness(self.beta);

        let constraints = self.all_constraints(gates, lookups);

        let ys = self
            .ys
            .iter()
            .map(|y| T::new_challenge(*y))
            .collect::<Vec<_>>();
        beta * T::linear_combination(&constraints, &ys)
    }
}

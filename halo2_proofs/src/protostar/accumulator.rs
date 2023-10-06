use std::iter::{self, zip};

use ff::Field;
use group::Curve;
use halo2curves::CurveAffine;
use rand_core::RngCore;

use crate::{
    arithmetic::{eval_polynomial, parallelize},
    dev::metadata::Gate,
    poly::{
        commitment::{Blind, Params},
        LagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

use self::committed::Committed;

use super::{
    constraints::{paired::Paired, polynomial::CommittedRef, Data},
    ProvingKey,
};

pub(super) mod committed;
pub(super) mod compressed_verifier;
pub(super) mod gate;
pub(super) mod lookup;

/// An `Accumulator` contains the entirety of the IOP transcript,
/// including commitments and verifier challenges.
#[derive(Debug, Clone, PartialEq)]
pub struct Accumulator<C: CurveAffine> {
    pub gate: gate::Transcript<C>,
    pub lookups: Vec<lookup::Transcript<C>>,
    pub beta: compressed_verifier::Transcript<C>,

    pub ys: Vec<C::Scalar>,

    // Error value for all constraints
    pub error: C::Scalar,
}

impl<C: CurveAffine> Accumulator<C> {
    /// Given two accumulators, run the folding reduction to produce a new accumulator.
    /// If both input accumulators are correct, the output accumulator will be correct w.h.p. .
    pub fn fold<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        pk: &ProvingKey<C>,
        acc0: Self,
        acc1: Self,
        transcript: &mut T,
    ) -> Self {
        // Create a data structure containing pairs of committed columns, from which we can compute the constraints to be evaluated.
        let paired_data = Paired::<'_, C::Scalar>::new_data(pk, &acc0, &acc1);

        // Get the full constraint polynomial for the gate and lookups
        let full_constraint = paired_data.full_constraint(pk.cs.gates(), pk.cs.lookups());

        /*
        Compute the error polynomial e(X) = ∑ᵢ βᵢ * Gᵢ(X)
        NOTE: There are sevaral optimizations that can be performed at this point:
        The i-th constraint Gᵢ is given by Gᵢ = ∑ⱼ yⱼ⋅ sⱼ,ᵢ⋅ Gⱼ,ᵢ, where
          - yⱼ is a challenge for keeping all constraints linearly independent.
          - sⱼ,ᵢ is a selector for the j-th constraint, which is 1 if the j-th constraint Gⱼ,ᵢ is active at row i,
            and 0 otherwise.
          - Gⱼ,ᵢ is the the j-th constraint, resulting from partially evaluating an expression Gⱼ in the the fixed
            columns of row i.

        - For each constraint Gⱼ, we can compute the polynomial eⱼ(X) = ∑ᵢ βᵢ⋅sⱼ,ᵢ⋅Gⱼ,ᵢ(X) indpendently of the other
          constraints, and sum them at the end to obtain e(X) = ∑ⱼ yⱼ(X)⋅eⱼ(X).
        - By checking the selectors, we can skip the evaluation of Gⱼ,ᵢ if it is not active at row i.
        - The variables of the expression Gⱼ only need to be interpolated upto deg(Gⱼ)+1 rather than deg(e),
          saving some unnecessary evaluations of Gⱼ.
        - If a constraint G is linear (i.e. Gᵢ = L₀⋅(wᵢ−1) for checking that w₀ == 1, where L₀ is a fixed column)
          then the error polynomial for this expression will always be 0, so we can skip the evaluation
        */
        let error_poly = Paired::<'_, C::Scalar>::evaluate_compressed_polynomial(
            full_constraint,
            pk.usable_rows.clone(),
            pk.num_rows,
        );

        debug_assert_eq!(error_poly.len(), pk.max_folding_constraints_degree() + 1);

        let error_poly_quotient = {
            let error0 = eval_polynomial(&error_poly, C::Scalar::ZERO);
            let error1 = eval_polynomial(&error_poly, C::Scalar::ONE);

            // Sanity checks for ensuring the error polynomial is correct

            assert_eq!(error0, acc0.error);
            assert_eq!(error1, acc1.error);

            let mut error_poly_vanish = error_poly.clone();
            // subtract (1-t)e0 + te1 = e0 + t(e1-e0)
            error_poly_vanish[0] -= error0;
            error_poly_vanish[1] -= error1-error0;
            quotient_by_boolean_vanishing(&error_poly_vanish)
        };

        // Send the coefficients of the error polynomial to the verifier in the clear.
        for coef in &error_poly_quotient {
            let _ = transcript.write_scalar(*coef);
        }
        /*
        Note: The verifier will have to check that e(0) = acc0.error and e(1) = acc1.error.
        We can instead send the quotient
                e(X) - (1-X)e(0) - Xe(1)
        e'(X) = ------------------------
                         (1-X)X
        and let the verifier compute
        e(α) = (1-α)α⋅e'(α) + (1-α)⋅e₀ + α⋅e₁
        */

        // Sample ₀₁, a challenge for computing the interpolation of both accumulators.
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();
        let error = eval_polynomial(&error_poly, alpha);

        let gate = gate::Transcript::merge(alpha, acc0.gate, acc1.gate);
        let lookups = zip(acc0.lookups.into_iter(), acc1.lookups.into_iter())
            .map(|(lookup0, lookup1)| lookup::Transcript::merge(alpha, lookup0, lookup1))
            .collect();
        let beta = compressed_verifier::Transcript::merge(alpha, acc0.beta, acc1.beta);

        let ys = zip(acc0.ys.into_iter(), acc1.ys.into_iter())
            .map(|(y0, y1)| y0 + alpha * (y1 - y0))
            .collect();

        Self {
            gate,
            lookups,
            beta,
            ys,
            error,
        }
    }

    /// Checks whether the accumulator is valid with regards to the proving key.
    /// - Check all commitments are correct
    /// - Check the error term is correct
    /// - Verify the linear lookup constraints skipped during folding
    /// - Check the correctness of the beta error vector
    /// NOTE: Permutation and shuffle constraints are not verified here.
    pub fn decide<'params, P: Params<'params, C>>(
        params: &P,
        pk: &ProvingKey<C>,
        acc: &Self,
    ) -> bool {
        // Check all Committed columns are correct (commit(values;bline) == commitment)
        let committed_ok = {
            let committed_iter: Vec<&Committed<C>> = acc
                .gate
                .instance
                .iter()
                .chain(&acc.gate.advice)
                .chain([&acc.beta.beta, &acc.beta.error].into_iter())
                .chain(
                    acc.lookups
                        .iter()
                        .flat_map(|lookup| [&lookup.m, &lookup.g, &lookup.h].into_iter()),
                )
                .collect();
            committed_iter.iter().all(|c| c.decide(params))
        };

        // Check Error term  (error == ∑ᵢ βᵢ * Gᵢ)
        let error_ok = { acc.error == Self::error(&pk, &acc) };

        // Check linear lookup constraint ∑ᵢ gᵢ == ∑ᵢ hᵢ
        let lookups_ok = {
            acc.lookups.iter().all(|lookup| {
                let lhs: C::Scalar = lookup.g.values.iter().sum();
                let rhs: C::Scalar = lookup.h.values.iter().sum();
                lhs == rhs
            })
        };

        // Check beta constraint eᵢ ≡ β ⋅ βᵢ − βᵢ₊₁, β₀ ≡ 1
        let beta_ok = {
            let beta_column = &acc.beta.beta.values;
            let error_column = &acc.beta.error.values;

            let beta = beta_column[1];

            let powers_ok = (1..pk.num_rows)
                .into_iter()
                .all(|i| error_column[i - 1] == beta_column[i - 1] * beta - beta_column[i]);

            let init_ok = beta_column[0] == C::Scalar::ONE;
            powers_ok && init_ok
        };

        committed_ok && error_ok && lookups_ok && beta_ok
    }

    /// Recompute the compressed error term e = ∑ᵢ βᵢ * Gᵢ
    pub fn error(pk: &ProvingKey<C>, acc: &Self) -> C::Scalar {
        let lagrange_data = Data::<CommittedRef<'_, C>>::new(&pk, &acc);

        let full_constraint = lagrange_data.full_constraint(pk.cs.gates(), pk.cs.lookups());

        let mut error = pk.domain.empty_lagrange();
        parallelize(&mut error, |value, start| {
            for (i, v) in value.iter_mut().enumerate() {
                let row_idx = i + start;
                *v = full_constraint.evaluate(
                    &|&c| c,
                    &|&challenge| *challenge.value,
                    &|&fixed| fixed.column.values[fixed.row_idx(row_idx, pk.num_rows)],
                    &|&witness| witness.column.values[witness.row_idx(row_idx, pk.num_rows)],
                    &|&e| -e,
                    &|a, b| a + b,
                    &|a, b| a * b,
                );
            }
        });
        error.iter().sum()
    }
}

// Given a polynomial p(X) of degree d > 1, compute its quotient q(X)
// such that p(X) = (1-X)X⋅q(X).
// Panics if deg(p) ≤ 1 or if p(0) ≠ 0 or p(1) ≠ 0
fn quotient_by_boolean_vanishing<F: Field>(poly: &[F]) -> Vec<F> {
    let n = poly.len();
    assert!(n >= 2, "deg(poly) < 2");
    assert!(poly[0].is_zero_vartime(), "poly(0) != 0");

    let mut tmp = F::ZERO;

    let mut quotient = vec![F::ZERO; n - 2];
    for i in 0..(n - 2) {
        tmp += poly[i + 1];
        quotient[i] = tmp;
    }

    // p(1) = ∑p_i = 0
    assert_eq!(
        *quotient.last().unwrap(),
        poly.last().unwrap().neg(),
        "poly(1) != 0"
    );
    quotient
}

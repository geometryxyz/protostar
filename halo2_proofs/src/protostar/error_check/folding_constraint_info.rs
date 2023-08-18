use core::num;
use std::{
    collections::BTreeMap,
    iter::zip,
    ops::{Add, Deref, DerefMut, Index, IndexMut, RangeFrom, RangeFull, Sub},
};

use crate::{
    arithmetic::{field_integers, parallelize, powers},
    poly::{
        commitment::{Blind, CommitmentScheme, Params},
        empty_lagrange, LagrangeCoeff, Polynomial, Rotation,
    },
    protostar::error_check::gate::GateEvaluator,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::{BatchInvert, Field};
use group::Curve;
use halo2curves::CurveAffine;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use super::{
    super::{
        keygen::ProvingKey,
        transcript::{
            advice::AdviceTranscript, compressed_verifier::CompressedVerifierTranscript,
            instance::InstanceTranscript, lookup::LookupTranscipt,
        },
    },
    BETA_POLY_DEGREE,
};

pub(super) struct FoldingConstraintInfo<C: CurveAffine> {
    /// The number of folding constraints in each gate in the proving key.
    pub num_constraints_per_gate: Vec<usize>,
    /// Total number of folding constraints in the proving key.
    pub num_constraints_total: usize,
    /// The number of evaluations of each error polynomial in each gate.
    ///
    /// For each gate at index `gate_idx` with polynomials G₀, …, Gₘ₋₁ and degrees d₀, …, dₘ₋₁,
    /// we evaluate at each row i the polynomials e₀,ᵢ(X), …, eₘ₋₁,ᵢ(X), where
    ///     eⱼ,ᵢ(X) = βᵢ(X)⋅Gⱼ((1−X)⋅acc[i] + X⋅new[i]),
    /// where we use the shorthand notation `acc[i]` and `new[i]` to denote the row-vector of values from the
    /// accumulators `acc` and `new` respectively.
    ///
    /// Each eⱼ,ᵢ(X) has degree dⱼ + `BETA_POLY_DEGREE`, so we need (dⱼ + `BETA_POLY_DEGREE` + 1) evaluations in order
    /// to interpolate the full polynomial.
    pub gates_error_polys_num_evals: Vec<Vec<usize>>,
    /// Compute the maximum number of evaluations over all error polynomials over all gates.
    /// This defines the maximal evaluation domain D = {0, 1, …, dₘₐₓ + BETA_POLY_DEGREE + 1}
    /// This will allow us to compute the evaluations βᵢ(D)
    pub max_num_evals: usize,
    /// A `GateEvaluator` pre-processes a gate's polynomials for more efficient evaluation.
    /// In particular, it collects all column queries and allocates buffers where the corresponding
    /// values can be stored. Each polynomial in the gate is evaluated several times,
    /// so it is advantageous to only fetch the values from the columns once.
    pub gate_evaluators: Vec<GateEvaluator<C::ScalarExt>>,
}

impl<C: CurveAffine> FoldingConstraintInfo<C> {
    pub(super) fn new(
        pk: &ProvingKey<C>,
        advice_transcript_challenges_1: &[Vec<C::Scalar>],
        advice_transcript_challenges_2: &[Vec<C::Scalar>],
    ) -> Self {
        let constraints_iter = pk.folding_constraints.iter();

        let num_constraints_per_gate: Vec<usize> =
            constraints_iter.clone().map(|polys| polys.len()).collect();
        let num_constraints_total = num_constraints_per_gate.iter().sum();
        let gates_error_polys_num_evals: Vec<Vec<usize>> = constraints_iter
            .clone()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| poly.folding_degree() + BETA_POLY_DEGREE + 1)
                    .collect()
            })
            .collect();

        let max_num_evals = *gates_error_polys_num_evals
            .iter()
            .flat_map(|gates_num_evals| gates_num_evals.iter())
            .max()
            .unwrap();

        let gate_evaluators = constraints_iter
            .zip(gates_error_polys_num_evals.iter())
            .map(|(polys, num_evals)| {
                let max_num_evals = num_evals.iter().max().unwrap();
                GateEvaluator::new(
                    polys,
                    &advice_transcript_challenges_1,
                    &advice_transcript_challenges_2,
                    *max_num_evals,
                )
            })
            .collect();

        Self {
            num_constraints_per_gate,
            num_constraints_total,
            gates_error_polys_num_evals,
            max_num_evals,
            gate_evaluators,
        }
    }
}

/// For each gate, we allocate a buffer for storing running sum of the evaluation e₀(X), …, eₘ₋₁(X).
/// The polynomial eⱼ(X) is given by
///     eⱼ(X) = ∑ᵢ eⱼ,ᵢ(X)
///
/// For each polynomial eⱼ(X), we define its evaluation domain as Dⱼ = {0, 1, …, dⱼ + BETA_POLY_DEGREE + 1},
/// and we denote by eⱼ(Dⱼ) = { p(i) | i ∈ Dⱼ } the list of evaluations of eⱼ(X) over Dⱼ.
///
/// Therefore, for each gate, `gates_error_polys_evals[gate_idx]` corresponds to the list
///     [e₀(D₀), …, eₘ₋₁(Dₘ₋₁)]
///
/// As an optimization, we observe that we can store in the accumulators the evaluations
///     [e₀(0), …, eₘ₋₁(0)] in `acc`, and
///     [e₀(1), …, eₘ₋₁(1)] in `new`
/// Therefore, we only actually evaluate the polynomials on the restricted sets Dⱼ' = Dⱼ \ {0,1}.
///
/// Separate from `FoldingConstraintInfo` because mutability.
pub(super) type GatesPolysEvaluations<C> = Vec<Vec<Vec<<C as CurveAffine>::ScalarExt>>>;
pub(super) struct GatesErrorPolyEvaluations<C: CurveAffine> {
    pub evals: GatesPolysEvaluations<C>,
    /// A clone of `evals` such that we may access the original values.
    /// For gate at index `gate_idx` and each row i of the accumulator, we temporarily store the list of evaluations
    ///     [e₀,ᵢ(D₀), …, eₘ₋₁,ᵢ(Dₘ₋₁)]
    /// in `row_gates_error_polys_evals[gate_idx]`
    // todo(tk): why do we need this?
    pub row_evals: GatesPolysEvaluations<C>,
}

impl<C: CurveAffine> GatesErrorPolyEvaluations<C> {
    pub(super) fn new(evals: &[Vec<usize>]) -> Self {
        let evals: Vec<_> = evals
            .iter()
            // .cloned()
            .map(|error_polys_num_evals| {
                error_polys_num_evals
                    .iter()
                    .map(|num_evals| vec![C::Scalar::ZERO; *num_evals])
                    .collect()
            })
            .collect();

        Self {
            row_evals: evals.clone(),
            evals,
        }
    }
}

use std::iter::zip;

use ff::{BatchInvert, Field};
use rand_core::RngCore;

use crate::arithmetic::field_integers;

use super::evaluated_poly::EvaluatedFrom2;

/// Inverse of a Vandermonde matrix. When multiplied by a list of evaluations,
/// the result is the list of coefficients of that polynomial.
struct InverseVandermodeMatrix<F: Field> {
    cols: Vec<Vec<F>>,
}

fn compute_denoms<F: Field>(points: &[F]) -> Vec<Vec<F>> {
    let mut denoms = Vec::with_capacity(points.len());
    for (j, x_j) in points.iter().enumerate() {
        let mut denom = Vec::with_capacity(points.len() - 1);
        for x_k in points
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != j)
            .map(|a| a.1)
        {
            denom.push(*x_j - x_k);
        }
        denoms.push(denom);
    }
    // Compute (x_j - x_k)^(-1) for each j != i
    denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();
    denoms
}

impl<F: Field> InverseVandermodeMatrix<F> {
    fn new(points: &[F], denoms: &[Vec<F>]) -> Self {
        let n = points.len();
        assert_ne!(n, 0);
        if n == 1 {
            return Self {
                cols: vec![vec![F::ONE]],
            };
        }
        assert!(n <= denoms.len());

        let mut cols = Vec::with_capacity(n);

        let mut tmp: Vec<F> = Vec::with_capacity(n);
        let mut product = Vec::with_capacity(n - 1);
        for (j, denoms) in denoms.iter().enumerate().take(n) {
            tmp.clear();
            product.clear();
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.iter())
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-*denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), n);
            assert_eq!(product.len(), n - 1);

            cols.push(tmp.clone());
        }

        Self { cols }
    }

    fn multiply_evaluations(&self, evals: &[F]) -> Vec<F> {
        let n = evals.len();
        assert_eq!(n, self.cols.len());
        let mut final_poly = vec![F::ZERO; n];

        for (col, eval) in self.cols.iter().zip(evals.iter()) {
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(col.iter()) {
                *final_coeff += *eval * interpolation_coeff;
            }
        }

        final_poly
    }
}

/// For a list of points [x₁, …, xₙ], this struct contains the inverse Vandermonde matrices
///   [M₁, …, Mₙ], such that given a list of k evaluations [p(x₁), …, p(xₖ)] for 1 ≤ k ≤ n,
/// and p(X) = p₀ + p₁⋅X + ⋯ pₖ₋₁⋅Xᵏ⁻¹
/// Mₖ⋅[p(x₁), …, p(xₖ)] = [p₀, p₁, …, pₖ₋₁]
pub struct LagrangeInterpolater<F: Field> {
    inverse_vandermonde_matrices: Vec<InverseVandermodeMatrix<F>>,
}

impl<F: Field> LagrangeInterpolater<F> {
    // Compute the matrices for [x₁, …, xₙ] = [0, 1, …, max_num_evals-1]
    pub fn new_integer_eval_domain(max_num_evals: usize) -> Self {
        let points: Vec<F> = field_integers().take(max_num_evals).collect();
        Self::new(&points)
    }

    pub fn new(points: &[F]) -> Self {
        let n = points.len();
        let mut matrices = Vec::with_capacity(n);

        let denoms = compute_denoms(&points);

        for n in 0..n {
            matrices.push(InverseVandermodeMatrix::new(
                &points[0..n + 1],
                &denoms[0..n + 1],
            ));
        }

        Self {
            inverse_vandermonde_matrices: matrices,
        }
    }

    pub fn interpolate(&self, evals: &[F]) -> Vec<F> {
        let n = evals.len();
        assert_ne!(n, 0);
        assert!(n <= self.inverse_vandermonde_matrices.len());
        self.inverse_vandermonde_matrices[n - 1].multiply_evaluations(evals)
    }
}

#[cfg(test)]
use rand_core::OsRng;

#[cfg(test)]
use crate::{arithmetic::eval_polynomial, halo2curves::pasta::Fp};

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;
    const MAX_NUM_EVALS: usize = 16;

    let points: Vec<Fp> = field_integers().take(MAX_NUM_EVALS).collect();

    let interpolater = LagrangeInterpolater::<Fp>::new_integer_eval_domain(MAX_NUM_EVALS);

    for num_evals in 1..MAX_NUM_EVALS {
        let evals: Vec<Fp> = (0..num_evals).map(|_| Fp::random(rng)).collect();

        let poly = interpolater.interpolate(&evals);

        assert_eq!(poly.len(), num_evals);

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), eval);
        }
    }
}

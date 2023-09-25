use std::{
    iter::{self, zip},
    ops::Range,
};

use crate::{
    arithmetic::{eval_polynomial, parallelize, powers},
    plonk::{
        evaluation::evaluate, permutation, shuffle, vanishing, Any, ChallengeBeta, ChallengeGamma,
        ChallengeTheta, ChallengeX, ConstraintSystem, Error, Expression,
    },
    poly::{
        commitment::{Blind, CommitmentScheme, Params, ParamsProver, Prover},
        Basis, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, ProverQuery,
        Rotation,
    },
    transcript::{ChallengeScalar, EncodedChallenge, TranscriptWrite},
};
use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::Curve;
use halo2curves::CurveAffine;
use rand_core::RngCore;

use super::{accumulator::Accumulator, ProvingKey};

pub fn create_proof<
    'params,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore,
    T: TranscriptWrite<Scheme::Curve, E>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    acc: Accumulator<Scheme::Curve>,
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
{
    let domain = &pk.domain;
    let cs = &pk.cs;
    let blinding_factors = cs.blinding_factors();
    let challenges = acc.advice.challenges;

    let first_row = 0;
    let num_rows = params.n() as usize;
    let active_rows = Range {
        start: first_row,
        end: num_rows - (blinding_factors + 1),
    };
    let last_row = active_rows.end;

    // Permutation and Shuffle constraints

    // Sample theta challenge for keeping shuffle columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    // Sample beta challenge for permutation
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge for permutation
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let advice: Vec<_> = acc
        .advice
        .committed
        .iter()
        .map(|c| c.values.clone())
        .collect();

    let instance: Vec<_> = acc
        .instance
        .committed
        .iter()
        .map(|c| c.values.clone())
        .collect();

    // Commit to permutations.
    let permutations: permutation::prover::Committed<Scheme::Curve> = pk.cs.permutation.commit(
        params,
        domain,
        &pk.cs,
        &pk.permutation_pk,
        &advice,
        &pk.fixed_polys,
        &instance,
        beta,
        gamma,
        &mut rng,
        transcript,
    )?;

    let shuffles: Vec<shuffle::prover::Committed<Scheme::Curve>> = pk
        .cs
        .shuffles
        .iter()
        .map(|shuffle| {
            shuffle.commit_product(
                params,
                domain,
                blinding_factors,
                theta,
                gamma,
                &advice,
                &pk.fixed_polys,
                &pk.selectors_polys,
                &instance,
                &challenges,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Commit to error vector

    let folding_error = Accumulator::error_vector(&pk, &acc);
    let folding_error_commtiment = params
        .commit_lagrange(&folding_error, Blind::default())
        .to_affine();
    let _ = transcript.write_point(folding_error_commtiment);

    // Get random challenge y

    let y_sumcheck = *transcript.squeeze_challenge_scalar::<Scheme::Scalar>();

    // Compute Sumcheck polynomial

    let sumcheck = {
        let mut sumcheck = domain.empty_lagrange();

        // Add contribution from folding constraints multiplied by beta
        parallelize(&mut sumcheck, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = i + start;
                *value = acc.beta.beta.values[idx] * folding_error[idx];
            }
        });

        // Add contribution of lookup sumcheck constraints
        for lookup in &acc.lookups {
            parallelize(&mut sumcheck, |values, start| {
                let g = &lookup.g.values;
                let h = &lookup.h.values;
                for (i, value) in values.iter_mut().enumerate() {
                    let idx = i + start;

                    *value *= y_sumcheck;
                    *value += g[idx] - h[idx];
                }
            });
        }

        // Add error sum contribution
        let mut partial_sum = Scheme::Scalar::ZERO;
        for i in 0..num_rows {
            let prev_sum = partial_sum;
            partial_sum += sumcheck[i];
            sumcheck[i] = prev_sum;
        }

        assert_eq!(sumcheck[last_row], acc.error);
        sumcheck
    };

    // Commit to sumcheck vector

    let sumcheck_commtiment = params
        .commit_lagrange(&sumcheck, Blind::default())
        .to_affine();
    let _ = transcript.write_point(sumcheck_commtiment);

    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;

    // Get random challenge y

    let y = *transcript.squeeze_challenge_scalar::<Scheme::Scalar>();

    // Coset stuff

    let selector_polys: Vec<_> = pk
        .selectors_polys
        .iter()
        .map(|poly| domain.lagrange_to_coeff(poly.clone()))
        .collect();

    let selector_cosets: Vec<_> = selector_polys
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let fixed_polys: Vec<_> = pk
        .fixed_polys
        .iter()
        .map(|poly| domain.lagrange_to_coeff(poly.clone()))
        .collect();

    let fixed_cosets: Vec<_> = fixed_polys
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let advice_polys: Vec<_> = advice
        .iter()
        .map(|poly| domain.lagrange_to_coeff(poly.clone()))
        .collect();

    let advice_cosets: Vec<_> = advice_polys
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let instance_polys: Vec<_> = instance
        .iter()
        .map(|poly| domain.lagrange_to_coeff(poly.clone()))
        .collect();

    let instance_cosets: Vec<_> = instance_polys
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let lookup_m_poly: Vec<_> = acc
        .lookups
        .iter()
        .map(|lookup| domain.lagrange_to_coeff(lookup.m.values.clone()))
        .collect();
    let lookup_m_cosets: Vec<_> = lookup_m_poly
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let lookup_g_poly: Vec<_> = acc
        .lookups
        .iter()
        .map(|lookup| domain.lagrange_to_coeff(lookup.g.values.clone()))
        .collect();
    let lookup_g_cosets: Vec<_> = lookup_g_poly
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let lookup_h_poly: Vec<_> = acc
        .lookups
        .iter()
        .map(|lookup| domain.lagrange_to_coeff(lookup.h.values.clone()))
        .collect();
    let lookup_h_cosets: Vec<_> = lookup_h_poly
        .iter()
        .map(|poly| domain.coeff_to_extended(poly.clone()))
        .collect();

    let folding_error_poly = domain.lagrange_to_coeff(folding_error);
    let folding_error_coset = domain.coeff_to_extended(folding_error_poly.clone());

    let sumcheck_poly = domain.lagrange_to_coeff(sumcheck);
    let sumcheck_coset = domain.coeff_to_extended(sumcheck_poly.clone());

    let beta_poly = domain.lagrange_to_coeff(acc.beta.beta.values.clone());
    let beta_coset = domain.coeff_to_extended(beta_poly.clone());

    // Compute l_0(X)
    let mut l0 = domain.empty_lagrange();
    l0[0] = Scheme::Scalar::ONE;
    let l0 = domain.lagrange_to_coeff(l0);
    let l0 = domain.coeff_to_extended(l0);

    // Compute l_blind(X) which evaluates to 1 for each blinding factor row
    // and 0 otherwise over the domain.
    let mut l_blind = domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(blinding_factors) {
        *evaluation = Scheme::Scalar::ONE;
    }
    let l_blind = domain.lagrange_to_coeff(l_blind);
    let l_blind = domain.coeff_to_extended(l_blind);

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain
    let mut l_last = domain.empty_lagrange();
    l_last[params.n() as usize - blinding_factors - 1] = Scheme::Scalar::ONE;
    let l_last = domain.lagrange_to_coeff(l_last);
    let l_last = domain.coeff_to_extended(l_last);

    // Compute l_active_row(X)
    let one = Scheme::Scalar::ONE;
    let mut l_active_row = domain.empty_extended();
    parallelize(&mut l_active_row, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last[idx] + l_blind[idx]);
        }
    });

    // We start by evaluating e = ∑yj*gj(a)
    // where gj are all the folding constraints (including lookups)
    //
    // the inclusion of e is to prevent the quotient degree from growing
    //
    // compute s such that s0 = acc.e, and si' = si + bi*ei + ∑lambda^j (h^j_i - g^j_i)
    // which includes the linear sumcheck constraints using a challenge lambda
    //  - error_vector
    //  - permutations
    //  - shuffles
    //  - sumcheck [ si + bi*ei + ∑l^j (h^j_i - g^j_i) - si' ] + l^{j+L}[ ∑j G'(a)] + Z'∏(b+wi) - Z∏(b+wi)

    let challenges = &acc.advice.challenges;

    let quotient_coset = evaluate_quotient(
        pk,
        &acc,
        l0,
        l_last,
        l_active_row,
        selector_cosets,
        fixed_cosets,
        advice_cosets,
        instance_cosets,
        lookup_m_cosets,
        lookup_g_cosets,
        lookup_h_cosets,
        beta_coset,
        folding_error_coset,
        sumcheck_coset,
        &permutations,
        &shuffles,
        challenges,
        theta,
        beta,
        gamma,
        y_sumcheck,
        y,
    );

    // Construct the vanishing argument's h(X) commitments
    let vanishing = vanishing.construct(params, domain, quotient_coset, &mut rng, transcript)?;

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow(&[params.n() as u64, 0, 0, 0]);
    let x_next = domain.rotate_omega(*x, Rotation::next());
    let x_last = domain.rotate_omega(*x, Rotation(-((blinding_factors + 1) as i32)));

    // if P::QUERY_INSTANCE {
    //     // Compute and hash instance evals for each circuit instance
    //     for instance in instance.iter() {
    //         // Evaluate polynomials at omega^i x
    //         let instance_evals: Vec<_> = meta
    //             .instance_queries
    //             .iter()
    //             .map(|&(column, at)| {
    //                 eval_polynomial(
    //                     &instance.instance_polys[column.index()],
    //                     domain.rotate_omega(*x, at),
    //                 )
    //             })
    //             .collect();

    //         // Hash each instance column evaluation
    //         for eval in instance_evals.iter() {
    //             transcript.write_scalar(*eval)?;
    //         }
    //     }
    // }

    // Compute and hash advice evals
    // Evaluate polynomials at omega^i x
    {
        let advice_evals: Vec<_> = cs
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(&advice_polys[column.index()], domain.rotate_omega(*x, at))
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    };

    // Compute and hash fixed evals (shared across all circuit instances)
    {
        let fixed_evals: Vec<_> = cs
            .fixed_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
            })
            .collect();

        // Hash each fixed column evaluation
        for eval in fixed_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    };
    // Compute and hash selector evals (shared across all circuit instances)
    {
        let selector_evals: Vec<_> = (0..cs.num_selectors)
            .into_iter()
            .map(|selector_index| {
                eval_polynomial(
                    &pk.selectors_polys[selector_index],
                    domain.rotate_omega(*x, Rotation::cur()),
                )
            })
            .collect();

        // Hash each fixed column evaluation
        for eval in selector_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    };

    // Compute and hash lookup evals (shared across all circuit instances)
    {
        let lookup_evals: Vec<_> = acc
            .lookups
            .iter()
            .flat_map(|lookup| [&lookup.m.values, &lookup.g.values, &lookup.h.values].into_iter())
            .map(|poly| eval_polynomial(poly, domain.rotate_omega(*x, Rotation::cur())))
            .collect();

        // Hash each fixed column evaluation
        for eval in lookup_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    }

    // Compute and hash sumcheck, error and beta evals
    {
        let sumcheck_eval_curr =
            eval_polynomial(&sumcheck_poly, domain.rotate_omega(*x, Rotation::cur()));
        let sumcheck_eval_next =
            eval_polynomial(&sumcheck_poly, domain.rotate_omega(*x, Rotation::next()));

        let folding_error_eval = eval_polynomial(
            &folding_error_poly,
            domain.rotate_omega(*x, Rotation::cur()),
        );

        let beta_eval = eval_polynomial(&beta_poly, domain.rotate_omega(*x, Rotation::cur()));

        transcript.write_scalar(sumcheck_eval_curr)?;
        transcript.write_scalar(sumcheck_eval_next)?;
        transcript.write_scalar(folding_error_eval)?;
        transcript.write_scalar(beta_eval)?;
    }

    let vanishing_evals = vanishing.evaluate(x, xn, domain, transcript)?;

    // Evaluate common permutation data
    pk.permutation_pk.evaluate(x, transcript)?;

    // Evaluate the permutations, if any, at omega^i x.
    let permutations_evals =
        permutations
            .construct()
            .evaluate(domain, blinding_factors, x, transcript)?;

    // Evaluate the shuffles, if any, at omega^i x.
    let shuffles_evals: Vec<shuffle::prover::Evaluated<Scheme::Curve>> = shuffles
        .into_iter()
        .map(|p| p.evaluate(domain, x, transcript))
        .collect::<Result<Vec<_>, _>>()?;

    let queries = iter::empty()
        // .chain(
        //     P::QUERY_INSTANCE
        //         .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| {
        //             ProverQuery {
        //                 point: domain.rotate_omega(*x, at),
        //                 poly: &instance.instance_polys[column.index()],
        //                 blind: Blind::default(),
        //             }
        //         }))
        //         .into_iter()
        //         .flatten(),
        // )
        .chain(cs.advice_queries.iter().map(|&(column, at)| ProverQuery {
            point: domain.rotate_omega(*x, at),
            poly: &advice_polys[column.index()],
            blind: acc.advice.committed[column.index()].blind,
        }))
        .chain(permutations_evals.open(x, x_next, x_last))
        .chain(
            shuffles_evals
                .iter()
                .flat_map(move |p| p.open(domain, x))
                .into_iter(),
        )
        .chain(
            [
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::cur()),
                    poly: &sumcheck_poly,
                    blind: Blind::default(),
                },
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::next()),
                    poly: &sumcheck_poly,
                    blind: Blind::default(),
                },
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::cur()),
                    poly: &folding_error_poly,
                    blind: Blind::default(),
                },
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::cur()),
                    poly: &beta_poly,
                    blind: Blind::default(),
                },
            ]
            .into_iter(),
        )
        .chain(acc.lookups.iter().enumerate().flat_map(|(i, lookup)| {
            [
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::cur()),
                    poly: &lookup_m_poly[i],
                    blind: lookup.m.blind,
                },
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::cur()),
                    poly: &lookup_g_poly[i],
                    blind: lookup.g.blind,
                },
                ProverQuery {
                    point: domain.rotate_omega(*x, Rotation::cur()),
                    poly: &lookup_h_poly[i],
                    blind: lookup.h.blind,
                },
            ]
            .into_iter()
        }))
        .chain(cs.fixed_queries.iter().map(|&(column, at)| ProverQuery {
            point: domain.rotate_omega(*x, at),
            poly: &fixed_polys[column.index()],
            blind: Blind::default(),
        }))
        .chain(
            (0..cs.num_selectors)
                .into_iter()
                .map(|selector_index| ProverQuery {
                    point: *x,
                    poly: &selector_polys[selector_index],
                    blind: Blind::default(),
                }),
        )
        .chain(pk.permutation_pk.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing_evals.open(x));

    let prover = P::new(params);
    prover
        .create_proof(rng, transcript, queries)
        .map_err(|_| Error::ConstraintSystemFailure)
}

fn evaluate_quotient<C: CurveAffine>(
    pk: &ProvingKey<C>,
    acc: &Accumulator<C>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_active_row: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    selector_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    advice_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    instance_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    lookup_m_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    lookup_g_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    lookup_h_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    beta_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    folding_error_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    sumcheck_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    permutations: &permutation::prover::Committed<C>,
    shuffles: &[shuffle::prover::Committed<C>],
    challenges: &[C::Scalar],
    theta: ChallengeScalar<C, crate::plonk::Theta>,
    beta: ChallengeScalar<C, crate::plonk::Beta>,
    gamma: ChallengeScalar<C, crate::plonk::Gamma>,
    y_sumcheck: C::Scalar,
    y: C::Scalar,
) -> Polynomial<C::Scalar, ExtendedLagrangeCoeff> {
    let domain = &pk.domain;
    let cs = &pk.cs;
    let size = domain.extended_len();
    let rot_scale = 1 << (domain.extended_k() - domain.k());
    let isize = size as i32;

    let blinding_factors = cs.blinding_factors();
    let cs_degree = pk.cs.degree();

    let extended_omega = domain.get_extended_omega();
    let one = C::Scalar::ONE;

    // h = (∑yj*gj_i - e_i)
    let mut values = domain.extended_from_vec(evaluate_error(
        cs,
        size,
        rot_scale,
        &selector_cosets,
        &fixed_cosets,
        &advice_cosets,
        &instance_cosets,
        &challenges,
        &acc.ys,
        &acc.lookup_transcript.r,
        &acc.lookup_transcript.thetas,
        &lookup_m_cosets,
        &lookup_g_cosets,
        &lookup_h_cosets,
    )) - &folding_error_coset;

    // Permutations
    let sets = &permutations.sets;
    if !sets.is_empty() {
        let last_rotation = Rotation(-((blinding_factors + 1) as i32));
        let chunk_len = cs_degree - 2;
        let delta_start = *beta * &C::Scalar::ZETA;

        let first_set = sets.first().unwrap();
        let last_set = sets.last().unwrap();

        // Permutation constraints
        parallelize(&mut values, |values, start| {
            let mut beta_term = extended_omega.pow_vartime(&[start as u64, 0, 0, 0]);
            for (i, value) in values.iter_mut().enumerate() {
                let idx = start + i;
                let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                // Enforce only for the first set.
                // l_0(X) * (1 - z_0(X)) = 0
                *value = *value * y + ((one - first_set.permutation_product_coset[idx]) * l0[idx]);
                // Enforce only for the last set.
                // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                *value = *value * y
                    + ((last_set.permutation_product_coset[idx]
                        * last_set.permutation_product_coset[idx]
                        - last_set.permutation_product_coset[idx])
                        * l_last[idx]);
                // Except for the first set, enforce.
                // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                for (set_idx, set) in sets.iter().enumerate() {
                    if set_idx != 0 {
                        *value = *value * y
                            + ((set.permutation_product_coset[idx]
                                - permutations.sets[set_idx - 1].permutation_product_coset
                                    [r_last])
                                * l0[idx]);
                    }
                }
                // And for all the sets we enforce:
                // (1 - (l_last(X) + l_blind(X))) * (
                //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                // )
                let mut current_delta = delta_start * beta_term;
                for ((set, columns), cosets) in sets
                    .iter()
                    .zip(cs.permutation.columns.chunks(chunk_len))
                    .zip(pk.permutation_pk.cosets.chunks(chunk_len))
                {
                    let mut left = set.permutation_product_coset[r_next];
                    for (values, permutation) in columns
                        .iter()
                        .map(|&column| match column.column_type() {
                            Any::Advice(_) => &advice_cosets[column.index()],
                            Any::Fixed => &fixed_cosets[column.index()],
                            Any::Instance => &instance_cosets[column.index()],
                        })
                        .zip(cosets.iter())
                    {
                        left *= values[idx] + *beta * permutation[idx] + *gamma;
                    }

                    let mut right = set.permutation_product_coset[idx];
                    for values in columns.iter().map(|&column| match column.column_type() {
                        Any::Advice(_) => &advice_cosets[column.index()],
                        Any::Fixed => &fixed_cosets[column.index()],
                        Any::Instance => &instance_cosets[column.index()],
                    }) {
                        right *= values[idx] + current_delta + *gamma;
                        current_delta *= &C::Scalar::DELTA;
                    }

                    *value = *value * y + ((left - right) * l_active_row[idx]);
                }
                beta_term *= &extended_omega;
            }
        });
    }

    // Shuffle constraints
    for (arg, committed) in zip(cs.shuffles.iter(), shuffles.iter()) {
        let product_coset = domain.coeff_to_extended(committed.product_poly.clone());

        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    domain.extended_from_vec(evaluate(
                        expression,
                        size,
                        1,
                        &selector_cosets,
                        &fixed_cosets,
                        &advice_cosets,
                        &instance_cosets,
                        challenges,
                    ))
                })
                .fold(domain.empty_extended(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        // Get values of input expressions involved in the shuffle and compress them
        let input_expression = compress_expressions(&arg.input_expressions);

        // Get values of table expressions involved in the shuffle and compress them
        let shuffle_expression = compress_expressions(&arg.shuffle_expressions);

        // Shuffle constraints
        parallelize(&mut values, |values, start| {
            for (i, value) in values.iter_mut().enumerate() {
                let idx = start + i;

                let input_value = input_expression[idx];

                let shuffle_value = shuffle_expression[idx];

                let r_next = get_rotation_idx(idx, 1, rot_scale, isize);

                // l_0(X) * (1 - z(X)) = 0
                *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                // l_last(X) * (z(X)^2 - z(X)) = 0
                *value = *value * y
                    + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                        * l_last[idx]);
                // (1 - (l_last(X) + l_blind(X))) * (z(\omega X) (s(X) + \gamma) - z(X) (a(X) + \gamma)) = 0
                *value = *value * y
                    + l_active_row[idx]
                        * (product_coset[r_next] * shuffle_value - product_coset[idx] * input_value)
            }
        });
    }

    // sumcheck constraints
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);

            // Enforce only for the first set.
            // l_0(X) * (1 - t(X)) = 0
            *value = *value * y + ((one - sumcheck_coset[idx]) * l0[idx]);
            // Enforce only for the last set.
            // l_last(X) * (t(X) - error) = 0
            *value = *value * y + ((sumcheck_coset[idx] - acc.error) * l_last[idx]);

            let t_next = sumcheck_coset[r_next];
            let t_curr = sumcheck_coset[idx];
            let mut t_diff = folding_error_coset[idx] * beta_coset[idx];

            for (g_coset, h_coset) in zip(lookup_g_cosets.iter(), lookup_h_cosets.iter()) {
                t_diff = t_diff * y_sumcheck + &g_coset[idx] - h_coset[idx];
            }

            *value = *value * y + ((t_next - t_curr - t_diff) * l_active_row[idx]);
        }
    });
    values
}

pub fn add_evaluated_lookup_input<F: Field, B: Basis>(
    values: &mut [F],
    scaling_factor: &F,
    input_expressions: &[Expression<F>],
    rot_scale: i32,
    selector: &[Polynomial<F, B>],
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
    g: &Polynomial<F, B>,
    thetas: &[F],
    r: &F,
) {
    let size = values.len();
    let isize = size as i32;
    parallelize(values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            let mut v = F::ZERO;

            for (expr, theta) in input_expressions.iter().zip(thetas.iter()) {
                v += expr.evaluate(
                    &|scalar| scalar,
                    &|query| selector[query.index()][idx],
                    &|query| {
                        fixed[query.column_index]
                            [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                    },
                    &|query| {
                        advice[query.column_index]
                            [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                    },
                    &|query| {
                        instance[query.column_index]
                            [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                    },
                    &|challenge| challenges[challenge.index()],
                    &|a| -a,
                    &|a, b| a + &b,
                    &|a, b| a * b,
                    &|a, scalar| a * scalar,
                ) * theta;
            }
            v += r;
            v *= g[i];
            v -= F::ONE;
            *value += v * scaling_factor;
        }
    });
}

pub fn add_evaluated_lookup_table<F: Field, B: Basis>(
    values: &mut [F],
    scaling_factor: &F,
    table_expressions: &[Expression<F>],
    rot_scale: i32,
    selector: &[Polynomial<F, B>],
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
    m: &Polynomial<F, B>,
    h: &Polynomial<F, B>,
    thetas: &[F],
    r: &F,
) {
    let size = values.len();
    let isize = size as i32;
    parallelize(values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            let mut v = F::ZERO;

            for (expr, theta) in table_expressions.iter().zip(thetas.iter()) {
                v += expr.evaluate(
                    &|scalar| scalar,
                    &|query| selector[query.index()][idx],
                    &|query| {
                        fixed[query.column_index]
                            [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                    },
                    &|query| {
                        advice[query.column_index]
                            [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                    },
                    &|query| {
                        instance[query.column_index]
                            [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                    },
                    &|challenge| challenges[challenge.index()],
                    &|a| -a,
                    &|a, b| a + &b,
                    &|a, b| a * b,
                    &|a, scalar| a * scalar,
                ) * theta;
            }
            v += r;
            v *= h[i];
            v -= m[i];
            *value += v * scaling_factor;
        }
    });
}

pub fn add_evaluated_expression<F: Field, B: Basis>(
    values: &mut [F],
    scaling_factor: &F,
    expr: &Expression<F>,
    rot_scale: i32,
    selector: &[Polynomial<F, B>],
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) {
    let size = values.len();
    let isize = size as i32;
    parallelize(values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;

            *value += expr.evaluate(
                &|scalar| scalar,
                &|query| selector[query.index()][idx],
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            ) * scaling_factor;
        }
    });
}

pub fn evaluate_error<F: Field, B: Basis>(
    cs: &ConstraintSystem<F>,
    size: usize,
    rot_scale: i32,
    selector: &[Polynomial<F, B>],
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
    ys: &[F],
    lookup_r: &Option<F>,
    lookup_thetas: &Option<Vec<F>>,
    lookup_m: &[Polynomial<F, B>],
    lookup_g: &[Polynomial<F, B>],
    lookup_h: &[Polynomial<F, B>],
) -> Vec<F> {
    let mut ys_iter = ys.iter();
    // Store βᵢ⋅eᵢ for each row i
    let mut errors = vec![F::ZERO; size];

    for (poly, y) in cs
        .gates
        .iter()
        .flat_map(|gate| gate.polynomials())
        .zip(ys_iter.by_ref())
    {
        // Add ∑ⱼ yⱼ⋅Gⱼ(acc[i]) to eᵢ
        add_evaluated_expression(
            &mut errors,
            y,
            poly,
            rot_scale,
            selector,
            fixed,
            advice,
            instance,
            challenges,
        );
    }

    for (j, lookup) in cs.lookups.iter().enumerate() {
        let thetas = lookup_thetas.as_ref().unwrap();
        let r = lookup_r.as_ref().unwrap();
        let m = &lookup_m[j];
        let g = &lookup_g[j];
        let h = &lookup_h[j];

        {
            let y_curr = *ys_iter.next().unwrap();
            add_evaluated_lookup_input(
                &mut errors,
                &y_curr,
                &lookup.input_expressions,
                rot_scale,
                selector,
                fixed,
                advice,
                instance,
                challenges,
                g,
                thetas,
                r,
            );
        }

        {
            let y_curr = *ys_iter.next().unwrap();
            add_evaluated_lookup_table(
                &mut errors,
                &y_curr,
                &lookup.table_expressions,
                rot_scale,
                selector,
                fixed,
                advice,
                instance,
                challenges,
                m,
                h,
                thetas,
                r,
            );
        }
    }

    errors
}

/// Return the index in the polynomial of size `isize` after rotation `rot`.
pub fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

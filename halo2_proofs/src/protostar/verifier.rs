use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use halo2curves::{CurveAffine, CurveExt};
use rand_core::RngCore;
use std::{collections::HashMap, iter::zip};

use super::transcript::{
    advice::AdviceTranscript,
    compressed_verifier::CompressedVerifierTranscript,
    instance::InstanceTranscript,
    lookup::{LookupTranscipt, LookupTranscriptSingle},
};
use super::{error_check::Accumulator, keygen::ProvingKey, row_evaluator::RowEvaluator};
use crate::arithmetic::{best_multiexp, compute_inner_product, parallelize, powers};
use crate::plonk::Error;
use crate::poly::commitment::{CommitmentScheme, Verifier};
use crate::poly::VerificationStrategy;
use crate::poly::{
    commitment::{Blind, Params, MSM},
    Guard, VerifierQuery,
};
use crate::transcript::{
    read_n_points, read_n_scalars, EncodedChallenge, TranscriptRead, TranscriptWrite,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

#[derive(Debug, Clone, PartialEq)]
// TODO(@gnosed): check the types for TranscriptRead, it doesn't seem to be correct
pub struct VerifierAccumulator<C: CurveAffine> {
    pub instance_commitments: Vec<C::Scalar>,
    pub challenges: Vec<Vec<C::Scalar>>,
    pub advice_commitments: Vec<C>,
    // thetas: Option<Vec<C::Scalar>>,
    // r: Option<C::Scalar>,
    // singles_transcript: Vec<LookupTranscriptSingle<C>>,
    pub m_commitments: Vec<C>,
    pub g_commitments: Vec<C>,
    pub h_commitments: Vec<C>,
    pub beta_commitment: C,
    pub e_commitments: Vec<C>,
    pub y: C::Scalar,
    pub alpha: C::Scalar,
}

impl<C: CurveAffine> VerifierAccumulator<C> {
    /// Create a new `VerifierAccumulator` by reading the IOP transcripts from the Prover and save commitments and challenges
    pub fn new_from_prover<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        transcript: &mut T,
        instances: &[&[&[C::Scalar]]],
        // TODO(@gnosed): replace pk with vk: VerifiyingKey<C>
        pk: &ProvingKey<C>,
        acc: &Accumulator<C>,
    ) -> Result<Self, Error> {
        //
        // Get instance commitments
        //
        // Check that instances matches the expected number of instance columns
        for instances in instances.iter() {
            if instances.len() != pk.cs.num_instance_columns {
                return Err(Error::InvalidInstances);
            }
        }

        // TODO(@gnosed): DOUBT, is that correct? can we read a point from transcript. if it was saved with common_point() interface
        // let instance_commitments: Vec<Vec<C>> = vec![vec![]; instances.len()];

        // let instance_commitments = vec![vec![]; instances.len()];

        let mut instance_commitments = vec![C::Scalar::ZERO; instances.len()];

        for instance_commitment in instance_commitments.iter_mut() {
            *instance_commitment = transcript.read_scalar()?;
        }

        assert_eq!(
            acc.instance_transcript.verifier_instance_commitments,
            instance_commitments
        );

        // Hash verification key into transcript
        // TODO(@gnosed): is it necessary? If yes, change it when the VerifyingKey was implemented
        // vk.hash_into(transcript)?;

        // for instance in instances.iter() {
        //     for instance in instance.iter() {
        //         for value in instance.iter() {
        //             transcript.common_scalar(*value)?;
        //         }
        //     }
        // }

        //
        // Get advice commitments and challenges
        //
        // Hash the prover's advice commitments into the transcript and squeeze challenges
        let mut advice_commitments = vec![C::identity(); pk.cs.num_advice_columns];

        for advice_commitment in advice_commitments.iter_mut() {
            *advice_commitment = transcript.read_point()?;
        }

        assert_eq!(acc.advice_transcript.advice_commitments, advice_commitments);

        let challenge = *transcript.squeeze_challenge_scalar::<C::Scalar>();

        let challenge_degrees = pk.max_challenge_powers();
        // Array of challenges and their powers
        // challenges[i][d] = challenge_i^{d+1}
        let challenges = vec![C::Scalar::ZERO; pk.cs.num_challenges]
            .into_iter()
            .zip(challenge_degrees)
            .map(|(c, d)| powers(c).skip(1).take(d).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        //
        // Get lookup commitments to m(x), g(x) and h(x) polys
        //
        let lookups = pk.cs().lookups();
        let num_lookups = lookups.len();

        let mut m_commitments = vec![C::identity(); num_lookups];
        for m_commitment in m_commitments.iter_mut() {
            *m_commitment = transcript.read_point()?;
        }
        // TODO(@gnosed): basic length check atm, verify both vectors instead for all lookups commitments
        assert_eq!(
            acc.lookup_transcript.singles_transcript.len(),
            m_commitments.len()
        );

        let mut g_commitments = vec![C::identity(); num_lookups];
        for g_commitment in g_commitments.iter_mut() {
            *g_commitment = transcript.read_point()?;
        }
        assert_eq!(
            acc.lookup_transcript.singles_transcript.len(),
            g_commitments.len()
        );

        let mut h_commitments = vec![C::identity(); num_lookups];
        for h_commitment in h_commitments.iter_mut() {
            *h_commitment = transcript.read_point()?;
        }
        assert_eq!(
            acc.lookup_transcript.singles_transcript.len(),
            h_commitments.len()
        );
        //
        // Get beta commitment
        //
        let beta_commitment = transcript.read_point()?;
        assert_eq!(
            acc.compressed_verifier_transcript.beta_commitment,
            beta_commitment
        );

        // Challenge for the RLC of all constraints (all gates and all lookups)
        let y = *transcript.squeeze_challenge_scalar::<C::Scalar>();
        // assert_eq!(acc.y, y);
        println!("{:?}", y);
        //
        // Get error commitments
        //
        let mut e_commitments = vec![C::identity(); num_lookups];
        for e_commitment in e_commitments.iter_mut() {
            *e_commitment = transcript.read_point()?;
        }

        // Get alpha challenge
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();
        println!("{:?}", alpha);

        Ok(VerifierAccumulator {
            instance_commitments,
            challenges,
            advice_commitments,
            beta_commitment,
            m_commitments,
            g_commitments,
            h_commitments,
            e_commitments,
            y,
            alpha,
        })
    }
}
/*
/// Returns a boolean indicating whether or not the accumulator is valid
pub fn verify_accumulator<
    'params,
    C: CurveAffine,
    P: Params<'params, C>,
    E: EncodedChallenge<C>,
    T: TranscriptWrite<C, E>,
>(
    acc: &Accumulator<C>,
    pk: &ProvingKey<C>,
    transcript: &mut T,
    params: &P,
) -> Result<bool, Error> {
    // Recompute commitments and compare them to those in the accumulator
    let commitments_ok = {
        // Commitments in projective coordinates
        let commitments_projective: Vec<_> = zip(acc_new.polynomials_iter(), acc_new.blinds_iter())
            .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
            .collect();
        // Convert to affine coordinates
        let mut commitments = vec![C::identity(); commitments_projective.len()];
        C::CurveExt::batch_normalize(&commitments_projective, &mut commitments);
        // Compare with accumulator commitments
        zip(commitments.into_iter(), acc_new.commitments_iter())
            .all(|(actual, expected)| actual == *expected)
    };

    // Recompute e = ∑ᵢ βᵢ⋅eᵢ, where eᵢ = ∑ⱼ yⱼ⋅Gⱼ(acc[i])
    let folding_errors_ok = {
        // Store βᵢ⋅eᵢ for each row i
        let mut folding_errors = vec![C::Scalar::ZERO; pk.num_usable_rows()];

        parallelize(&mut folding_errors, |errors, start| {
            let gates_selectors = pk.folding_constraints_selectors();
            let folding_constraints = pk.folding_constraints();

            // For each gate, create a slice of y's corresponding to the polynomials in the gate
            let gates_ys: Vec<&[C::Scalar]> = {
                let mut start = 0;
                let mut gates_ys = Vec::with_capacity(folding_constraints.len());

                for polys in folding_constraints {
                    let size = polys.len(); // Get size of inner vector
                    let end = start + size;
                    gates_ys.push(&acc_new.ys[start..end]);
                    start = end;
                }

                gates_ys
            };

            // Create a RowEvaluator for each gate
            let mut folding_gate_evs: Vec<_> = folding_constraints
                .iter()
                .map(|polys| RowEvaluator::new(&polys, &acc_new.advice_challenges()))
                .collect();

            for (i, error) in errors.iter_mut().enumerate() {
                let row = start + i;

                for (gate_ev, (gate_ys, gate_selector)) in folding_gate_evs
                    .iter_mut()
                    .zip(zip(gates_ys.iter(), gates_selectors.iter()))
                {
                    // Check whether the gate's selector is active at this row
                    if let Some(selector) = gate_selector {
                        if !pk.selectors[selector.index()][row] {
                            continue;
                        }
                    }

                    // Evaluate [Gⱼ(acc[i])] for all Gⱼ in the gate
                    let row_evals = gate_ev.evaluate(
                        row,
                        &pk.selectors,
                        &pk.fixed,
                        &acc_new.instance_polys(),
                        &acc_new.advice_polys(),
                    );

                    // Add ∑ⱼ yⱼ⋅Gⱼ(acc[i]) to eᵢ
                    *error += zip(gate_ys.iter(), row_evals.iter())
                        .fold(C::Scalar::ZERO, |acc, (y, eval)| acc + *y * eval);
                }
                // eᵢ *= βᵢ⋅eᵢ
                *error *= acc_new.beta_poly()[row];
            }
        });

        // Sum all eᵢ
        let folding_error = folding_errors.par_iter().sum::<C::Scalar>();

        folding_error == acc_new.error
    };

    // Check all linear constraints, which were ignored during folding
    let linear_errors_ok = {
        let mut linear_errors = vec![true; pk.num_usable_rows()];

        parallelize(&mut linear_errors, |errors, start| {
            let linear_constraints = pk.linear_constraints();

            // Create a single evaluator for all linear constraints
            let mut linear_gate_ev =
                RowEvaluator::new(linear_constraints, &acc_new.advice_challenges());

            for (i, error) in errors.iter_mut().enumerate() {
                let row = start + i;

                let row_evals = linear_gate_ev.evaluate(
                    row,
                    &pk.selectors,
                    &pk.fixed,
                    &acc_new.instance_polys(),
                    &acc_new.advice_polys(),
                );

                // Check that all constraints evaluate to 0.
                *error = row_evals.iter().all(|eval| eval.is_zero_vartime());
            }
        });

        linear_errors.par_iter().all(|row_ok| *row_ok)
    };

    // Get alpha challenge
    let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();

    // Check fold challenges
    for (c_acc, c_new) in zip(acc0.challenges_iter_mut(), acc1.challenges_iter()) {
        *c_acc += alpha * (*c_new - *c_acc);
    }
    let fold_challenges_ok = acc0
        .challenges_iter()
        .zip(acc_new.challenges_iter())
        .map(|(c, c_new)| c == c_new)
        .fold(true, |acc, x| acc && x);

    // Check fold polynomials
    for (poly_acc, poly_new) in zip(acc0.polynomials_iter_mut(), acc1.polynomials_iter()) {
        poly_acc.boolean_linear_combination(poly_new, alpha);
    }
    let fold_polynomials_ok = acc0
        .polynomials_iter()
        .zip(acc_new.polynomials_iter())
        .map(|(c, c_new)| c == c_new)
        .fold(true, |acc, x| acc && x);
    // Check fold commitments
    {
        // Compute folded commitments in projective coordinates
        let commitments_projective: Vec<_> = zip(acc0.commitments_iter(), acc1.commitments_iter())
            .map(|(c_acc, c_new)| (*c_new - *c_acc) * alpha + *c_acc)
            .collect();
        // Convert to affine coordinates
        let mut commitments_affine = vec![C::identity(); commitments_projective.len()];
        C::CurveExt::batch_normalize(&commitments_projective, &mut commitments_affine);
        for (c_acc, c_new) in zip(acc0.commitments_iter_mut(), commitments_affine.into_iter()) {
            *c_acc = c_new
        }
    }
    let fold_commitments_ok = acc0
        .commitments_iter()
        .zip(acc_new.commitments_iter())
        .map(|(c, c_new)| c == c_new)
        .fold(true, |acc, x| acc && x);
    // Check fold blinds
    for (blind_acc, blind_new) in zip(acc0.blinds_iter_mut(), acc1.blinds_iter()) {
        *blind_acc += (*blind_new - *blind_acc) * alpha
    }
    let fold_blinds_ok = acc0
        .blinds_iter()
        .zip(acc_new.blinds_iter())
        .map(|(c, c_new)| c == c_new)
        .fold(true, |acc, x| acc && x);

    Ok(commitments_ok & folding_errors_ok & linear_errors_ok
        && fold_challenges_ok
        && fold_polynomials_ok
        && fold_commitments_ok
        && fold_blinds_ok)
    // TODO:
    // - Check lookup h, g, m poly
    // - Check error poly, getting it from transcript and evaluating at alpha
    // - Recompute commitments and compare them to those in the accumulator
    // - Recompute e = ∑ᵢ βᵢ⋅eᵢ, where eᵢ = ∑ⱼ yⱼ⋅Gⱼ(acc[i])
    // - Check all linear constraints, which were ignored during folding

    // let instance_commitments = if V::QUERY_INSTANCE {
    //     instances
    //         .iter()
    //         .map(|instance| {
    //             instance
    //                 .iter()
    //                 .map(|instance| {
    //                     if instance.len() > params.n() as usize - (vk.cs.blinding_factors() + 1) {
    //                         return Err(Error::InstanceTooLarge);
    //                     }
    //                     let mut poly = instance.to_vec();
    //                     poly.resize(params.n() as usize, Scheme::Scalar::ZERO);
    //                     let poly = vk.domain.lagrange_from_vec(poly);

    //                     Ok(params.commit_lagrange(&poly, Blind::default()).to_affine())
    //                 })
    //                 .collect::<Result<Vec<_>, _>>()
    //         })
    //         .collect::<Result<Vec<_>, _>>()?
    // } else {
    //     vec![vec![]; instances.len()]
    // };

    // let num_proofs = instance_commitments.len();

    // Hash verification key into transcript
    // vk.hash_into(transcript)?;

    // if V::QUERY_INSTANCE {
    //     for instance_commitments in instance_commitments.iter() {
    //         // Hash the instance (external) commitments into the transcript
    //         for commitment in instance_commitments {
    //             transcript.common_point(*commitment)?
    //         }
    //     }
    // } else {
    //     for instance in instances.iter() {
    //         for instance in instance.iter() {
    //             for value in instance.iter() {
    //                 transcript.common_scalar(*value)?;
    //             }
    //         }
    //     }
    // }
    /*

    // Hash the prover's advice commitments into the transcript and squeeze challenges
    let (advice_commitments, challenges) = {
        let mut advice_commitments =
            vec![vec![Scheme::Curve::default(); vk.cs.num_advice_columns]; num_proofs];
        let mut challenges = vec![Scheme::Scalar::ZERO; vk.cs.num_challenges];

        for current_phase in vk.cs.phases() {
            for advice_commitments in advice_commitments.iter_mut() {
                for (phase, commitment) in vk
                    .cs
                    .advice_column_phase
                    .iter()
                    .zip(advice_commitments.iter_mut())
                {
                    if current_phase == *phase {
                        *commitment = transcript.read_point()?;
                    }
                }
            }
            for (phase, challenge) in vk.cs.challenge_phase.iter().zip(challenges.iter_mut()) {
                if current_phase == *phase {
                    *challenge = *transcript.squeeze_challenge_scalar::<()>();
                }
            }
        }

        (advice_commitments, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    let lookups_permuted = (0..num_proofs)
        .map(|_| -> Result<Vec<_>, _> {
            // Hash each lookup permuted commitment
            vk.cs
                .lookups
                .iter()
                .map(|argument| argument.read_permuted_commitments(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let permutations_committed = (0..num_proofs)
        .map(|_| {
            // Hash each permutation product commitment
            vk.cs.permutation.read_product_commitments(vk, transcript)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let lookups_committed = lookups_permuted
        .into_iter()
        .map(|lookups| {
            // Hash each lookup product commitment
            lookups
                .into_iter()
                .map(|lookup| lookup.read_product_commitment(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let shuffles_committed = (0..num_proofs)
        .map(|_| -> Result<Vec<_>, _> {
            // Hash each shuffle product commitment
            vk.cs
                .shuffles
                .iter()
                .map(|argument| argument.read_product_commitment(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let vanishing = vanishing::Argument::read_commitments_before_y(transcript)?;

    // Sample y challenge, which keeps the gates linearly independent.
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    let vanishing = vanishing.read_commitments_after_y(vk, transcript)?;

    // Sample x challenge, which is used to ensure the circuit is
    // satisfied with high probability.
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let instance_evals = if V::QUERY_INSTANCE {
        (0..num_proofs)
            .map(|_| -> Result<Vec<_>, _> {
                read_n_scalars(transcript, vk.cs.instance_queries.len())
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let xn = x.pow(&[params.n() as u64, 0, 0, 0]);
        let (min_rotation, max_rotation) =
            vk.cs
                .instance_queries
                .iter()
                .fold((0, 0), |(min, max), (_, rotation)| {
                    if rotation.0 < min {
                        (rotation.0, max)
                    } else if rotation.0 > max {
                        (min, rotation.0)
                    } else {
                        (min, max)
                    }
                });
        let max_instance_len = instances
            .iter()
            .flat_map(|instance| instance.iter().map(|instance| instance.len()))
            .max_by(Ord::cmp)
            .unwrap_or_default();
        let l_i_s = &vk.domain.l_i_range(
            *x,
            xn,
            -max_rotation..max_instance_len as i32 + min_rotation.abs(),
        );
        instances
            .iter()
            .map(|instances| {
                vk.cs
                    .instance_queries
                    .iter()
                    .map(|(column, rotation)| {
                        let instances = instances[column.index()];
                        let offset = (max_rotation - rotation.0) as usize;
                        compute_inner_product(instances, &l_i_s[offset..offset + instances.len()])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    let advice_evals = (0..num_proofs)
        .map(|_| -> Result<Vec<_>, _> { read_n_scalars(transcript, vk.cs.advice_queries.len()) })
        .collect::<Result<Vec<_>, _>>()?;

    let fixed_evals = read_n_scalars(transcript, vk.cs.fixed_queries.len())?;

    let vanishing = vanishing.evaluate_after_x(transcript)?;

    let permutations_common = vk.permutation.evaluate(transcript)?;

    let permutations_evaluated = permutations_committed
        .into_iter()
        .map(|permutation| permutation.evaluate(transcript))
        .collect::<Result<Vec<_>, _>>()?;

    let lookups_evaluated = lookups_committed
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|lookup| lookup.evaluate(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let shuffles_evaluated = shuffles_committed
        .into_iter()
        .map(|shuffles| -> Result<Vec<_>, _> {
            shuffles
                .into_iter()
                .map(|shuffle| shuffle.evaluate(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // This check ensures the circuit is satisfied so long as the polynomial
    // commitments open to the correct values.
    let vanishing = {
        // x^n
        let xn = x.pow(&[params.n() as u64, 0, 0, 0]);

        let blinding_factors = vk.cs.blinding_factors();
        let l_evals = vk
            .domain
            .l_i_range(*x, xn, (-((blinding_factors + 1) as i32))..=0);
        assert_eq!(l_evals.len(), 2 + blinding_factors);
        let l_last = l_evals[0];
        let l_blind: Scheme::Scalar = l_evals[1..(1 + blinding_factors)]
            .iter()
            .fold(Scheme::Scalar::ZERO, |acc, eval| acc + eval);
        let l_0 = l_evals[1 + blinding_factors];

        // Compute the expected value of h(x)
        let expressions = advice_evals
            .iter()
            .zip(instance_evals.iter())
            .zip(permutations_evaluated.iter())
            .zip(lookups_evaluated.iter())
            .zip(shuffles_evaluated.iter())
            .flat_map(
                |((((advice_evals, instance_evals), permutation), lookups), shuffles)| {
                    let challenges = &challenges;
                    let fixed_evals = &fixed_evals;
                    std::iter::empty()
                        // Evaluate the circuit using the custom gates provided
                        .chain(vk.cs.gates.iter().flat_map(move |gate| {
                            gate.polynomials().iter().map(move |poly| {
                                poly.evaluate(
                                    &|scalar| scalar,
                                    &|_| {
                                        panic!("virtual selectors are removed during optimization")
                                    },
                                    &|query| fixed_evals[query.index.unwrap()],
                                    &|query| advice_evals[query.index.unwrap()],
                                    &|query| instance_evals[query.index.unwrap()],
                                    &|challenge| challenges[challenge.index()],
                                    &|a| -a,
                                    &|a, b| a + &b,
                                    &|a, b| a * &b,
                                    &|a, scalar| a * &scalar,
                                )
                            })
                        }))
                        .chain(permutation.expressions(
                            vk,
                            &vk.cs.permutation,
                            &permutations_common,
                            advice_evals,
                            fixed_evals,
                            instance_evals,
                            l_0,
                            l_last,
                            l_blind,
                            beta,
                            gamma,
                            x,
                        ))
                        .chain(
                            lookups
                                .iter()
                                .zip(vk.cs.lookups.iter())
                                .flat_map(move |(p, argument)| {
                                    p.expressions(
                                        l_0,
                                        l_last,
                                        l_blind,
                                        argument,
                                        theta,
                                        beta,
                                        gamma,
                                        advice_evals,
                                        fixed_evals,
                                        instance_evals,
                                        challenges,
                                    )
                                })
                                .into_iter(),
                        )
                        .chain(
                            shuffles
                                .iter()
                                .zip(vk.cs.shuffles.iter())
                                .flat_map(move |(p, argument)| {
                                    p.expressions(
                                        l_0,
                                        l_last,
                                        l_blind,
                                        argument,
                                        theta,
                                        gamma,
                                        advice_evals,
                                        fixed_evals,
                                        instance_evals,
                                        challenges,
                                    )
                                })
                                .into_iter(),
                        )
                },
            );

        vanishing.verify(params, expressions, y, xn)
    };

    let queries = instance_commitments
        .iter()
        .zip(instance_evals.iter())
        .zip(advice_commitments.iter())
        .zip(advice_evals.iter())
        .zip(permutations_evaluated.iter())
        .zip(lookups_evaluated.iter())
        .zip(shuffles_evaluated.iter())
        .flat_map(
            |(
                (
                    (
                        (
                            ((instance_commitments, instance_evals), advice_commitments),
                            advice_evals,
                        ),
                        permutation,
                    ),
                    lookups,
                ),
                shuffles,
            )| {
                iter::empty()
                    .chain(
                        V::QUERY_INSTANCE
                            .then_some(vk.cs.instance_queries.iter().enumerate().map(
                                move |(query_index, &(column, at))| {
                                    VerifierQuery::new_commitment(
                                        &instance_commitments[column.index()],
                                        vk.domain.rotate_omega(*x, at),
                                        instance_evals[query_index],
                                    )
                                },
                            ))
                            .into_iter()
                            .flatten(),
                    )
                    .chain(vk.cs.advice_queries.iter().enumerate().map(
                        move |(query_index, &(column, at))| {
                            VerifierQuery::new_commitment(
                                &advice_commitments[column.index()],
                                vk.domain.rotate_omega(*x, at),
                                advice_evals[query_index],
                            )
                        },
                    ))
                    .chain(permutation.queries(vk, x))
                    .chain(
                        lookups
                            .iter()
                            .flat_map(move |p| p.queries(vk, x))
                            .into_iter(),
                    )
                    .chain(
                        shuffles
                            .iter()
                            .flat_map(move |p| p.queries(vk, x))
                            .into_iter(),
                    )
            },
        )
        .chain(
            vk.cs
                .fixed_queries
                .iter()
                .enumerate()
                .map(|(query_index, &(column, at))| {
                    VerifierQuery::new_commitment(
                        &vk.fixed_commitments[column.index()],
                        vk.domain.rotate_omega(*x, at),
                        fixed_evals[query_index],
                    )
                }),
        )
        .chain(permutations_common.queries(&vk.permutation, x))
        .chain(vanishing.queries(x));

    // We are now convinced the circuit is satisfied so long as the
    // polynomial commitments open to the correct values.

    let verifier = V::new(params);
    strategy.process(|msm| {
        verifier
            .verify_proof(transcript, queries, msm)
            .map_err(|_| Error::Opening)
    })
    */
}

*/

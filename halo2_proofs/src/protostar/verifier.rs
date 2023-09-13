use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use halo2curves::{CurveAffine, CurveExt};
use rand_core::RngCore;
use std::{
    collections::{BTreeSet, HashMap},
    iter::zip,
};

use super::transcript::{
    advice::AdviceTranscript,
    compressed_verifier::CompressedVerifierTranscript,
    instance::InstanceTranscript,
    lookup::{LookupTranscipt, LookupTranscriptSingle},
};
use super::{error_check::Accumulator, keygen::ProvingKey, row_evaluator::RowEvaluator};
use crate::arithmetic::{
    best_multiexp, compute_inner_product, eval_polynomial, parallelize, powers,
};
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

const BETA_POLY_DEGREE: usize = 1;

#[derive(Debug, Clone, PartialEq)]
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
    pub y: C::Scalar,
    pub constraint_errors: Vec<C::Scalar>,
    pub error: C::Scalar,
}

impl<C: CurveAffine> VerifierAccumulator<C> {
    /// Create a new `VerifierAccumulator` by reading the IOP transcripts from the Prover and save commitments and challenges
    pub fn new_from_prover<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        transcript: &mut T,
        instances: &[&[C::Scalar]],
        // TODO(@gnosed): replace pk with vk: VerifiyingKey<C>
        pk: &ProvingKey<C>,
        // TODO(@gnosed): remove accumulator when testing is correct
    ) -> Result<Self, Error> {
        //
        // Get instance commitments
        //
        // Check that instances matches the expected number of instance columns
        if instances.len() != pk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }

        let mut instance_commitments = vec![C::Scalar::ZERO; instances.len()];

        for instance_commitment in instance_commitments.iter_mut() {
            *instance_commitment = transcript.read_scalar()?;
        }

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
        let num_proofs = instance_commitments.len();
        let (advice_commitments, challenges) = {
            let mut advice_commitments = vec![C::identity(); pk.cs.num_advice_columns];
            let mut challenges = vec![C::Scalar::ZERO; pk.cs.num_challenges];

            for current_phase in pk.cs.phases() {
                    for (phase, commitment) in pk
                        .cs
                        .advice_column_phase
                        .iter()
                        .zip(advice_commitments.iter_mut())
                    {
                        if current_phase == *phase {
                            *commitment = transcript.read_point()?;
                        }
                    }
                for (phase, challenge) in pk.cs.challenge_phase.iter().zip(challenges.iter_mut()) {
                    if current_phase == *phase {
                        *challenge = *transcript.squeeze_challenge_scalar::<()>();
                    }
                }
            }

            (advice_commitments, challenges)
        };
        println!("AAAAA {:?}", advice_commitments);
        println!("CCC {:?}", challenges);
        // let mut advice_commitments = vec![C::identity(); pk.cs.num_advice_columns];

        // FIXME: read advice commitments with phase

        // for advice_commitment in advice_commitments.iter_mut() {
        //     *advice_commitment = transcript.read_point()?;
        // }

        // let mut challenges = HashMap::with_capacity(pk.cs.num_challenges);

        // for current_phase in pk.cs.phases() {
        //     for (index, phase) in pk.cs.challenge_phase.iter().enumerate() {
        //         if current_phase == *phase {
        //             challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
        //         }
        //     }
        // }

        // let challenges = (0..pk.cs.num_challenges)
        //     .map(|index| challenges.remove(&index).unwrap())
        // println!("VERIFIER challenges 1 {:?}", challenges);

        let challenge_degrees = pk.max_challenge_powers();
        // println!("challenge degrees {:?}", challenge_degrees);
        let challenges = challenges
            .into_iter()
            .zip(challenge_degrees)
            .map(|(c, d)| powers(c).skip(1).take(d).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        // println!("challenges 2 {:?}", challenges);
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

        let mut g_commitments = vec![C::identity(); num_lookups];
        for g_commitment in g_commitments.iter_mut() {
            *g_commitment = transcript.read_point()?;
        }

        let mut h_commitments = vec![C::identity(); num_lookups];
        for h_commitment in h_commitments.iter_mut() {
            *h_commitment = transcript.read_point()?;
        }
        //
        // Get beta commitment
        //
        let _beta = *transcript.squeeze_challenge_scalar::<C::Scalar>();
        let beta_commitment = transcript.read_point()?;

        // FIXME: `y` is not the same as in the prover
        // Challenge for the RLC of all constraints (all gates and all lookups)
        let y = *transcript.squeeze_challenge_scalar::<C::Scalar>();

        let num_constraints = pk.num_folding_constraints();

        let constraint_errors = vec![C::Scalar::ZERO; num_constraints];

        Ok(VerifierAccumulator {
            instance_commitments,
            challenges,
            advice_commitments,
            beta_commitment,
            m_commitments,
            g_commitments,
            h_commitments,
            y,
            constraint_errors,
            error: C::Scalar::ZERO,
        })
    }
    pub fn fold<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        &mut self,
        acc1: &VerifierAccumulator<C>,
        pk: &ProvingKey<C>,
        transcript: &mut T,
    ) {
        /*
        Compute the number of constraints in each gate as well as the total number of constraints in all gates.
        */
        let gates_error_polys_num_evals: Vec<Vec<usize>> = pk
            .folding_constraints()
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| poly.folding_degree() + BETA_POLY_DEGREE + 1)
                    .collect()
            })
            .collect();

        let max_gate_num_evals = gates_error_polys_num_evals
            .iter()
            .flat_map(|gates_num_evals| gates_num_evals.iter())
            .max()
            .unwrap();
        //
        // Get error commitments
        let final_error_poly_len = max_gate_num_evals + 1;
        // Prover doesn't send the first two coefficient since Verifier already know e(0) and e(1)
        let quotient_final_error_poly_len = final_error_poly_len - 2;

        let mut e_commitments = vec![C::Scalar::ZERO; quotient_final_error_poly_len];
        for e_commitment in e_commitments.iter_mut() {
            *e_commitment = transcript.read_scalar().unwrap();
        }
        let alpha = *transcript.squeeze_challenge_scalar::<C::Scalar>();

        // eval e'(alpha), then eval e(alpha) = (1-alpha)*alpha*e'(alpha) + (1-alpha)*e(0) + alpha*e(1)
        let quotient_final_error_poly = eval_polynomial(&e_commitments, alpha);
        let final_error = alpha * (C::Scalar::ONE - alpha) * quotient_final_error_poly
            + (C::Scalar::ONE - alpha) * self.error
            + alpha * acc1.error;
        self.error = final_error;
    }
}

#[cfg(test)]
mod tests {
    use ff::{BatchInvert, FromUniformBytes};

    use crate::{
        arithmetic::{CurveAffine, Field},
        circuit::{floor_planner::V1, Layouter, Value},
        dev::{metadata, FailureLocation, MockProver, VerifyFailure},
        plonk::*,
        poly::{
            self,
            commitment::ParamsProver,
            ipa::{
                commitment::{IPACommitmentScheme, ParamsIPA},
                multiopen::{ProverIPA, VerifierIPA},
            },
            VerificationStrategy,
        },
        protostar,
        protostar::error_check::Accumulator,
        protostar::verifier::VerifierAccumulator,
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    };

    use halo2curves::pasta::pallas;
    use rand_core::{OsRng, RngCore};
    use std::iter::{self, zip};

    fn rand_2d_array<F: Field, R: RngCore, const W: usize, const H: usize>(
        rng: &mut R,
    ) -> [[F; H]; W] {
        [(); W].map(|_| [(); H].map(|_| F::random(&mut *rng)))
    }

    fn shuffled<F: Field, R: RngCore, const W: usize, const H: usize>(
        original: [[F; H]; W],
        rng: &mut R,
    ) -> [[F; H]; W] {
        let mut shuffled = original;

        for row in (1..H).rev() {
            let rand_row = (rng.next_u32() as usize) % row;
            for column in shuffled.iter_mut() {
                column.swap(row, rand_row);
            }
        }

        shuffled
    }

    #[derive(Clone)]
    pub struct MyConfig<const W: usize> {
        q_shuffle: Selector,
        q_first: Selector,
        q_last: Selector,
        original: [Column<Advice>; W],
        shuffled: [Column<Advice>; W],
        theta: Challenge,
        gamma: Challenge,
        z: Column<Advice>,
    }

    impl<const W: usize> MyConfig<W> {
        fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {
            let [q_shuffle, q_first, q_last] = [(); 3].map(|_| meta.selector());
            // First phase
            let original = [(); W].map(|_| meta.advice_column_in(FirstPhase));
            let shuffled = [(); W].map(|_| meta.advice_column_in(FirstPhase));
            let [theta, gamma] = [(); 2].map(|_| meta.challenge_usable_after(FirstPhase));
            // Second phase
            let z = meta.advice_column_in(SecondPhase);

            meta.create_gate("z should start with 1", |_| {
                let one = Expression::Constant(F::ONE);

                vec![q_first.expr() * (one - z.cur())]
            });

            meta.create_gate("z should end with 1", |_| {
                let one = Expression::Constant(F::ONE);

                vec![q_last.expr() * (one - z.cur())]
            });

            meta.create_gate("z should have valid transition", |_| {
                let q_shuffle = q_shuffle.expr();
                let original = original.map(|advice| advice.cur());
                let shuffled = shuffled.map(|advice| advice.cur());
                let [theta, gamma] = [theta, gamma].map(|challenge| challenge.expr());

                // Compress
                let original = original
                    .iter()
                    .cloned()
                    .reduce(|acc, a| acc * theta.clone() + a)
                    .unwrap();
                let shuffled = shuffled
                    .iter()
                    .cloned()
                    .reduce(|acc, a| acc * theta.clone() + a)
                    .unwrap();

                vec![
                    q_shuffle
                        * (z.cur() * (original + gamma.clone()) - z.next() * (shuffled + gamma)),
                ]
            });

            Self {
                q_shuffle,
                q_first,
                q_last,
                original,
                shuffled,
                theta,
                gamma,
                z,
            }
        }
    }

    #[derive(Clone, Default)]
    pub struct MyCircuit<F: Field, const W: usize, const H: usize> {
        original: Value<[[F; H]; W]>,
        shuffled: Value<[[F; H]; W]>,
    }

    impl<F: Field, const W: usize, const H: usize> MyCircuit<F, W, H> {
        pub fn rand<R: RngCore>(rng: &mut R) -> Self {
            let original = rand_2d_array::<F, _, W, H>(rng);
            let shuffled = shuffled(original, rng);

            Self {
                original: Value::known(original),
                shuffled: Value::known(shuffled),
            }
        }
    }

    impl<F: Field, const W: usize, const H: usize> Circuit<F> for MyCircuit<F, W, H> {
        type Config = MyConfig<W>;
        type FloorPlanner = V1;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            MyConfig::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let theta = layouter.get_challenge(config.theta);
            let gamma = layouter.get_challenge(config.gamma);

            layouter.assign_region(
                || "Shuffle original into shuffled",
                |mut region| {
                    // Keygen
                    config.q_first.enable(&mut region, 0)?;
                    config.q_last.enable(&mut region, H)?;
                    for offset in 0..H {
                        config.q_shuffle.enable(&mut region, offset)?;
                    }

                    // First phase
                    for (idx, (&column, values)) in zip(
                        config.original.iter(),
                        self.original.transpose_array().iter(),
                    )
                    .enumerate()
                    {
                        for (offset, &value) in values.transpose_array().iter().enumerate() {
                            region.assign_advice(
                                || format!("original[{}][{}]", idx, offset),
                                column,
                                offset,
                                || value,
                            )?;
                        }
                    }
                    for (idx, (&column, values)) in zip(
                        config.shuffled.iter(),
                        self.shuffled.transpose_array().iter(),
                    )
                    .enumerate()
                    {
                        for (offset, &value) in values.transpose_array().iter().enumerate() {
                            region.assign_advice(
                                || format!("shuffled[{}][{}]", idx, offset),
                                column,
                                offset,
                                || value,
                            )?;
                        }
                    }

                    // Second phase
                    let z = self.original.zip(self.shuffled).zip(theta).zip(gamma).map(
                        |(((original, shuffled), theta), gamma)| {
                            let mut product = vec![F::ZERO; H];
                            for (idx, product) in product.iter_mut().enumerate() {
                                let mut compressed = F::ZERO;
                                for value in shuffled.iter() {
                                    compressed *= theta;
                                    compressed += value[idx];
                                }

                                *product = compressed + gamma;
                            }

                            product.iter_mut().batch_invert();

                            for (idx, product) in product.iter_mut().enumerate() {
                                let mut compressed = F::ZERO;
                                for value in original.iter() {
                                    compressed *= theta;
                                    compressed += value[idx];
                                }

                                *product *= compressed + gamma;
                            }

                            #[allow(clippy::let_and_return)]
                            let z = iter::once(F::ONE)
                                .chain(product)
                                .scan(F::ONE, |state, cur| {
                                    *state *= &cur;
                                    Some(*state)
                                })
                                .collect::<Vec<_>>();

                            #[cfg(feature = "sanity-checks")]
                            assert_eq!(F::ONE, *z.last().unwrap());

                            z
                        },
                    );
                    for (offset, value) in z.transpose_vec(H + 1).into_iter().enumerate() {
                        region.assign_advice(
                            || format!("z[{}]", offset),
                            config.z,
                            offset,
                            || value,
                        )?;
                    }

                    Ok(())
                },
            )
        }
    }

    fn check_v_and_p_transcripts<C: CurveAffine>(
        v_acc: VerifierAccumulator<C>,
        p_acc: Accumulator<C>,
    ) {
        // Check acc0 and v_acc0 transcripts
        assert_eq!(
            p_acc.instance_transcript.verifier_instance_commitments, v_acc.instance_commitments,
            "V and P Instance Transcripts NOT EQUAL"
        );
        assert_eq!(
            p_acc.advice_transcript.advice_commitments, v_acc.advice_commitments,
            "V and P Advice Transcripts NOT EQUAL"
        );
        assert_eq!(
            p_acc.advice_transcript.challenges, v_acc.challenges,
            "V and P Advice Challenges NOT EQUAL"
        );
        assert_eq!(
            p_acc.lookup_transcript.singles_transcript.len(),
            v_acc.m_commitments.len(),
            "V and P m(x) Commitments NOT EQUAL"
        );
        assert_eq!(
            p_acc.lookup_transcript.singles_transcript.len(),
            v_acc.g_commitments.len(),
            "V and P g(x) Commitments NOT EQUAL"
        );
        assert_eq!(
            p_acc.lookup_transcript.singles_transcript.len(),
            v_acc.h_commitments.len(),
            "V and P h(x) Commitments NOT EQUAL"
        );
        assert_eq!(
            p_acc.compressed_verifier_transcript.beta_commitment, v_acc.beta_commitment,
            "V and P Beta Commitment NOT EQUAL"
        );
        assert_eq!(p_acc.y, v_acc.y, "V and P Y challenge NOT EQUAL");
    }

    #[test]
    fn test_one_verifier_acc() {
        let mut rng: OsRng = OsRng;

        const W: usize = 4;
        const H: usize = 32;
        const K: u32 = 8;

        let params = poly::ipa::commitment::ParamsIPA::<pallas::Affine>::new(K);

        let circuit = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);

        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let pk = protostar::ProvingKey::new(&params, &circuit).unwrap();

        let p_acc = protostar::prover::create_accumulator(
            &params,
            &pk,
            &circuit,
            &[],
            &mut rng,
            &mut transcript,
        )
        .unwrap();

        let proof: Vec<u8> = transcript.finalize();

        let mut v_transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let v_acc = VerifierAccumulator::new_from_prover(&mut v_transcript, &[], &pk).unwrap();

        check_v_and_p_transcripts(v_acc, p_acc);
    }

    #[test]
    fn test_two_verifier_acc() {
        let mut rng: OsRng = OsRng;

        const W: usize = 4;
        const H: usize = 32;
        const K: u32 = 8;

        let params = poly::ipa::commitment::ParamsIPA::<pallas::Affine>::new(K);

        let circuit0 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);
        let circuit1 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);

        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let pk = protostar::ProvingKey::new(&params, &circuit0).unwrap();

        let acc0 = protostar::prover::create_accumulator(
            &params,
            &pk,
            &circuit0,
            &[],
            &mut rng,
            &mut transcript,
        )
        .unwrap();
        let acc1 = protostar::prover::create_accumulator(
            &params,
            &pk,
            &circuit1,
            &[],
            &mut rng,
            &mut transcript,
        )
        .unwrap();

        let proof: Vec<u8> = transcript.finalize();

        let mut v_transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        let v_acc0 = VerifierAccumulator::new_from_prover(&mut v_transcript, &[], &pk).unwrap();
        let v_acc1 = VerifierAccumulator::new_from_prover(&mut v_transcript, &[], &pk).unwrap();

        // Check acc0 and v_acc0 transcripts
        check_v_and_p_transcripts(v_acc0, acc0);
        // Check acc1 and v_acc1 transcripts
        check_v_and_p_transcripts(v_acc1, acc1);
    }

    #[test]
    fn test_two_verifier_acc_folding() {
        let mut rng: OsRng = OsRng;

        const W: usize = 4;
        const H: usize = 32;
        const K: u32 = 8;

        let params = poly::ipa::commitment::ParamsIPA::<pallas::Affine>::new(K);

        let circuit0 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);
        let circuit1 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);

        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let pk = protostar::ProvingKey::new(&params, &circuit0).unwrap();

        println!("---Create Accumulator #0---");
        let mut acc0 = protostar::prover::create_accumulator(
            &params,
            &pk,
            &circuit0,
            &[],
            &mut rng,
            &mut transcript,
        )
        .unwrap();
        println!("---Create Accumulator #1---");
        let acc1 = protostar::prover::create_accumulator(
            &params,
            &pk,
            &circuit1,
            &[],
            &mut rng,
            &mut transcript,
        )
        .unwrap();
        let acc0_old = acc0.clone();

        println!("---Fold Accumulator #1 into #0---");
        acc0.fold(&pk, acc1.clone(), &mut transcript);

        println!("---Finalize Transcript---");
        let proof: Vec<u8> = transcript.finalize();

        println!("---Create Verifier Transcript---");
        let mut v_transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

        println!("---Create Verifier Accumulator #0---");
        let mut v_acc0 = VerifierAccumulator::new_from_prover(&mut v_transcript, &[], &pk).unwrap();

        println!("---Create Verifier Accumulator #1---");
        let v_acc1 = VerifierAccumulator::new_from_prover(&mut v_transcript, &[], &pk).unwrap();

        println!("---Create Folded Verifier Accumulator #1 into #0---");
        let v_acc0_old = v_acc0.clone();
        v_acc0.fold(&v_acc1.clone(), &pk, &mut v_transcript);

        println!("---Check V and P #0---");
        check_v_and_p_transcripts(v_acc0_old, acc0_old);
        println!("---Check V and P #1---");
        check_v_and_p_transcripts(v_acc1, acc1);
        println!("---Check V and P fold #1 into #0---");
        println!("V error: {:?}, P error {:?}", v_acc0.error, acc0.error);
        // check_v_and_p_transcripts(v_acc0, acc0);
    }
}
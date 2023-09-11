use ff::{BatchInvert, FromUniformBytes};
use halo2_proofs::{
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
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    protostar::{self, decider::create_proof},
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::pallas;
use rand_core::{OsRng, RngCore};
use std::iter::{self, zip};

fn rand_2d_array<F: Field, R: RngCore, const W: usize, const H: usize>(rng: &mut R) -> [[F; H]; W] {
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

            vec![q_shuffle * (z.cur() * (original + gamma.clone()) - z.next() * (shuffled + gamma))]
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

fn main() {
    let mut rng = OsRng;

    const W: usize = 4;
    const H: usize = 32;
    const K: u32 = 8;

    let params = poly::ipa::commitment::ParamsIPA::<pallas::Affine>::new(K);

    let circuit1 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);
    let circuit2 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);
    let circuit3 = MyCircuit::<pallas::Scalar, W, H>::rand(&mut rng);

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let pk = protostar::ProvingKey::new(&params, &circuit1).unwrap();
    let mut acc = protostar::prover::create_accumulator(
        &params,
        &pk,
        &circuit1,
        &[],
        &mut rng,
        &mut transcript,
    )
    .unwrap();
    assert!(acc.decide(&params, &pk));

    // Folding an accumulator with itself should yield the same one,
    // since (1-X)*acc + X*acc = acc
    let acc1 = acc.clone();
    acc.fold(&pk, acc1.clone(), &mut transcript);
    assert!(acc.decide(&params, &pk));
    assert_eq!(acc, acc1);

    let acc2 = protostar::prover::create_accumulator(
        &params,
        &pk,
        &circuit2,
        &[],
        &mut rng,
        &mut transcript,
    )
    .unwrap();
    acc.fold(&pk, acc2, &mut transcript);
    assert!(acc.decide(&params, &pk));

    let acc3 = protostar::prover::create_accumulator(
        &params,
        &pk,
        &circuit3,
        &[],
        &mut rng,
        &mut transcript,
    )
    .unwrap();
    acc.fold(&pk, acc3, &mut transcript);

    let acc4 = protostar::prover::create_accumulator(
        &params,
        &pk,
        &circuit3,
        &[],
        &mut rng,
        &mut transcript,
    )
    .unwrap();
    acc.fold(&pk, acc4, &mut transcript);
    assert!(acc.decide(&params, &pk));

    let _ = create_proof::<_, ProverIPA<_>, _, _, _>(&params, &pk, acc, &mut rng, &mut transcript);
}

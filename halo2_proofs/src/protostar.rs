//! Contains logic for handling halo2 circuits with Protostar

mod error_check;
mod keygen;
pub mod prover;
mod row_evaluator;
mod transcript;
pub mod verifier;

pub use error_check::Accumulator;
pub use keygen::ProvingKey;

//! Contains logic for handling halo2 circuits with Protostar

pub mod decider;
mod error_check;
mod keygen;
pub mod prover;
mod row_evaluator;
mod transcript;

pub use error_check::Accumulator;
pub use keygen::ProvingKey;

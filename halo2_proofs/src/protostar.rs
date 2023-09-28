//! Contains logic for handling halo2 circuits with Protostar

pub mod accumulator;
mod constraints;
mod keygen;
pub mod prover;
pub mod verifier;

pub use keygen::ProvingKey;

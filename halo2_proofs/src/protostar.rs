//! Contains logic for handling halo2 circuits with Protostar

mod error_check;
mod gate;
mod keygen;
pub mod prover;
mod transcript;

pub use error_check::Accumulator;
pub use keygen::ProvingKey;

/// Each Transcript handles a different section of the IOP for the circuit
use halo2curves::CurveAffine;

use super::keygen::ProvingKey;

pub mod advice;
pub mod compressed_verifier;
pub mod instance;
pub mod lookup;

// trait InteractiveTranscript<C: CurveAffine> {
//     fn polynomials(&self) -> &[Poly]
// }

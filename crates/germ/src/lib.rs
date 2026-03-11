//! Minimal SP1-native GERM + AADP primitives.
//!
//! This crate intentionally excludes prover orchestration and heavy recursion machinery.
//! Build bundle extraction at the caller/bin layer, then use this crate for:
//! - commitment binding (`C`)
//! - post-commitment challenge derivation (`r_lin`, `r_mul`) from public arming context + `C`
//! - global vanishing checks `V_lin(r_lin)=0` and `V_mul(r_mul)=0`
//! - linear/multiplicative proof-binding checks
//! - tiny-verifier AADP constraint compilation
//! - paper-style AADP matrix encryption/decryption helpers

pub mod aadp;
mod ajtai;
pub mod bundle;
pub mod germ;

pub use aadp::{
    aadp_encrypt_bytes, aadp_encrypt_scalar, aadp_encrypt_u128, AadpByteCiphertext, AadpCiphertext,
    AadpConstraintSystem, AadpField, AadpLinearForm, AadpMulConstraint, Sp1AadpField,
};
pub use bundle::{
    GermArmCapsule, GermResidualPlan, GermVerifierStage, LinearResidualDescriptor,
    MultiplicativeResidualDescriptor, Sp1GermBundle, Sp1GermProofObject, Sp1LinProof, Sp1LinTerm,
    Sp1MulSumcheckProof, Sp1MulSumcheckRound, Sp1MulTerm, TranscriptBoundSp1GermProofObject,
};
pub use germ::{
    arm_germ_aadp_template, bind_bundle, bind_bundle_to_capsule, compile_germ_aadp_template,
    compute_commitment_root, derive_challenges, derive_challenges_from_arming_digest,
    derive_challenges_from_capsule, materialize_transcript_bound_germ_aadp_witness, verify_lin,
    verify_mul, verify_transcript_bound_sp1_germ_proof_object, ArmedGermAadpCiphertext,
    GermAadpConstraintStats, GermAadpVerifierTemplate, GermAadpWitness, GermAadpWitnessLayout,
    GermChallenges, GermError, GermPublicValues, GermRelationCheck,
};

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
pub mod ext_linear;
pub mod germ;
pub mod koala_ring;
pub mod orbweaver_opening;

pub use aadp::{
    aadp_encrypt_bytes, aadp_encrypt_scalar, aadp_encrypt_u128, AadpByteCiphertext, AadpCiphertext,
    AadpConstraintSystem, AadpField, AadpLinearForm, AadpMulConstraint, Sp1AadpField,
};
pub use bundle::{
    GermArmCapsule, GermResidualPlan, GermVerifierStage, LinearResidualDescriptor,
    MultiplicativeResidualDescriptor, Sp1GermBundle, Sp1GermProofObject, Sp1LinProof, Sp1LinTerm,
    Sp1MulSumcheckProof, Sp1MulSumcheckRound, Sp1MulTerm, Sp1MulTerminalOpeningProofs,
    TranscriptBoundSp1GermProofObject,
};
pub use ext_linear::{
    apply_sp1_extension_mul_matrix, flatten_extension_limbs,
    linearize_extension_weights_to_base_forms, sp1_extension_mul_matrix,
};
pub use germ::{
    arm_germ_aadp_template, arm_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings,
    bind_bundle,
    bind_bundle_to_capsule, bind_bundle_to_capsule_with_orbweaver_terminal_openings,
    bind_bundle_with_orbweaver_terminal_openings, build_orbweaver_terminal_openings,
    build_orbweaver_terminal_openings_from_capsule, compile_germ_aadp_template,
    compile_transcript_bound_germ_aadp_template, compute_commitment_root, derive_challenges,
    compile_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings,
    derive_challenges_from_arming_digest, derive_challenges_from_capsule,
    materialize_transcript_bound_germ_aadp_witness,
    materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings, verify_lin,
    verify_mul, verify_mul_with_orbweaver_terminal_openings,
    verify_orbweaver_terminal_openings_from_capsule, verify_transcript_bound_sp1_germ_proof_object,
    verify_transcript_bound_sp1_germ_proof_object_with_orbweaver_terminal_openings,
    ArmedGermAadpCiphertext, GermAadpConstraintStats, GermAadpVerifierTemplate, GermAadpWitness,
    GermAadpWitnessLayout, GermChallenges, GermError, GermPublicValues, GermRelationCheck,
};
pub use koala_ring::{KoalaRing64, KOALA_RING64_DIM};
pub use orbweaver_opening::{
    aggregate_scalar_image_form, aggregate_scalar_image_value,
    build_aggregated_scalar_image_openings_from_mul_terms, decode_opening_proof,
    decode_scalar_opening_proof, decode_srs, decode_terminal_value_proof, digest_srs,
    encode_opening_proof, encode_scalar_opening_proof, encode_srs, encode_terminal_value_proof,
    extension_terminal_openings_l_inf_u32, generate_local_dev_srs, preverify_dense,
    preverify_factorized_mle, read_srs_from_file, scalar_ring_element_to_base_field,
    srs_max_supported_width, terminal_scalar_image_forms, terminal_scalar_image_values,
    validate_srs, verify_aggregated_scalar_image_openings_from_mul_terms,
    verify_opening_equation, write_srs_to_file, OrbweaverOpeningProof, OrbweaverOpeningSrs,
    OrbweaverOpeningVerificationKey, OrbweaverScalarOpeningProof,
};

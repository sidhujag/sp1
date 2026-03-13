//! SP1-native GERM relation checks.
//!
//! This module follows the "two global vanishing checks" shape:
//! - `V_lin(r_lin) = 0` for globally mixed linear residuals
//! - `V_mul(r_mul) = 0` for globally mixed multiplicative residuals
//!
//! `r_lin` and `r_mul` are designated points derived from one commitment-bound
//! algebraic seed `theta = Pi(D || C)` where `D` is arming context and `C` is
//! the transcript commitment coordinates.

use rand::RngCore;
use sha2::{Digest, Sha256};
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use sp1_primitives::{SP1ExtensionField, SP1Field};
use std::collections::BTreeMap;

use crate::aadp::{
    aadp_encrypt_scalar, AadpCiphertext, AadpConstraintSystem, AadpLinearForm, AadpMulConstraint,
};
use crate::ajtai::{
    derive_package_ajtai_seed, package_ajtai_commitment, package_ajtai_matrix_entry,
    PACKAGE_AJTAI_RING_DIM, PACKAGE_AJTAI_ROWS,
};
use crate::bundle::{
    GermArmCapsule, GermResidualPlan, GermVerifierStage, LinearResidualDescriptor,
    MultiplicativeResidualDescriptor, Sp1GermBundle, Sp1LinProof, Sp1LinTerm, Sp1MulSumcheckProof,
    Sp1MulSumcheckRound, Sp1MulTerm, Sp1MulTerminalOpeningProofs, Sp1PackageCommitment,
    TranscriptBoundSp1GermProofObject,
};
use crate::koala_ring::{KoalaRing64, KOALA_RING64_DIM};
use crate::orbweaver_opening::{
    aggregate_scalar_image_form, aggregate_scalar_image_value,
    build_aggregated_scalar_image_openings_from_mul_terms, decode_scalar_opening_proof, digest_srs,
    scalar_ring_element_to_base_field, terminal_scalar_image_forms, terminal_scalar_image_values,
    validate_srs, verify_aggregated_scalar_image_openings_from_mul_terms, OrbweaverOpeningSrs,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GermChallenges {
    /// Designated point for linear vanishing polynomial evaluation.
    pub r_lin: SP1ExtensionField,
    /// Designated point for multiplicative vanishing polynomial evaluation.
    pub r_mul: SP1ExtensionField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GermRelationCheck {
    /// Evaluated vanishing value (`V_lin(r_lin)` or `V_mul(r_mul)`).
    pub folded_residual: SP1ExtensionField,
    /// Root/context/challenge/transcript binding fingerprint.
    pub fingerprint: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GermPublicValues {
    pub statement_digest: [u8; 32],
    pub descriptor_digest: [u8; 32],
    pub verifier_shape_digest: [u8; 32],
    pub share_index: u32,
    pub share_domain_separator: [u8; 32],
}

impl GermPublicValues {
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"sp1-germ/public-values/v1");
        h.update(self.statement_digest);
        h.update(self.descriptor_digest);
        h.update(self.verifier_shape_digest);
        h.update(self.share_index.to_le_bytes());
        h.update(self.share_domain_separator);
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }

    #[must_use]
    pub fn arm_capsule(&self, residual_plan: &GermResidualPlan) -> GermArmCapsule {
        GermArmCapsule {
            public_values_digest: self.digest(),
            schedule_descriptor_digest: residual_plan.schedule_descriptor_digest,
            residual_plan_digest: residual_plan.digest(),
            verifier_stage: residual_plan.verifier_stage,
            sumcheck_rounds: residual_plan.sumcheck_rounds,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GermAadpConstraintStats {
    pub linear_round_checks: usize,
    pub opening_checks: usize,
    pub multiplication_gates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiplicativeTermField {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentWitnessSlot {
    LinearCoefficient { term_idx: usize },
    LinearValue { term_idx: usize },
    MultiplicativeField { term_idx: usize, field: MultiplicativeTermField },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AffineWitnessExpr {
    var_idx: Option<usize>,
    scale: SP1ExtensionField,
    constant: SP1ExtensionField,
}

impl AffineWitnessExpr {
    #[must_use]
    fn constant(constant: SP1ExtensionField) -> Self {
        Self { var_idx: None, scale: ext_zero(), constant }
    }

    #[must_use]
    fn variable(var_idx: usize) -> Self {
        Self { var_idx: Some(var_idx), scale: ext_one(), constant: ext_zero() }
    }
}

fn add_scaled_affine_expr(
    form: &mut AadpLinearForm<SP1ExtensionField>,
    expr: AffineWitnessExpr,
    scale: SP1ExtensionField,
) {
    form.constant += scale * expr.constant;
    if let Some(var_idx) = expr.var_idx {
        form.terms.push((var_idx, scale * expr.scale));
    }
}

const MUL_SUMCHECK_ROUND_CHALLENGE_DOMAIN: &[u8] = b"sp1-germ/sumcheck-round-challenge/v3";
const MUL_SUMCHECK_ROUND_STATE_DOMAIN: &[u8] = b"sp1-germ/sumcheck-round-state/v1";
const ORBWEAVER_AGGREGATED_SCALAR_OPENINGS: usize = 4;
const ORBWEAVER_SCALAR_IMAGE_COUNT: usize = 16;
const ORBWEAVER_RING_COEFFS_PER_EXTENSION_BLOCK: usize = 4;
const ORBWEAVER_JL_TOTAL_PROJECTIONS: usize = 8;
const ORBWEAVER_JL_NORM_BITS: usize = 31;

#[derive(Debug, Clone)]
struct OrbweaverTranscriptTemplateData {
    aggregated_vk_values: [SP1Field; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS],
    aggregated_output_multipliers:
        [[SP1ExtensionField; 4]; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS],
    c_flat_field_multipliers: Vec<[SP1ExtensionField; 4]>,
    lhs_block_multipliers: Vec<SP1ExtensionField>,
    jl_rows: Vec<Vec<i8>>,
    proof_commitment: Sp1PackageCommitment,
    proof_pi_len: usize,
}

#[derive(Debug, Clone)]
struct CommitmentBindingLayout {
    commitment_slots: Vec<CommitmentWitnessSlot>,
    commitment_forms: Vec<AadpLinearForm<SP1ExtensionField>>,
    expected_linear_terms: usize,
    expected_mul_terms: usize,
    linear_coefficient_indices: Vec<Option<usize>>,
    linear_value_indices: Vec<usize>,
    multiplicative_field_exprs: Vec<[AffineWitnessExpr; 4]>,
}

#[derive(Debug, Clone)]
pub struct GermAadpWitnessLayout {
    pub sumcheck_rounds: usize,
    expected_linear_terms: usize,
    expected_mul_terms: usize,
    commitment_slots: Vec<CommitmentWitnessSlot>,
    linear_term_has_explicit_coefficient: Vec<bool>,
    orbweaver_terminal_pi_len: usize,
    orbweaver_aggregated_proof_count: usize,
}

#[derive(Debug, Clone)]
pub struct GermAadpVerifierTemplate {
    pub cs: AadpConstraintSystem<SP1ExtensionField>,
    pub layout: GermAadpWitnessLayout,
    pub stats: GermAadpConstraintStats,
    pub capsule_digest: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct GermAadpWitness {
    pub witness: Vec<SP1ExtensionField>,
}

impl GermAadpWitness {
    #[must_use]
    pub fn as_slice(&self) -> &[SP1ExtensionField] {
        self.witness.as_slice()
    }
}

#[derive(Debug, Clone)]
pub struct ArmedGermAadpCiphertext {
    pub template: GermAadpVerifierTemplate,
    pub ciphertext: AadpCiphertext<SP1ExtensionField>,
    pub capsule_digest: [u8; 32],
}

impl GermAadpVerifierTemplate {
    pub fn check_witness(&self, witness: &GermAadpWitness) -> Result<(), GermError> {
        self.cs.check_witness(witness.as_slice()).map_err(GermError::AadpWitnessRejected)
    }

    pub fn decap_checked(
        &self,
        ciphertext: &AadpCiphertext<SP1ExtensionField>,
        witness: &GermAadpWitness,
    ) -> Result<SP1ExtensionField, GermError> {
        self.check_witness(witness)?;
        ciphertext.decrypt_scalar(witness.as_slice()).map_err(GermError::AadpDecryptFailed)
    }
}

impl ArmedGermAadpCiphertext {
    pub fn decap_checked(&self, witness: &GermAadpWitness) -> Result<SP1ExtensionField, GermError> {
        self.template.decap_checked(&self.ciphertext, witness)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GermError {
    CommitmentRootMismatch,
    TooManyLinearTerms(usize),
    EmptyProof(&'static str),
    EmptyTerms(&'static str),
    NonZeroResidual(&'static str),
    BindingTagMismatch(&'static str),
    MalformedLinProof,
    LinProofTranscriptMismatch,
    MalformedMulProof,
    SumcheckRoundsMismatch { got: usize, expected: usize },
    SumcheckIdentityFailed(usize),
    SumcheckTranscriptMismatch,
    SumcheckFinalResidualNonZero,
    SumcheckInterpolationDenominatorZero,
    InvalidResidualPlan(String),
    TranscriptShapeMismatch { which: &'static str, got: usize, expected: usize },
    InvalidOrbweaverSrs(String),
    MissingMulTerminalOpening(&'static str),
    MalformedMulTerminalOpening { which: &'static str, msg: String },
    MulTerminalOpeningMismatch(&'static str),
    MulTerminalOpeningFailed { which: &'static str, msg: String },
    AadpConstraintUnsatisfied(String),
    TemplateCapsuleMismatch,
    AadpEncryptFailed(String),
    AadpWitnessRejected(String),
    AadpDecryptFailed(String),
}

impl core::fmt::Display for GermError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CommitmentRootMismatch => write!(f, "commitment root mismatch"),
            Self::TooManyLinearTerms(count) => {
                write!(f, "linear term count does not fit in u32: {count}")
            }
            Self::EmptyProof(which) => write!(f, "empty proof bytes for {which}"),
            Self::EmptyTerms(which) => write!(f, "empty relation terms for {which}"),
            Self::NonZeroResidual(which) => write!(f, "non-zero folded residual for {which}"),
            Self::BindingTagMismatch(which) => write!(f, "binding tag mismatch for {which}"),
            Self::MalformedLinProof => write!(f, "malformed linear proof bytes"),
            Self::LinProofTranscriptMismatch => write!(f, "linear proof transcript mismatch"),
            Self::MalformedMulProof => write!(f, "malformed multiplicative proof bytes"),
            Self::SumcheckRoundsMismatch { got, expected } => {
                write!(f, "sumcheck rounds mismatch: got={got} expected={expected}")
            }
            Self::SumcheckIdentityFailed(round_idx) => {
                write!(f, "sumcheck round identity failed at round {round_idx}")
            }
            Self::SumcheckTranscriptMismatch => write!(f, "sumcheck transcript mismatch"),
            Self::SumcheckFinalResidualNonZero => {
                write!(f, "final sumcheck opening residual non-zero")
            }
            Self::SumcheckInterpolationDenominatorZero => {
                write!(f, "sumcheck interpolation denominator was zero")
            }
            Self::InvalidResidualPlan(msg) => write!(f, "invalid residual plan: {msg}"),
            Self::TranscriptShapeMismatch { which, got, expected } => {
                write!(f, "transcript shape mismatch for {which}: got={got} expected={expected}")
            }
            Self::InvalidOrbweaverSrs(msg) => write!(f, "invalid Orbweaver SRS: {msg}"),
            Self::MissingMulTerminalOpening(which) => {
                write!(f, "missing multiplicative terminal opening proof for {which}")
            }
            Self::MalformedMulTerminalOpening { which, msg } => {
                write!(f, "malformed multiplicative terminal opening proof for {which}: {msg}")
            }
            Self::MulTerminalOpeningMismatch(which) => {
                write!(f, "multiplicative terminal opening value mismatch for {which}")
            }
            Self::MulTerminalOpeningFailed { which, msg } => {
                write!(f, "multiplicative terminal opening verification failed for {which}: {msg}")
            }
            Self::AadpConstraintUnsatisfied(msg) => {
                write!(f, "compiled AADP verifier witness does not satisfy constraints: {msg}")
            }
            Self::TemplateCapsuleMismatch => write!(f, "AADP verifier template capsule mismatch"),
            Self::AadpEncryptFailed(msg) => write!(f, "AADP encryption failed: {msg}"),
            Self::AadpWitnessRejected(msg) => write!(f, "AADP witness rejected: {msg}"),
            Self::AadpDecryptFailed(msg) => write!(f, "AADP decryption failed: {msg}"),
        }
    }
}

impl std::error::Error for GermError {}

#[must_use]
pub fn compute_commitment_root(package_commitment: &Sp1PackageCommitment) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/commitment/v1");
    for row in package_commitment {
        for coeff in row {
            hash_extension(&mut h, coeff);
        }
    }
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn compute_transcript_commitment(
    lin_terms: &[Sp1LinTerm],
    mul_terms: &[Sp1MulTerm],
) -> Sp1PackageCommitment {
    let seed = derive_package_ajtai_seed();
    let msg = transcript_commitment_message(lin_terms, mul_terms);
    package_ajtai_commitment(&seed, msg.as_slice())
}

fn transcript_commitment_message(
    lin_terms: &[Sp1LinTerm],
    mul_terms: &[Sp1MulTerm],
) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(2 + (lin_terms.len() * 2) + (mul_terms.len() * 4));
    out.push(domain_tag_extension(b"sp1-germ/transcript/lin/v1"));
    for term in lin_terms {
        out.push(term.coefficient);
        out.push(term.value);
    }
    out.push(domain_tag_extension(b"sp1-germ/transcript/mul/v1"));
    for term in mul_terms {
        out.push(term.a);
        out.push(term.b);
        out.push(term.c);
        out.push(term.d);
    }
    out
}

fn domain_tag_extension(domain: &[u8]) -> SP1ExtensionField {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/transcript-tag/v1");
    h.update(domain);
    let digest: [u8; 32] = h.finalize().into();
    bytes_to_extension(&digest)
}

fn hash_extension_value(h: &mut Sha256, value: &SP1ExtensionField) {
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value) {
        h.update(limb.as_canonical_u32().to_le_bytes());
    }
}

fn package_commitment_mix(commitment: &Sp1PackageCommitment) -> SP1ExtensionField {
    let mut commitment_mix = ext_zero();
    let mut coord_idx = 0usize;
    for row in commitment {
        for coord in row {
            commitment_mix += commitment_mix_weight(coord_idx) * *coord;
            coord_idx += 1;
        }
    }
    commitment_mix
}

fn first_limb_functional_multiplier(weights: [SP1Field; 4]) -> SP1ExtensionField {
    let w_inv = SP1Field::from_canonical_u32(3)
        .try_inverse()
        .expect("SP1 extension binomial constant must be invertible");
    SP1ExtensionField::from_base_slice(&[
        weights[0],
        weights[3] * w_inv,
        weights[2] * w_inv,
        weights[1] * w_inv,
    ])
}

fn pack_scalar_weights_to_extension_multipliers(weights: &[SP1Field]) -> Vec<SP1ExtensionField> {
    assert_eq!(
        weights.len() % ORBWEAVER_RING_COEFFS_PER_EXTENSION_BLOCK,
        0,
        "orbweaver packed weights must align to extension blocks"
    );
    weights
        .chunks_exact(ORBWEAVER_RING_COEFFS_PER_EXTENSION_BLOCK)
        .map(|chunk| {
            first_limb_functional_multiplier(
                chunk.try_into().expect("orbweaver extension block has 4 coefficients"),
            )
        })
        .collect()
}

fn ring_mul_coeff0_weights(multiplier: &KoalaRing64) -> [SP1Field; KOALA_RING64_DIM] {
    let coeffs = multiplier.coeffs();
    core::array::from_fn(|idx| {
        if idx == 0 {
            coeffs[0]
        } else {
            -coeffs[KOALA_RING64_DIM - idx]
        }
    })
}

fn packed_scalar_opening_blocks(proof: &crate::orbweaver_opening::OrbweaverScalarOpeningProof) -> Vec<SP1ExtensionField> {
    let mut out =
        Vec::with_capacity(proof.pi0.len() * (KOALA_RING64_DIM / ORBWEAVER_RING_COEFFS_PER_EXTENSION_BLOCK));
    for ring in &proof.pi0 {
        out.extend(
            ring.coeffs()
                .chunks_exact(ORBWEAVER_RING_COEFFS_PER_EXTENSION_BLOCK)
                .map(|chunk| SP1ExtensionField::from_base_slice(
                    chunk.try_into().expect("orbweaver packed proof block has 4 coefficients"),
                )),
        );
    }
    out
}

fn packed_scalar_opening_family_blocks(
    proofs: &[crate::orbweaver_opening::OrbweaverScalarOpeningProof; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS],
) -> Vec<SP1ExtensionField> {
    let mut out = Vec::new();
    for proof in proofs {
        out.extend(packed_scalar_opening_blocks(proof));
    }
    out
}

fn orbweaver_output_field_multipliers(
    aggregation_coeffs: &[SP1Field; ORBWEAVER_SCALAR_IMAGE_COUNT],
) -> [SP1ExtensionField; 4] {
    core::array::from_fn(|field_idx| {
        let start = field_idx * 4;
        first_limb_functional_multiplier(
            aggregation_coeffs[start..start + 4]
                .try_into()
                .expect("orbweaver output multiplier slice has 4 coefficients"),
        )
    })
}

fn orbweaver_proof_commitment_message(
    proofs: &[crate::orbweaver_opening::OrbweaverScalarOpeningProof; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS],
) -> Vec<SP1ExtensionField> {
    let mut out = Vec::new();
    out.push(domain_tag_extension(b"sp1-germ/orbweaver/proof-commit/v1"));
    out.extend(packed_scalar_opening_family_blocks(proofs));
    out
}

fn compute_orbweaver_proof_commitment(
    proofs: &[crate::orbweaver_opening::OrbweaverScalarOpeningProof; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS],
) -> Sp1PackageCommitment {
    let seed = derive_package_ajtai_seed();
    let msg = orbweaver_proof_commitment_message(proofs);
    package_ajtai_commitment(&seed, msg.as_slice())
}

fn derive_orbweaver_aggregation_coeffs(
    arming_digest: &[u8; 32],
    transcript_commitment: &Sp1PackageCommitment,
    pi_mul_bytes: &[u8],
) -> [[SP1Field; ORBWEAVER_SCALAR_IMAGE_COUNT]; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS] {
    let challenge_seed = derive_commitment_bound_seed(arming_digest, transcript_commitment);
    core::array::from_fn(|agg_idx| {
        let mut coeffs = core::array::from_fn(|eq_idx| {
            let mut h = Sha256::new();
            h.update(b"sp1-germ/orbweaver-aggregation/v1");
            hash_extension_value(&mut h, &challenge_seed);
            h.update((agg_idx as u32).to_le_bytes());
            h.update((eq_idx as u32).to_le_bytes());
            h.update((pi_mul_bytes.len() as u64).to_le_bytes());
            h.update(pi_mul_bytes);
            let digest: [u8; 32] = h.finalize().into();
            SP1Field::from_wrapped_u32(u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]))
        });
        if coeffs.iter().all(|coeff| coeff.is_zero()) {
            coeffs[agg_idx] = SP1Field::one();
        }
        coeffs
    })
}

fn derive_orbweaver_jl_seed(
    arming_digest: &[u8; 32],
    transcript_commitment: &Sp1PackageCommitment,
    proof_commitment: &Sp1PackageCommitment,
    srs: &OrbweaverOpeningSrs,
) -> [u8; 32] {
    let challenge_seed = derive_commitment_bound_seed(arming_digest, transcript_commitment);
    let proof_mix = package_commitment_mix(proof_commitment);
    let jl_seed = derive_seed_from_challenge_seed(
        b"sp1-germ/orbweaver-jl-seed/v2",
        challenge_seed,
        &[bytes_to_extension(&digest_srs(srs)), proof_mix],
        0,
    );
    let mut h = Sha256::new();
    h.update(b"sp1-germ/orbweaver-jl-seed-bytes/v1");
    hash_extension_value(&mut h, &jl_seed);
    h.finalize().into()
}

fn orbweaver_c_flat_field_multipliers(
    srs: &OrbweaverOpeningSrs,
    mul_terms: usize,
) -> Result<Vec<[SP1ExtensionField; 4]>, GermError> {
    let v = scalar_ring_element_to_base_field(&srs.v).map_err(GermError::InvalidOrbweaverSrs)?;
    let mut cur = v;
    let mut out = Vec::with_capacity(mul_terms);
    for _ in 0..mul_terms {
        let field_weights = core::array::from_fn(|_| {
            let weights = [cur, cur * v, cur * v * v, cur * v * v * v];
            cur *= v * v * v * v;
            first_limb_functional_multiplier(weights)
        });
        out.push(field_weights);
    }
    Ok(out)
}

fn orbweaver_lhs_block_multipliers(srs: &OrbweaverOpeningSrs) -> Vec<SP1ExtensionField> {
    let mut out =
        Vec::with_capacity(srs.a0.len() * (KOALA_RING64_DIM / ORBWEAVER_RING_COEFFS_PER_EXTENSION_BLOCK));
    for ring in &srs.a0 {
        out.extend(pack_scalar_weights_to_extension_multipliers(
            ring_mul_coeff0_weights(ring).as_slice(),
        ));
    }
    out
}

fn decode_orbweaver_aggregated_scalar_openings(
    openings: &Sp1MulTerminalOpeningProofs,
) -> Result<
    [crate::orbweaver_opening::OrbweaverScalarOpeningProof; ORBWEAVER_AGGREGATED_SCALAR_OPENINGS],
    GermError,
> {
    Ok([
        decode_scalar_opening_proof(openings.a.as_slice())
            .map_err(|msg| GermError::MalformedMulTerminalOpening { which: "agg0", msg })?,
        decode_scalar_opening_proof(openings.b.as_slice())
            .map_err(|msg| GermError::MalformedMulTerminalOpening { which: "agg1", msg })?,
        decode_scalar_opening_proof(openings.c.as_slice())
            .map_err(|msg| GermError::MalformedMulTerminalOpening { which: "agg2", msg })?,
        decode_scalar_opening_proof(openings.d.as_slice())
            .map_err(|msg| GermError::MalformedMulTerminalOpening { which: "agg3", msg })?,
    ])
}

fn expected_orbweaver_scalar_image_values(mul_proof: &Sp1MulSumcheckProof) -> [SP1Field; ORBWEAVER_SCALAR_IMAGE_COUNT] {
    terminal_scalar_image_values(&[
        mul_proof.opening.a,
        mul_proof.opening.b,
        mul_proof.opening.c,
        mul_proof.opening.d,
    ])
}

fn build_orbweaver_transcript_template_data(
    capsule: &GermArmCapsule,
    proof_object: &Sp1GermBundle,
    srs: &OrbweaverOpeningSrs,
) -> Result<OrbweaverTranscriptTemplateData, GermError> {
    let mul_proof = decode_mul_sumcheck(&proof_object.pi_mul)?;
    let arming_digest = capsule.digest();
    let challenges = derive_challenges_from_capsule(capsule, &proof_object.shared_object_commitment);
    let sumcheck_seed =
        derive_mul_sumcheck_seed(&arming_digest, &proof_object.shared_object_commitment, &challenges.r_mul);
    let opening_weights =
        eq_table(collect_sumcheck_round_challenges(&sumcheck_seed, &mul_proof)?.as_slice());
    let aggregation_coeffs = derive_orbweaver_aggregation_coeffs(
        &arming_digest,
        &proof_object.shared_object_commitment,
        proof_object.pi_mul.as_slice(),
    );
    let forms = terminal_scalar_image_forms(opening_weights.as_slice());
    let witness_width = proof_object
        .mul_terms
        .len()
        .checked_mul(ORBWEAVER_SCALAR_IMAGE_COUNT)
        .ok_or_else(|| {
            GermError::InvalidOrbweaverSrs("orbweaver witness width overflow".to_string())
        })?;
    let v_inv_powers = crate::orbweaver_opening::derive_negative_powers(srs, witness_width)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let aggregated_vk_values = core::array::from_fn(|agg_idx| {
        let aggregated_form = aggregate_scalar_image_form(&forms, &aggregation_coeffs[agg_idx])
            .expect("orbweaver aggregated forms must align");
        let vk =
            crate::orbweaver_opening::preverify_dense(v_inv_powers.as_slice(), aggregated_form.as_slice())
                .expect("orbweaver dense preverify should succeed");
        scalar_ring_element_to_base_field(&vk.value)
            .expect("orbweaver aggregated verifier key must remain scalar-subring")
    });
    let aggregated_output_multipliers =
        core::array::from_fn(|agg_idx| orbweaver_output_field_multipliers(&aggregation_coeffs[agg_idx]));
    let proofs = decode_orbweaver_aggregated_scalar_openings(&proof_object.pi_mul_terminal_openings)?;
    let proof_pi_len = proofs[0].pi0.len();
    if proofs.iter().any(|proof| proof.pi0.len() != proof_pi_len) {
        return Err(GermError::TranscriptShapeMismatch {
            which: "orbweaver aggregated proof length",
            got: proofs.iter().map(|proof| proof.pi0.len()).max().unwrap_or(0),
            expected: proof_pi_len,
        });
    }
    let proof_commitment = compute_orbweaver_proof_commitment(&proofs);
    let jl_seed = derive_orbweaver_jl_seed(
        &arming_digest,
        &proof_object.shared_object_commitment,
        &proof_commitment,
        srs,
    );
    let jl_rows = derive_orbweaver_jl_rows(
        &jl_seed,
        ORBWEAVER_AGGREGATED_SCALAR_OPENINGS * proof_pi_len * KOALA_RING64_DIM,
    );
    Ok(OrbweaverTranscriptTemplateData {
        aggregated_vk_values,
        aggregated_output_multipliers,
        c_flat_field_multipliers: orbweaver_c_flat_field_multipliers(srs, proof_object.mul_terms.len())?,
        lhs_block_multipliers: orbweaver_lhs_block_multipliers(srs),
        jl_rows,
        proof_commitment,
        proof_pi_len,
    })
}

fn linear_descriptor_term_count(descriptor: &LinearResidualDescriptor) -> usize {
    match descriptor {
        LinearResidualDescriptor::PublicValuesPadding { count, .. } => *count,
        _ => 1,
    }
}

fn orbweaver_witness_width_from_mul_terms(mul_terms: usize) -> Result<usize, GermError> {
    mul_terms.checked_mul(ORBWEAVER_SCALAR_IMAGE_COUNT).ok_or_else(|| {
        GermError::InvalidOrbweaverSrs("orbweaver witness width overflow".to_string())
    })
}

fn expected_mul_term_count(plan: &GermResidualPlan) -> Result<usize, GermError> {
    let target = 1usize.checked_shl(u32::from(plan.sumcheck_rounds)).ok_or_else(|| {
        GermError::InvalidResidualPlan(format!(
            "sumcheck_rounds too large for usize shift: {}",
            plan.sumcheck_rounds
        ))
    })?;
    if plan.multiplicative_descriptors.len() > target {
        return Err(GermError::InvalidResidualPlan(format!(
            "descriptor prefix exceeds sumcheck capacity: descriptors={} capacity={target}",
            plan.multiplicative_descriptors.len()
        )));
    }
    Ok(target)
}

fn validate_residual_plan_for_template(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
) -> Result<(), GermError> {
    if !residual_plan.has_valid_descriptor_digest() {
        return Err(GermError::InvalidResidualPlan(
            "residual_plan_digest does not match descriptor schedule".to_string(),
        ));
    }
    if capsule.schedule_descriptor_digest != residual_plan.schedule_descriptor_digest
        || capsule.residual_plan_digest != residual_plan.digest()
        || !matches!(
            (capsule.verifier_stage, residual_plan.verifier_stage),
            (GermVerifierStage::Compressed, GermVerifierStage::Compressed)
        )
        || capsule.sumcheck_rounds != residual_plan.sumcheck_rounds
    {
        return Err(GermError::TemplateCapsuleMismatch);
    }
    let _ = expected_mul_term_count(residual_plan)?;
    Ok(())
}

fn add_commitment_message_affine_entry(
    forms: &mut [AadpLinearForm<SP1ExtensionField>],
    seed: &[u8; 32],
    column_idx: usize,
    var_idx: Option<usize>,
    var_scale: SP1ExtensionField,
    constant: SP1ExtensionField,
) {
    for row in 0..PACKAGE_AJTAI_ROWS {
        let coeffs = package_ajtai_matrix_entry(seed, row, column_idx);
        for coeff_idx in 0..PACKAGE_AJTAI_RING_DIM {
            let form_idx = row * PACKAGE_AJTAI_RING_DIM + coeff_idx;
            let matrix_coeff = coeffs[coeff_idx];
            if let Some(var_idx) = var_idx {
                forms[form_idx].terms.push((var_idx, -(matrix_coeff * var_scale)));
            }
            forms[form_idx].constant -= matrix_coeff * constant;
        }
    }
}

fn build_commitment_binding_layout(
    alloc: &mut impl FnMut() -> usize,
    commitment_coord_indices: &[usize; PACKAGE_AJTAI_ROWS * PACKAGE_AJTAI_RING_DIM],
    residual_plan: &GermResidualPlan,
) -> Result<CommitmentBindingLayout, GermError> {
    let seed = derive_package_ajtai_seed();
    let mut forms = commitment_coord_indices
        .iter()
        .map(|coord_idx| AadpLinearForm {
            constant: ext_zero(),
            terms: vec![(*coord_idx, ext_one())],
        })
        .collect::<Vec<_>>();
    let mut commitment_slots = Vec::new();
    let mut degree_bit_slots = BTreeMap::<(String, usize), usize>::new();
    let mut linear_coefficient_indices = Vec::new();
    let mut linear_value_indices = Vec::new();
    let mut multiplicative_field_exprs = Vec::new();
    let mut column_idx = 0usize;

    add_commitment_message_affine_entry(
        forms.as_mut_slice(),
        &seed,
        column_idx,
        None,
        ext_zero(),
        domain_tag_extension(b"sp1-germ/transcript/lin/v1"),
    );
    column_idx += 1;

    let mut linear_term_idx = 0usize;
    for descriptor in &residual_plan.linear_descriptors {
        for _ in 0..linear_descriptor_term_count(descriptor) {
            let coeff_var_idx = match descriptor {
                LinearResidualDescriptor::Explicit => {
                    let coeff_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::LinearCoefficient {
                        term_idx: linear_term_idx,
                    });
                    add_commitment_message_affine_entry(
                        forms.as_mut_slice(),
                        &seed,
                        column_idx,
                        Some(coeff_idx),
                        ext_one(),
                        ext_zero(),
                    );
                    Some(coeff_idx)
                }
                _ => {
                    add_commitment_message_affine_entry(
                        forms.as_mut_slice(),
                        &seed,
                        column_idx,
                        None,
                        ext_zero(),
                        ext_one(),
                    );
                    None
                }
            };
            linear_coefficient_indices.push(coeff_var_idx);
            column_idx += 1;

            let value_idx = alloc();
            linear_value_indices.push(value_idx);
            match descriptor {
                _ => {
                    commitment_slots
                        .push(CommitmentWitnessSlot::LinearValue { term_idx: linear_term_idx });
                    add_commitment_message_affine_entry(
                        forms.as_mut_slice(),
                        &seed,
                        column_idx,
                        Some(value_idx),
                        ext_one(),
                        ext_zero(),
                    );
                }
            }
            column_idx += 1;
            linear_term_idx += 1;
        }
    }

    add_commitment_message_affine_entry(
        forms.as_mut_slice(),
        &seed,
        column_idx,
        None,
        ext_zero(),
        domain_tag_extension(b"sp1-germ/transcript/mul/v1"),
    );
    column_idx += 1;

    let mut mul_term_idx = 0usize;
    for descriptor in &residual_plan.multiplicative_descriptors {
        match descriptor {
            MultiplicativeResidualDescriptor::Explicit => {
                let a_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::A,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(a_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                let b_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::B,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(b_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                let c_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::C,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(c_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                let d_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::D,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(d_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                multiplicative_field_exprs.push([
                    AffineWitnessExpr::variable(a_idx),
                    AffineWitnessExpr::variable(b_idx),
                    AffineWitnessExpr::variable(c_idx),
                    AffineWitnessExpr::variable(d_idx),
                ]);
            }
            MultiplicativeResidualDescriptor::DegreeBitBooleanity { .. } => {
                let (chip_name, bit_number) = match descriptor {
                    MultiplicativeResidualDescriptor::DegreeBitBooleanity {
                        chip_name,
                        bit_idx,
                    } => (chip_name.clone(), *bit_idx),
                    _ => unreachable!(),
                };
                let bit_idx = if let Some(existing_idx) =
                    degree_bit_slots.get(&(chip_name.clone(), bit_number))
                {
                    *existing_idx
                } else {
                    let fresh_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                        term_idx: mul_term_idx,
                        field: MultiplicativeTermField::A,
                    });
                    degree_bit_slots.insert((chip_name, bit_number), fresh_idx);
                    fresh_idx
                };
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(bit_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(bit_idx),
                    ext_one(),
                    -ext_one(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;

                multiplicative_field_exprs.push([
                    AffineWitnessExpr::variable(bit_idx),
                    AffineWitnessExpr {
                        var_idx: Some(bit_idx),
                        scale: ext_one(),
                        constant: -ext_one(),
                    },
                    AffineWitnessExpr::constant(ext_zero()),
                    AffineWitnessExpr::constant(ext_one()),
                ]);
            }
            MultiplicativeResidualDescriptor::DegreeHeightProduct { .. } => {
                let (chip_name, bit_number) = match descriptor {
                    MultiplicativeResidualDescriptor::DegreeHeightProduct {
                        chip_name,
                        bit_idx,
                    } => (chip_name.clone(), *bit_idx),
                    _ => unreachable!(),
                };
                let field_a_idx = if let Some(existing_idx) =
                    degree_bit_slots.get(&(chip_name.clone(), bit_number))
                {
                    *existing_idx
                } else {
                    let fresh_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                        term_idx: mul_term_idx,
                        field: MultiplicativeTermField::A,
                    });
                    degree_bit_slots.insert((chip_name.clone(), bit_number), fresh_idx);
                    fresh_idx
                };
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(field_a_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
                let field_b_idx = if let Some(existing_idx) =
                    degree_bit_slots.get(&(chip_name.clone(), 0usize))
                {
                    *existing_idx
                } else {
                    let fresh_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                        term_idx: mul_term_idx,
                        field: MultiplicativeTermField::B,
                    });
                    degree_bit_slots.insert((chip_name, 0usize), fresh_idx);
                    fresh_idx
                };
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(field_b_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;

                multiplicative_field_exprs.push([
                    AffineWitnessExpr::variable(field_a_idx),
                    AffineWitnessExpr::variable(field_b_idx),
                    AffineWitnessExpr::constant(ext_zero()),
                    AffineWitnessExpr::constant(ext_one()),
                ]);
            }
            MultiplicativeResidualDescriptor::GkrPowWitness
            | MultiplicativeResidualDescriptor::GkrCumulativeSum
            | MultiplicativeResidualDescriptor::GkrRoundClaimedSum { .. }
            | MultiplicativeResidualDescriptor::GkrTracePointCoord { .. }
            | MultiplicativeResidualDescriptor::GkrFinalNumeratorEval
            | MultiplicativeResidualDescriptor::GkrFinalDenominatorEval => {
                let a_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::A,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(a_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;

                multiplicative_field_exprs.push([
                    AffineWitnessExpr::variable(a_idx),
                    AffineWitnessExpr::constant(ext_one()),
                    AffineWitnessExpr::constant(ext_zero()),
                    AffineWitnessExpr::constant(ext_one()),
                ]);
            }
            MultiplicativeResidualDescriptor::GkrDenominatorInverse { .. } => {
                let denominator_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::A,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(denominator_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                let inverse_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::B,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(inverse_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;

                multiplicative_field_exprs.push([
                    AffineWitnessExpr::variable(denominator_idx),
                    AffineWitnessExpr::variable(inverse_idx),
                    AffineWitnessExpr::constant(ext_one()),
                    AffineWitnessExpr::constant(ext_one()),
                ]);
            }
            MultiplicativeResidualDescriptor::GkrRoundFinalEval { .. } => {
                let eq_eval_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::A,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(eq_eval_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                let combined_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::B,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(combined_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;

                let final_eval_idx = alloc();
                commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                    term_idx: mul_term_idx,
                    field: MultiplicativeTermField::C,
                });
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(final_eval_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;

                multiplicative_field_exprs.push([
                    AffineWitnessExpr::variable(eq_eval_idx),
                    AffineWitnessExpr::variable(combined_idx),
                    AffineWitnessExpr::variable(final_eval_idx),
                    AffineWitnessExpr::constant(ext_one()),
                ]);
            }
        }
        mul_term_idx += 1;
    }

    let expected_mul_terms = expected_mul_term_count(residual_plan)?;
    for _ in mul_term_idx..expected_mul_terms {
        add_commitment_message_affine_entry(
            forms.as_mut_slice(),
            &seed,
            column_idx,
            None,
            ext_zero(),
            ext_zero(),
        );
        column_idx += 1;
        add_commitment_message_affine_entry(
            forms.as_mut_slice(),
            &seed,
            column_idx,
            None,
            ext_zero(),
            ext_one(),
        );
        column_idx += 1;
        add_commitment_message_affine_entry(
            forms.as_mut_slice(),
            &seed,
            column_idx,
            None,
            ext_zero(),
            ext_zero(),
        );
        column_idx += 1;
        add_commitment_message_affine_entry(
            forms.as_mut_slice(),
            &seed,
            column_idx,
            None,
            ext_zero(),
            ext_one(),
        );
        column_idx += 1;

        multiplicative_field_exprs.push([
            AffineWitnessExpr::constant(ext_zero()),
            AffineWitnessExpr::constant(ext_one()),
            AffineWitnessExpr::constant(ext_zero()),
            AffineWitnessExpr::constant(ext_one()),
        ]);
    }

    Ok(CommitmentBindingLayout {
        commitment_slots,
        commitment_forms: forms,
        expected_linear_terms: linear_term_idx,
        expected_mul_terms,
        linear_coefficient_indices,
        linear_value_indices,
        multiplicative_field_exprs,
    })
}

fn slot_value_from_proof_object(
    proof_object: &Sp1GermBundle,
    slot: CommitmentWitnessSlot,
) -> Result<SP1ExtensionField, GermError> {
    match slot {
        CommitmentWitnessSlot::LinearCoefficient { term_idx } => {
            proof_object.lin_terms.get(term_idx).map(|term| term.coefficient).ok_or(
                GermError::TranscriptShapeMismatch {
                    which: "linear commitment slot",
                    got: proof_object.lin_terms.len(),
                    expected: term_idx + 1,
                },
            )
        }
        CommitmentWitnessSlot::LinearValue { term_idx } => {
            proof_object.lin_terms.get(term_idx).map(|term| term.value).ok_or(
                GermError::TranscriptShapeMismatch {
                    which: "linear commitment slot",
                    got: proof_object.lin_terms.len(),
                    expected: term_idx + 1,
                },
            )
        }
        CommitmentWitnessSlot::MultiplicativeField { term_idx, field } => proof_object
            .mul_terms
            .get(term_idx)
            .map(|term| match field {
                MultiplicativeTermField::A => term.a,
                MultiplicativeTermField::B => term.b,
                MultiplicativeTermField::C => term.c,
                MultiplicativeTermField::D => term.d,
            })
            .ok_or(GermError::TranscriptShapeMismatch {
                which: "multiplicative commitment slot",
                got: proof_object.mul_terms.len(),
                expected: term_idx + 1,
            }),
    }
}

#[must_use]
pub fn derive_challenges(
    public_values: &GermPublicValues,
    commitment: &Sp1PackageCommitment,
) -> GermChallenges {
    derive_challenges_from_arming_digest(&public_values.digest(), commitment)
}

#[must_use]
pub fn derive_challenges_from_capsule(
    capsule: &GermArmCapsule,
    commitment: &Sp1PackageCommitment,
) -> GermChallenges {
    derive_challenges_from_arming_digest(&capsule.digest(), commitment)
}

#[must_use]
pub fn derive_challenges_from_arming_digest(
    arming_digest: &[u8; 32],
    commitment: &Sp1PackageCommitment,
) -> GermChallenges {
    let challenge_seed = derive_commitment_bound_seed(arming_digest, commitment);
    GermChallenges {
        r_lin: derive_algebraic_challenge(b"sp1-germ/r_lin/v2", challenge_seed, &[], 0),
        r_mul: derive_algebraic_challenge(b"sp1-germ/r_mul/v2", challenge_seed, &[], 1),
    }
}

fn derive_commitment_bound_seed(
    arming_digest: &[u8; 32],
    commitment: &Sp1PackageCommitment,
) -> SP1ExtensionField {
    let after_digest = algebraic_absorb(
        bytes_to_extension(b"sp1-germ/challenge-seed/v2"),
        bytes_to_extension(arming_digest),
        1,
    );
    algebraic_absorb(after_digest, package_commitment_mix(commitment), 2)
}

fn commitment_mix_weight(coord_idx: usize) -> SP1ExtensionField {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/commitment-mix-weight/v1");
    h.update((coord_idx as u32).to_le_bytes());
    let digest: [u8; 32] = h.finalize().into();
    bytes_to_extension(&digest)
}

pub fn bind_bundle_to_capsule(
    bundle: &mut Sp1GermBundle,
    capsule: &GermArmCapsule,
) -> Result<([u8; 32], GermChallenges), GermError> {
    bundle.shared_object_commitment =
        compute_transcript_commitment(&bundle.lin_terms, &bundle.mul_terms);
    let commitment_root = compute_commitment_root(&bundle.shared_object_commitment);
    let arming_digest = capsule.digest();
    let challenges = derive_challenges_from_capsule(capsule, &bundle.shared_object_commitment);
    let lin_term_count = u32::try_from(bundle.lin_terms.len())
        .map_err(|_| GermError::TooManyLinearTerms(bundle.lin_terms.len()))?;
    let lin_folded_residual = fold_linear_terms(
        &bundle.lin_terms,
        &arming_digest,
        &bundle.shared_object_commitment,
        &challenges.r_lin,
    );
    let lin_proof = Sp1LinProof {
        term_count: lin_term_count,
        folded_residual: lin_folded_residual,
        // Open the exact same common commitment `C` that seeds challenges.
        ajtai_commitment: bundle.shared_object_commitment,
    };
    bundle.pi_lin = encode_lin_proof(&lin_proof);
    let sumcheck_seed = derive_mul_sumcheck_seed(
        &arming_digest,
        &bundle.shared_object_commitment,
        &challenges.r_mul,
    );
    let mul_sumcheck = prove_mul_sumcheck(&sumcheck_seed, &bundle.mul_terms);
    bundle.pi_mul = encode_mul_sumcheck(&mul_sumcheck);

    let lin_check = evaluate_linear_relation(
        bundle,
        &arming_digest,
        &commitment_root,
        &challenges.r_lin,
        b"sp1-germ/lin_bind/v1",
    )?;
    if !is_zero_ext(&lin_check.folded_residual) {
        return Err(GermError::NonZeroResidual("lin"));
    }
    let mul_check = evaluate_multiplicative_relation(
        bundle,
        &arming_digest,
        &commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
    )?;
    if !is_zero_ext(&mul_check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }

    bundle.lin_binding_tag = lin_check.fingerprint;
    bundle.mul_binding_tag = mul_check.fingerprint;
    Ok((commitment_root, challenges))
}

pub fn bind_bundle_to_capsule_with_orbweaver_terminal_openings(
    bundle: &mut Sp1GermBundle,
    capsule: &GermArmCapsule,
    srs: &OrbweaverOpeningSrs,
) -> Result<([u8; 32], GermChallenges), GermError> {
    validate_srs(srs, orbweaver_witness_width_from_mul_terms(bundle.mul_terms.len())?)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let (commitment_root, challenges) = bind_bundle_to_capsule(bundle, capsule)?;
    bundle.pi_mul_terminal_openings =
        build_orbweaver_terminal_openings_from_capsule(bundle, capsule, srs)?;
    let arming_digest = capsule.digest();
    let mul_check = evaluate_multiplicative_relation_internal(
        bundle,
        &arming_digest,
        &commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
        Some(srs),
    )?;
    if !is_zero_ext(&mul_check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    bundle.mul_binding_tag = mul_check.fingerprint;
    Ok((commitment_root, challenges))
}

pub fn bind_bundle(
    bundle: &mut Sp1GermBundle,
    public_values: &GermPublicValues,
) -> Result<([u8; 32], GermChallenges), GermError> {
    bundle.shared_object_commitment =
        compute_transcript_commitment(&bundle.lin_terms, &bundle.mul_terms);
    let commitment_root = compute_commitment_root(&bundle.shared_object_commitment);
    let public_values_digest = public_values.digest();
    let challenges = derive_challenges(public_values, &bundle.shared_object_commitment);
    let lin_term_count = u32::try_from(bundle.lin_terms.len())
        .map_err(|_| GermError::TooManyLinearTerms(bundle.lin_terms.len()))?;
    let lin_folded_residual = fold_linear_terms(
        &bundle.lin_terms,
        &public_values_digest,
        &bundle.shared_object_commitment,
        &challenges.r_lin,
    );
    let lin_proof = Sp1LinProof {
        term_count: lin_term_count,
        folded_residual: lin_folded_residual,
        // Open the exact same common commitment `C` that seeds challenges.
        ajtai_commitment: bundle.shared_object_commitment,
    };
    bundle.pi_lin = encode_lin_proof(&lin_proof);
    let sumcheck_seed = derive_mul_sumcheck_seed(
        &public_values_digest,
        &bundle.shared_object_commitment,
        &challenges.r_mul,
    );
    let mul_sumcheck = prove_mul_sumcheck(&sumcheck_seed, &bundle.mul_terms);
    bundle.pi_mul = encode_mul_sumcheck(&mul_sumcheck);

    let lin_check = evaluate_linear_relation(
        bundle,
        &public_values_digest,
        &commitment_root,
        &challenges.r_lin,
        b"sp1-germ/lin_bind/v1",
    )?;
    if !is_zero_ext(&lin_check.folded_residual) {
        return Err(GermError::NonZeroResidual("lin"));
    }
    let mul_check = evaluate_multiplicative_relation(
        bundle,
        &public_values_digest,
        &commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
    )?;
    if !is_zero_ext(&mul_check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }

    bundle.lin_binding_tag = lin_check.fingerprint;
    bundle.mul_binding_tag = mul_check.fingerprint;
    Ok((commitment_root, challenges))
}

pub fn bind_bundle_with_orbweaver_terminal_openings(
    bundle: &mut Sp1GermBundle,
    public_values: &GermPublicValues,
    srs: &OrbweaverOpeningSrs,
) -> Result<([u8; 32], GermChallenges), GermError> {
    validate_srs(srs, orbweaver_witness_width_from_mul_terms(bundle.mul_terms.len())?)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let (commitment_root, challenges) = bind_bundle(bundle, public_values)?;
    bundle.pi_mul_terminal_openings =
        build_orbweaver_terminal_openings(bundle, public_values, srs)?;
    let public_values_digest = public_values.digest();
    let mul_check = evaluate_multiplicative_relation_internal(
        bundle,
        &public_values_digest,
        &commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
        Some(srs),
    )?;
    if !is_zero_ext(&mul_check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    bundle.mul_binding_tag = mul_check.fingerprint;
    Ok((commitment_root, challenges))
}

/// Verify the global linear vanishing check `V_lin(r_lin) = 0`.
pub fn verify_lin(
    bundle: &Sp1GermBundle,
    public_values: &GermPublicValues,
    commitment_root: &[u8; 32],
) -> Result<GermRelationCheck, GermError> {
    let public_values_digest = public_values.digest();
    let challenges = derive_challenges(public_values, &bundle.shared_object_commitment);
    let check = evaluate_linear_relation(
        bundle,
        &public_values_digest,
        commitment_root,
        &challenges.r_lin,
        b"sp1-germ/lin_bind/v1",
    )?;
    if !is_zero_ext(&check.folded_residual) {
        return Err(GermError::NonZeroResidual("lin"));
    }
    if check.fingerprint != bundle.lin_binding_tag {
        return Err(GermError::BindingTagMismatch("lin"));
    }
    Ok(check)
}

/// Verify the global multiplicative vanishing check `V_mul(r_mul) = 0`.
///
/// The multiplicative path is additionally bound to `pi_mul`, which encodes
/// the packed sumcheck transcript for the same `r_mul`.
pub fn verify_mul(
    bundle: &Sp1GermBundle,
    public_values: &GermPublicValues,
    commitment_root: &[u8; 32],
) -> Result<GermRelationCheck, GermError> {
    let public_values_digest = public_values.digest();
    let challenges = derive_challenges(public_values, &bundle.shared_object_commitment);
    let check = evaluate_multiplicative_relation_internal(
        bundle,
        &public_values_digest,
        commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
        None,
    )?;
    if !is_zero_ext(&check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    if check.fingerprint != bundle.mul_binding_tag {
        return Err(GermError::BindingTagMismatch("mul"));
    }
    Ok(check)
}

pub fn verify_mul_with_orbweaver_terminal_openings(
    bundle: &Sp1GermBundle,
    public_values: &GermPublicValues,
    commitment_root: &[u8; 32],
    srs: &OrbweaverOpeningSrs,
) -> Result<GermRelationCheck, GermError> {
    validate_srs(srs, orbweaver_witness_width_from_mul_terms(bundle.mul_terms.len())?)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let public_values_digest = public_values.digest();
    let challenges = derive_challenges(public_values, &bundle.shared_object_commitment);
    let check = evaluate_multiplicative_relation_internal(
        bundle,
        &public_values_digest,
        commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
        Some(srs),
    )?;
    if !is_zero_ext(&check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    if check.fingerprint != bundle.mul_binding_tag {
        return Err(GermError::BindingTagMismatch("mul"));
    }
    Ok(check)
}

pub fn build_orbweaver_terminal_openings_from_capsule(
    bundle: &Sp1GermBundle,
    capsule: &GermArmCapsule,
    srs: &OrbweaverOpeningSrs,
) -> Result<crate::bundle::Sp1MulTerminalOpeningProofs, GermError> {
    validate_srs(srs, orbweaver_witness_width_from_mul_terms(bundle.mul_terms.len())?)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let arming_digest = capsule.digest();
    let challenges = derive_challenges_from_capsule(capsule, &bundle.shared_object_commitment);
    build_orbweaver_terminal_openings_internal(bundle, &arming_digest, &challenges.r_mul, srs)
}

pub fn build_orbweaver_terminal_openings(
    bundle: &Sp1GermBundle,
    public_values: &GermPublicValues,
    srs: &OrbweaverOpeningSrs,
) -> Result<crate::bundle::Sp1MulTerminalOpeningProofs, GermError> {
    validate_srs(srs, orbweaver_witness_width_from_mul_terms(bundle.mul_terms.len())?)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let public_values_digest = public_values.digest();
    let challenges = derive_challenges(public_values, &bundle.shared_object_commitment);
    build_orbweaver_terminal_openings_internal(
        bundle,
        &public_values_digest,
        &challenges.r_mul,
        srs,
    )
}

pub fn verify_orbweaver_terminal_openings_from_capsule(
    bundle: &Sp1GermBundle,
    capsule: &GermArmCapsule,
    srs: &OrbweaverOpeningSrs,
) -> Result<(), GermError> {
    validate_srs(srs, orbweaver_witness_width_from_mul_terms(bundle.mul_terms.len())?)
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let arming_digest = capsule.digest();
    let commitment_root = compute_commitment_root(&bundle.shared_object_commitment);
    let challenges = derive_challenges_from_capsule(capsule, &bundle.shared_object_commitment);
    let _ = evaluate_multiplicative_relation(
        bundle,
        &arming_digest,
        &commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
    )?;

    let mul_proof = decode_mul_sumcheck(&bundle.pi_mul)?;
    let round_challenges = collect_sumcheck_round_challenges(
        &derive_mul_sumcheck_seed(
            &arming_digest,
            &bundle.shared_object_commitment,
            &challenges.r_mul,
        ),
        &mul_proof,
    )?;
    let weights = eq_table(round_challenges.as_slice());
    let aggregation_coeffs = derive_orbweaver_aggregation_coeffs(
        &arming_digest,
        &bundle.shared_object_commitment,
        bundle.pi_mul.as_slice(),
    );
    let expected_scalar_values = expected_orbweaver_scalar_image_values(&mul_proof);
    verify_aggregated_scalar_image_openings_from_mul_terms(
        srs,
        bundle.mul_terms.as_slice(),
        weights.as_slice(),
        &aggregation_coeffs,
        &expected_scalar_values,
        &bundle.pi_mul_terminal_openings,
    )
    .map_err(|msg| GermError::MulTerminalOpeningFailed { which: "terminal", msg })?;
    Ok(())
}

pub fn verify_transcript_bound_sp1_germ_proof_object(
    transcript_bound: &TranscriptBoundSp1GermProofObject,
    capsule: &GermArmCapsule,
) -> Result<(), GermError> {
    let arming_digest = capsule.digest();
    let computed_root =
        compute_commitment_root(&transcript_bound.proof_object.shared_object_commitment);
    if computed_root != transcript_bound.commitment_root {
        return Err(GermError::CommitmentRootMismatch);
    }
    let challenges = derive_challenges_from_capsule(
        capsule,
        &transcript_bound.proof_object.shared_object_commitment,
    );
    let lin_check = evaluate_linear_relation(
        &transcript_bound.proof_object,
        &arming_digest,
        &transcript_bound.commitment_root,
        &challenges.r_lin,
        b"sp1-germ/lin_bind/v1",
    )?;
    if !is_zero_ext(&lin_check.folded_residual) {
        return Err(GermError::NonZeroResidual("lin"));
    }
    if lin_check.fingerprint != transcript_bound.proof_object.lin_binding_tag {
        return Err(GermError::BindingTagMismatch("lin"));
    }
    let mul_check = evaluate_multiplicative_relation(
        &transcript_bound.proof_object,
        &arming_digest,
        &transcript_bound.commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
    )?;
    if !is_zero_ext(&mul_check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    if mul_check.fingerprint != transcript_bound.proof_object.mul_binding_tag {
        return Err(GermError::BindingTagMismatch("mul"));
    }
    Ok(())
}

pub fn verify_transcript_bound_sp1_germ_proof_object_with_orbweaver_terminal_openings(
    transcript_bound: &TranscriptBoundSp1GermProofObject,
    capsule: &GermArmCapsule,
    srs: &OrbweaverOpeningSrs,
) -> Result<(), GermError> {
    validate_srs(
        srs,
        orbweaver_witness_width_from_mul_terms(transcript_bound.proof_object.mul_terms.len())?,
    )
        .map_err(GermError::InvalidOrbweaverSrs)?;
    let arming_digest = capsule.digest();
    let computed_root =
        compute_commitment_root(&transcript_bound.proof_object.shared_object_commitment);
    if computed_root != transcript_bound.commitment_root {
        return Err(GermError::CommitmentRootMismatch);
    }
    let challenges = derive_challenges_from_capsule(
        capsule,
        &transcript_bound.proof_object.shared_object_commitment,
    );
    let lin_check = evaluate_linear_relation(
        &transcript_bound.proof_object,
        &arming_digest,
        &transcript_bound.commitment_root,
        &challenges.r_lin,
        b"sp1-germ/lin_bind/v1",
    )?;
    if !is_zero_ext(&lin_check.folded_residual) {
        return Err(GermError::NonZeroResidual("lin"));
    }
    if lin_check.fingerprint != transcript_bound.proof_object.lin_binding_tag {
        return Err(GermError::BindingTagMismatch("lin"));
    }
    let mul_check = evaluate_multiplicative_relation_internal(
        &transcript_bound.proof_object,
        &arming_digest,
        &transcript_bound.commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
        Some(srs),
    )?;
    if !is_zero_ext(&mul_check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    if mul_check.fingerprint != transcript_bound.proof_object.mul_binding_tag {
        return Err(GermError::BindingTagMismatch("mul"));
    }
    Ok(())
}

pub fn compile_germ_aadp_template(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
) -> Result<GermAadpVerifierTemplate, GermError> {
    compile_germ_aadp_template_internal(capsule, residual_plan, None, None)
}

pub fn compile_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
    transcript_bound: &TranscriptBoundSp1GermProofObject,
    srs: &OrbweaverOpeningSrs,
) -> Result<GermAadpVerifierTemplate, GermError> {
    validate_residual_plan_for_template(capsule, residual_plan)?;
    validate_srs(
        srs,
        orbweaver_witness_width_from_mul_terms(transcript_bound.proof_object.mul_terms.len())?,
    )
        .map_err(GermError::InvalidOrbweaverSrs)?;
    verify_transcript_bound_sp1_germ_proof_object_with_orbweaver_terminal_openings(
        transcript_bound,
        capsule,
        srs,
    )?;
    let expected_linear_terms =
        residual_plan.linear_descriptors.iter().map(linear_descriptor_term_count).sum::<usize>();
    if transcript_bound.proof_object.lin_terms.len() != expected_linear_terms {
        return Err(GermError::TranscriptShapeMismatch {
            which: "linear terms",
            got: transcript_bound.proof_object.lin_terms.len(),
            expected: expected_linear_terms,
        });
    }
    let expected_mul_terms = expected_mul_term_count(residual_plan)?;
    if transcript_bound.proof_object.mul_terms.len() != expected_mul_terms {
        return Err(GermError::TranscriptShapeMismatch {
            which: "multiplicative terms",
            got: transcript_bound.proof_object.mul_terms.len(),
            expected: expected_mul_terms,
        });
    }
    let orbweaver_data =
        build_orbweaver_transcript_template_data(capsule, &transcript_bound.proof_object, srs)?;
    compile_germ_aadp_template_internal(
        capsule,
        residual_plan,
        None,
        Some(&orbweaver_data),
    )
}

pub fn arm_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings<R: RngCore>(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
    transcript_bound: &TranscriptBoundSp1GermProofObject,
    srs: &OrbweaverOpeningSrs,
    message: SP1ExtensionField,
    rng: &mut R,
) -> Result<ArmedGermAadpCiphertext, GermError> {
    let template = compile_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings(
        capsule,
        residual_plan,
        transcript_bound,
        srs,
    )?;
    let ciphertext =
        aadp_encrypt_scalar(&template.cs, message, rng).map_err(GermError::AadpEncryptFailed)?;
    Ok(ArmedGermAadpCiphertext { template, ciphertext, capsule_digest: capsule.digest() })
}

pub fn compile_transcript_bound_germ_aadp_template(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
    transcript_bound: &TranscriptBoundSp1GermProofObject,
) -> Result<GermAadpVerifierTemplate, GermError> {
    validate_residual_plan_for_template(capsule, residual_plan)?;
    verify_transcript_bound_sp1_germ_proof_object(transcript_bound, capsule)?;

    let expected_linear_terms =
        residual_plan.linear_descriptors.iter().map(linear_descriptor_term_count).sum::<usize>();
    if transcript_bound.proof_object.lin_terms.len() != expected_linear_terms {
        return Err(GermError::TranscriptShapeMismatch {
            which: "linear terms",
            got: transcript_bound.proof_object.lin_terms.len(),
            expected: expected_linear_terms,
        });
    }
    let expected_mul_terms = expected_mul_term_count(residual_plan)?;
    if transcript_bound.proof_object.mul_terms.len() != expected_mul_terms {
        return Err(GermError::TranscriptShapeMismatch {
            which: "multiplicative terms",
            got: transcript_bound.proof_object.mul_terms.len(),
            expected: expected_mul_terms,
        });
    }

    let proof_object = &transcript_bound.proof_object;
    let mul_proof = decode_mul_sumcheck(&proof_object.pi_mul)?;
    if usize::from(capsule.sumcheck_rounds) != mul_proof.rounds.len() {
        return Err(GermError::SumcheckRoundsMismatch {
            got: mul_proof.rounds.len(),
            expected: usize::from(capsule.sumcheck_rounds),
        });
    }
    let arming_digest = capsule.digest();
    let challenges =
        derive_challenges_from_capsule(capsule, &proof_object.shared_object_commitment);
    let sumcheck_seed = derive_mul_sumcheck_seed(
        &arming_digest,
        &proof_object.shared_object_commitment,
        &challenges.r_mul,
    );
    let opening_weights =
        eq_table(collect_sumcheck_round_challenges(&sumcheck_seed, &mul_proof)?.as_slice());

    compile_germ_aadp_template_internal(
        capsule,
        residual_plan,
        Some(opening_weights.as_slice()),
        None,
    )
}

fn compile_germ_aadp_template_internal(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
    terminal_opening_weights: Option<&[SP1ExtensionField]>,
    orbweaver_template_data: Option<&OrbweaverTranscriptTemplateData>,
) -> Result<GermAadpVerifierTemplate, GermError> {
    validate_residual_plan_for_template(capsule, residual_plan)?;

    let mut num_variables = 0usize;
    let mut constraints = Vec::<AadpMulConstraint<SP1ExtensionField>>::new();
    let mut linear_round_checks = 0usize;
    let mut opening_checks = 0usize;
    let mut multiplication_gates = 0usize;

    let mut alloc = || {
        let idx = num_variables;
        num_variables += 1;
        idx
    };

    // Projective-safety anchor: force the homogenizing bit to 1 and boolean.
    let safety_bit_idx = alloc();
    add_bit_constraint(&mut constraints, safety_bit_idx);
    add_linear_zero_constraint(
        &mut constraints,
        AadpLinearForm { constant: -ext_one(), terms: vec![(safety_bit_idx, ext_one())] },
    );
    linear_round_checks += 1;
    multiplication_gates += 1;

    // In-boundary challenge derivation anchor:
    // witness carries commitment coordinates, and challenges derive from their fixed linear mix.
    let commitment_coord_indices: [usize; PACKAGE_AJTAI_ROWS * PACKAGE_AJTAI_RING_DIM] =
        core::array::from_fn(|_| alloc());
    let commitment_mix_idx = alloc();
    let mut commitment_mix_terms = Vec::with_capacity(1 + commitment_coord_indices.len());
    commitment_mix_terms.push((commitment_mix_idx, ext_one()));
    for (coord_idx, coord_var_idx) in commitment_coord_indices.iter().enumerate() {
        commitment_mix_terms.push((*coord_var_idx, -commitment_mix_weight(coord_idx)));
    }
    add_linear_zero_constraint(
        &mut constraints,
        AadpLinearForm { constant: ext_zero(), terms: commitment_mix_terms },
    );
    linear_round_checks += 1;

    let capsule_digest = capsule.digest();
    let challenge_seed_after_digest_const = algebraic_absorb(
        bytes_to_extension(b"sp1-germ/challenge-seed/v2"),
        bytes_to_extension(&capsule_digest),
        1,
    );
    let (_challenge_seed_sq_idx, challenge_seed_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        None,
        challenge_seed_after_digest_const,
        Some(commitment_mix_idx),
        ext_zero(),
        2,
    );
    multiplication_gates += 2;

    // r_lin = Challenge(theta, idx=0)
    let (_r_lin_seed_sq_idx, r_lin_seed_state_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        None,
        bytes_to_extension(b"sp1-germ/r_lin/v2"),
        Some(challenge_seed_idx),
        ext_zero(),
        11,
    );
    let (_r_lin_idx_sq_idx, r_lin_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        Some(r_lin_seed_state_idx),
        ext_zero(),
        None,
        ext_zero(),
        13,
    );
    multiplication_gates += 4;

    // r_mul = Challenge(theta, idx=1)
    let (_r_mul_seed_sq_idx, r_mul_seed_state_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        None,
        bytes_to_extension(b"sp1-germ/r_mul/v2"),
        Some(challenge_seed_idx),
        ext_zero(),
        12,
    );
    let (_r_mul_idx_sq_idx, r_mul_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        Some(r_mul_seed_state_idx),
        ext_zero(),
        None,
        ext_from_u32(1),
        14,
    );
    multiplication_gates += 4;

    // sumcheck_seed = left + (1+7) * right, where:
    // left  = Challenge(".../mul-sumcheck-seed/v2", theta, [r_mul], 0)
    // right = Challenge(".../mul-sumcheck-seed/v2", theta, [r_mul], 1)
    let (_left_seed_sq_idx, left_seed_state_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        None,
        bytes_to_extension(b"sp1-germ/mul-sumcheck-seed/v2"),
        Some(challenge_seed_idx),
        ext_zero(),
        11,
    );
    let (_left_idx_sq_idx, left_idx_state_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        Some(left_seed_state_idx),
        ext_zero(),
        None,
        ext_zero(),
        13,
    );
    let (_left_extra_sq_idx, left_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        Some(left_idx_state_idx),
        ext_zero(),
        Some(r_mul_idx),
        ext_zero(),
        17,
    );
    let (_right_seed_sq_idx, right_seed_state_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        None,
        bytes_to_extension(b"sp1-germ/mul-sumcheck-seed/v2"),
        Some(challenge_seed_idx),
        ext_zero(),
        12,
    );
    let (_right_idx_sq_idx, right_idx_state_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        Some(right_seed_state_idx),
        ext_zero(),
        None,
        ext_from_u32(1),
        14,
    );
    let (_right_extra_sq_idx, right_idx) = add_absorb_constraints(
        &mut constraints,
        &mut alloc,
        Some(right_idx_state_idx),
        ext_zero(),
        Some(r_mul_idx),
        ext_zero(),
        18,
    );
    multiplication_gates += 12;
    let sumcheck_seed_idx = alloc();
    add_linear_zero_constraint(
        &mut constraints,
        AadpLinearForm {
            constant: ext_zero(),
            terms: vec![
                (sumcheck_seed_idx, ext_one()),
                (left_idx, -ext_one()),
                (right_idx, -(ext_one() + ext_from_u32(7))),
            ],
        },
    );
    linear_round_checks += 1;

    let inv2 = ext_from_u32(2).try_inverse().expect("2 must be invertible in SP1 extension field");
    let inv6 = ext_from_u32(6).try_inverse().expect("6 must be invertible in SP1 extension field");

    // Exact in-boundary binding: `C = Com(T_pre)` over the compressed transcript basis.
    let binding_layout =
        build_commitment_binding_layout(&mut alloc, &commitment_coord_indices, residual_plan)?;
    for form in binding_layout.commitment_forms.iter().cloned() {
        add_linear_zero_constraint(&mut constraints, form);
        opening_checks += 1;
    }

    if binding_layout.expected_linear_terms > 0 {
        // lin_point_seed = left + (1+7) * right, where:
        // left  = Challenge(".../lin-point-seed/v2", theta, [r_lin], 2)
        // right = Challenge(".../lin-point-seed/v2", theta, [r_lin], 3)
        let (_lin_left_seed_sq_idx, lin_left_seed_state_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            None,
            bytes_to_extension(b"sp1-germ/lin-point-seed/v2"),
            Some(challenge_seed_idx),
            ext_zero(),
            13,
        );
        let (_lin_left_idx_sq_idx, lin_left_idx_state_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            Some(lin_left_seed_state_idx),
            ext_zero(),
            None,
            ext_from_u32(2),
            15,
        );
        let (_lin_left_extra_sq_idx, lin_left_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            Some(lin_left_idx_state_idx),
            ext_zero(),
            Some(r_lin_idx),
            ext_zero(),
            19,
        );
        let (_lin_right_seed_sq_idx, lin_right_seed_state_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            None,
            bytes_to_extension(b"sp1-germ/lin-point-seed/v2"),
            Some(challenge_seed_idx),
            ext_zero(),
            14,
        );
        let (_lin_right_idx_sq_idx, lin_right_idx_state_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            Some(lin_right_seed_state_idx),
            ext_zero(),
            None,
            ext_from_u32(3),
            16,
        );
        let (_lin_right_extra_sq_idx, lin_right_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            Some(lin_right_idx_state_idx),
            ext_zero(),
            Some(r_lin_idx),
            ext_zero(),
            20,
        );
        multiplication_gates += 12;
        let lin_point_seed_idx = alloc();
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (lin_point_seed_idx, ext_one()),
                    (lin_left_idx, -ext_one()),
                    (lin_right_idx, -(ext_one() + ext_from_u32(7))),
                ],
            },
        );
        linear_round_checks += 1;

        let lin_nvars = mul_sumcheck_nvars(binding_layout.expected_linear_terms);
        let lin_point_domain = bytes_to_extension(b"sp1-germ/lin-point/v1");
        let mut lin_point_indices = Vec::with_capacity(lin_nvars);
        for var_idx in 0..lin_nvars {
            let (_lin_point_seed_sq_idx, lin_point_seed_state_idx) = add_absorb_constraints(
                &mut constraints,
                &mut alloc,
                None,
                lin_point_domain,
                Some(lin_point_seed_idx),
                ext_zero(),
                (var_idx as u32).wrapping_add(11),
            );
            let (_lin_point_idx_sq_idx, lin_point_idx) = add_absorb_constraints(
                &mut constraints,
                &mut alloc,
                Some(lin_point_seed_state_idx),
                ext_zero(),
                None,
                ext_from_u32(var_idx as u32),
                (var_idx as u32).wrapping_add(13),
            );
            multiplication_gates += 4;
            lin_point_indices.push(lin_point_idx);
        }

        let mut lin_table_forms = Vec::<AadpLinearForm<SP1ExtensionField>>::with_capacity(
            binding_layout.expected_linear_terms.max(1).next_power_of_two(),
        );
        for term_idx in 0..binding_layout.expected_linear_terms {
            let value_idx = binding_layout.linear_value_indices[term_idx];
            if let Some(coeff_idx) = binding_layout.linear_coefficient_indices[term_idx] {
                let product_idx = alloc();
                add_mul_equals_var(
                    &mut constraints,
                    linear_form_single_var(coeff_idx),
                    linear_form_single_var(value_idx),
                    product_idx,
                );
                multiplication_gates += 1;
                lin_table_forms.push(linear_form_single_var(product_idx));
            } else {
                lin_table_forms.push(linear_form_single_var(value_idx));
            }
        }
        let padded_linear_terms = binding_layout.expected_linear_terms.max(1).next_power_of_two();
        while lin_table_forms.len() < padded_linear_terms {
            lin_table_forms.push(linear_form_constant(ext_zero()));
        }
        for r_idx in lin_point_indices {
            let mut next_forms =
                Vec::<AadpLinearForm<SP1ExtensionField>>::with_capacity(lin_table_forms.len() / 2);
            for pair in lin_table_forms.chunks_exact(2) {
                let delta_idx = alloc();
                add_mul_equals_var(
                    &mut constraints,
                    linear_form_sub_forms(&pair[1], &pair[0]),
                    linear_form_single_var(r_idx),
                    delta_idx,
                );
                multiplication_gates += 1;
                next_forms
                    .push(linear_form_add_forms(&pair[0], &linear_form_single_var(delta_idx)));
            }
            lin_table_forms = next_forms;
        }
        let lin_claim_idx = alloc();
        add_linear_zero_constraint(
            &mut constraints,
            linear_form_sub_forms(&linear_form_single_var(lin_claim_idx), &lin_table_forms[0]),
        );
        linear_round_checks += 1;
        add_linear_zero_constraint(&mut constraints, linear_form_single_var(lin_claim_idx));
        linear_round_checks += 1;
    }

    // Packed sumcheck verifier with round-chained extractor updates.
    let mut claimed_prev_idx: Option<usize> = None;
    let mut round_challenge_indices = Vec::with_capacity(usize::from(capsule.sumcheck_rounds));
    let mut round_state_idx = sumcheck_seed_idx;
    for _round_idx in 0..usize::from(capsule.sumcheck_rounds) {
        let eval_indices = [alloc(), alloc(), alloc(), alloc()];

        let mut identity_terms = vec![(eval_indices[0], ext_one()), (eval_indices[1], ext_one())];
        if let Some(prev_idx) = claimed_prev_idx {
            identity_terms.push((prev_idx, -ext_one()));
        }
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm { constant: ext_zero(), terms: identity_terms },
        );
        linear_round_checks += 1;

        let r_sc_idx = add_challenge_from_seed_constraints(
            &mut constraints,
            &mut alloc,
            MUL_SUMCHECK_ROUND_CHALLENGE_DOMAIN,
            round_state_idx,
            0,
        );
        multiplication_gates += 4;
        round_challenge_indices.push(r_sc_idx);

        // Newton/Horner interpolation at points 0,1,2,3:
        // p(r)=e0 + r*(d1 + (r-1)*(c2 + (r-2)*c3))
        // d1=e1-e0, c2=(e2-2e1+e0)/2, c3=(e3-3e2+3e1-e0)/6
        let m_a_idx = alloc(); // (r-2)*c3
        let m_b_idx = alloc(); // (c2 + m_a)*(r-1)
        let m_c_idx = alloc(); // (d1 + m_b)*r
        let claimed_next_idx = alloc();

        add_mul_equals_var(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (eval_indices[3], inv6),
                    (eval_indices[2], -(inv6 + inv6 + inv6)),
                    (eval_indices[1], inv6 + inv6 + inv6),
                    (eval_indices[0], -inv6),
                ],
            },
            AadpLinearForm { constant: -ext_from_u32(2), terms: vec![(r_sc_idx, ext_one())] },
            m_a_idx,
        );
        add_mul_equals_var(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (eval_indices[2], inv2),
                    (eval_indices[1], -(inv2 + inv2)),
                    (eval_indices[0], inv2),
                    (m_a_idx, ext_one()),
                ],
            },
            AadpLinearForm { constant: -ext_from_u32(1), terms: vec![(r_sc_idx, ext_one())] },
            m_b_idx,
        );
        add_mul_equals_var(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (eval_indices[1], ext_one()),
                    (eval_indices[0], -ext_one()),
                    (m_b_idx, ext_one()),
                ],
            },
            linear_form_single_var(r_sc_idx),
            m_c_idx,
        );
        multiplication_gates += 3;

        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (claimed_next_idx, ext_one()),
                    (eval_indices[0], -ext_one()),
                    (m_c_idx, -ext_one()),
                ],
            },
        );
        linear_round_checks += 1;

        round_state_idx = add_sumcheck_round_state_constraints(
            &mut constraints,
            &mut alloc,
            round_state_idx,
            eval_indices,
            claimed_next_idx,
        );
        multiplication_gates += 12;
        claimed_prev_idx = Some(claimed_next_idx);
    }

    let opening_a_idx = alloc();
    let opening_b_idx = alloc();
    let opening_c_idx = alloc();
    let opening_d_idx = alloc();

    let opening_lhs_product_idx = alloc();
    let opening_rhs_product_idx = alloc();

    constraints.push(AadpMulConstraint {
        a: linear_form_single_var(opening_a_idx),
        b: linear_form_single_var(opening_b_idx),
        c: linear_form_constant(ext_one()),
        d: linear_form_single_var(opening_lhs_product_idx),
    });
    constraints.push(AadpMulConstraint {
        a: linear_form_single_var(opening_c_idx),
        b: linear_form_single_var(opening_d_idx),
        c: linear_form_constant(ext_one()),
        d: linear_form_single_var(opening_rhs_product_idx),
    });
    multiplication_gates += 2;

    let orbweaver_terminal_pi_len = if let Some(orbweaver_data) = orbweaver_template_data {
        let seed = derive_package_ajtai_seed();
        let mut proof_commitment_forms = (0..PACKAGE_AJTAI_ROWS)
            .flat_map(|row| {
                (0..PACKAGE_AJTAI_RING_DIM).map(move |coeff_idx| AadpLinearForm {
                    constant: orbweaver_data.proof_commitment[row][coeff_idx],
                    terms: Vec::new(),
                })
            })
            .collect::<Vec<_>>();
        let mut column_idx = 0usize;
        add_commitment_message_affine_entry(
            proof_commitment_forms.as_mut_slice(),
            &seed,
            column_idx,
            None,
            ext_zero(),
            domain_tag_extension(b"sp1-germ/orbweaver/proof-commit/v1"),
        );
        column_idx += 1;
        let proof_block_len = orbweaver_data.lhs_block_multipliers.len();
        let mut proof_block_indices =
            vec![Vec::<usize>::with_capacity(proof_block_len); ORBWEAVER_AGGREGATED_SCALAR_OPENINGS];
        for proof_idx in 0..ORBWEAVER_AGGREGATED_SCALAR_OPENINGS {
            for _ in 0..proof_block_len {
                let block_idx = alloc();
                proof_block_indices[proof_idx].push(block_idx);
                add_commitment_message_affine_entry(
                    proof_commitment_forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    Some(block_idx),
                    ext_one(),
                    ext_zero(),
                );
                column_idx += 1;
            }
        }
        let flat_proof_block_indices = proof_block_indices
            .iter()
            .flat_map(|indices| indices.iter().copied())
            .collect::<Vec<_>>();
        let mut projection_square_indices = Vec::with_capacity(ORBWEAVER_JL_TOTAL_PROJECTIONS);
        for form in proof_commitment_forms {
            add_linear_zero_constraint(&mut constraints, form);
            opening_checks += 1;
        }

        let mut c_flat_form = linear_form_constant(ext_zero());
        for (field_multipliers, exprs) in orbweaver_data
            .c_flat_field_multipliers
            .iter()
            .zip(binding_layout.multiplicative_field_exprs.iter())
        {
            for field_idx in 0..4 {
                add_scaled_affine_expr(&mut c_flat_form, exprs[field_idx], field_multipliers[field_idx]);
            }
        }
        let opening_indices = [opening_a_idx, opening_b_idx, opening_c_idx, opening_d_idx];
        for proof_idx in 0..ORBWEAVER_AGGREGATED_SCALAR_OPENINGS {
            let mut lhs_form = linear_form_constant(ext_zero());
            for (block_idx, coeff) in proof_block_indices[proof_idx]
                .iter()
                .zip(orbweaver_data.lhs_block_multipliers.iter())
            {
                lhs_form.terms.push((*block_idx, *coeff));
            }
            let mut opening_form = linear_form_add_forms(
                &lhs_form,
                &linear_form_scale(
                    &c_flat_form,
                    -ext_from_base_field(orbweaver_data.aggregated_vk_values[proof_idx]),
                ),
            );
            for field_idx in 0..4 {
                opening_form.terms.push((
                    opening_indices[field_idx],
                    orbweaver_data.aggregated_output_multipliers[proof_idx][field_idx],
                ));
            }
            let opening_limb_indices = [alloc(), alloc(), alloc(), alloc()];
            opening_form.terms.push((opening_limb_indices[0], -ext_one()));
            opening_form.terms.push((opening_limb_indices[1], -ext_basis_u()));
            opening_form.terms.push((opening_limb_indices[2], -ext_basis_u_squared()));
            opening_form.terms.push((opening_limb_indices[3], -ext_basis_u_cubed()));
            add_linear_zero_constraint(&mut constraints, opening_form);
            opening_checks += 1;
            add_linear_zero_constraint(
                &mut constraints,
                AadpLinearForm {
                    constant: ext_zero(),
                    terms: vec![(opening_limb_indices[0], ext_one())],
                },
            );
            opening_checks += 1;
        }
        for row in &orbweaver_data.jl_rows {
            let row_weights = row
                .iter()
                .map(|coeff| match *coeff {
                    1 => SP1Field::one(),
                    -1 => -SP1Field::one(),
                    _ => SP1Field::zero(),
                })
                .collect::<Vec<_>>();
            let row_multipliers =
                pack_scalar_weights_to_extension_multipliers(row_weights.as_slice());
            let mut projection_form = linear_form_constant(ext_zero());
            for (block_idx, coeff) in flat_proof_block_indices.iter().zip(row_multipliers.iter()) {
                projection_form.terms.push((*block_idx, *coeff));
            }
            let projection_limb_indices = [alloc(), alloc(), alloc(), alloc()];
            projection_form.terms.push((projection_limb_indices[0], -ext_one()));
            projection_form.terms.push((projection_limb_indices[1], -ext_basis_u()));
            projection_form.terms.push((projection_limb_indices[2], -ext_basis_u_squared()));
            projection_form.terms.push((projection_limb_indices[3], -ext_basis_u_cubed()));
            add_linear_zero_constraint(&mut constraints, projection_form);
            opening_checks += 1;
            let square_idx = alloc();
            add_mul_equals_var(
                &mut constraints,
                linear_form_single_var(projection_limb_indices[0]),
                linear_form_single_var(projection_limb_indices[0]),
                square_idx,
            );
            multiplication_gates += 1;
            projection_square_indices.push(square_idx);
        }
        let mut jl_norm_bits_terms =
            Vec::with_capacity(projection_square_indices.len() + ORBWEAVER_JL_NORM_BITS);
        for square_idx in projection_square_indices {
            jl_norm_bits_terms.push((square_idx, ext_one()));
        }
        for bit in 0..ORBWEAVER_JL_NORM_BITS {
            let bit_idx = alloc();
            add_bit_constraint(&mut constraints, bit_idx);
            multiplication_gates += 1;
            jl_norm_bits_terms.push((bit_idx, -ext_from_u32(1u32 << bit)));
        }
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm { constant: ext_zero(), terms: jl_norm_bits_terms },
        );
        opening_checks += 1;
        orbweaver_data.proof_pi_len
    } else if let Some(weights) = terminal_opening_weights {
        if weights.len() != binding_layout.multiplicative_field_exprs.len() {
            return Err(GermError::TranscriptShapeMismatch {
                which: "terminal opening weights",
                got: weights.len(),
                expected: binding_layout.multiplicative_field_exprs.len(),
            });
        }
        let opening_bindings = [
            (
                opening_a_idx,
                build_weighted_mul_opening_form(
                    weights,
                    binding_layout.multiplicative_field_exprs.as_slice(),
                    MultiplicativeTermField::A,
                ),
            ),
            (
                opening_b_idx,
                build_weighted_mul_opening_form(
                    weights,
                    binding_layout.multiplicative_field_exprs.as_slice(),
                    MultiplicativeTermField::B,
                ),
            ),
            (
                opening_c_idx,
                build_weighted_mul_opening_form(
                    weights,
                    binding_layout.multiplicative_field_exprs.as_slice(),
                    MultiplicativeTermField::C,
                ),
            ),
            (
                opening_d_idx,
                build_weighted_mul_opening_form(
                    weights,
                    binding_layout.multiplicative_field_exprs.as_slice(),
                    MultiplicativeTermField::D,
                ),
            ),
        ];
        for (opening_idx, opening_form) in opening_bindings {
            add_linear_zero_constraint(
                &mut constraints,
                linear_form_sub_forms(&linear_form_single_var(opening_idx), &opening_form),
            );
            opening_checks += 1;
        }
        0
    } else {
        0
    };

    if capsule.sumcheck_rounds == 0 {
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (opening_lhs_product_idx, ext_one()),
                    (opening_rhs_product_idx, -ext_one()),
                ],
            },
        );
        linear_round_checks += 1;
    } else {
        let mut point_indices = Vec::with_capacity(usize::from(capsule.sumcheck_rounds));
        let mul_point_domain = bytes_to_extension(b"sp1-germ/mul-point/v1");
        for var_idx in 0..usize::from(capsule.sumcheck_rounds) {
            let (_point_seed_sq_idx, point_seed_state_idx) = add_absorb_constraints(
                &mut constraints,
                &mut alloc,
                None,
                mul_point_domain,
                Some(sumcheck_seed_idx),
                ext_zero(),
                (var_idx as u32).wrapping_add(11),
            );
            let (_point_idx_sq_idx, point_idx) = add_absorb_constraints(
                &mut constraints,
                &mut alloc,
                Some(point_seed_state_idx),
                ext_zero(),
                None,
                ext_from_u32(var_idx as u32),
                (var_idx as u32).wrapping_add(13),
            );
            multiplication_gates += 4;
            point_indices.push(point_idx);
        }

        let mut eq_prev_idx: Option<usize> = None;
        for (round_idx, (r_idx, s_idx)) in
            point_indices.iter().zip(round_challenge_indices.iter()).enumerate()
        {
            let rs_idx = alloc(); // r*s
            add_mul_equals_var(
                &mut constraints,
                linear_form_single_var(*r_idx),
                linear_form_single_var(*s_idx),
                rs_idx,
            );
            let factor_idx = alloc();
            add_linear_zero_constraint(
                &mut constraints,
                AadpLinearForm {
                    constant: -ext_one(),
                    terms: vec![
                        (factor_idx, ext_one()),
                        (*r_idx, ext_one()),
                        (*s_idx, ext_one()),
                        (rs_idx, -(ext_from_u32(2))),
                    ],
                },
            );
            linear_round_checks += 1;

            let eq_next_idx = alloc();
            if round_idx == 0 {
                add_linear_zero_constraint(
                    &mut constraints,
                    AadpLinearForm {
                        constant: ext_zero(),
                        terms: vec![(eq_next_idx, ext_one()), (factor_idx, -ext_one())],
                    },
                );
                linear_round_checks += 1;
            } else if let Some(prev_idx) = eq_prev_idx {
                add_mul_equals_var(
                    &mut constraints,
                    linear_form_single_var(prev_idx),
                    linear_form_single_var(factor_idx),
                    eq_next_idx,
                );
                multiplication_gates += 1;
            }
            multiplication_gates += 1;
            eq_prev_idx = Some(eq_next_idx);
        }
        let eq_eval_idx =
            eq_prev_idx.expect("sumcheck rounds > 0 must allocate eq-eval accumulator");

        let final_scaled_idx = alloc();
        let delta_idx = alloc();
        let claimed_last_idx = claimed_prev_idx.expect("sumcheck rounds > 0 must allocate claims");
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![
                    (delta_idx, ext_one()),
                    (opening_lhs_product_idx, -ext_one()),
                    (opening_rhs_product_idx, ext_one()),
                ],
            },
        );
        constraints.push(AadpMulConstraint {
            a: linear_form_single_var(eq_eval_idx),
            b: linear_form_single_var(delta_idx),
            c: linear_form_constant(ext_one()),
            d: linear_form_single_var(final_scaled_idx),
        });
        multiplication_gates += 1;
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![(claimed_last_idx, ext_one()), (final_scaled_idx, -ext_one())],
            },
        );
        linear_round_checks += 1;
    }

    Ok(GermAadpVerifierTemplate {
        cs: AadpConstraintSystem { num_variables, constraints },
        layout: GermAadpWitnessLayout {
            sumcheck_rounds: usize::from(capsule.sumcheck_rounds),
            expected_linear_terms: binding_layout.expected_linear_terms,
            expected_mul_terms: binding_layout.expected_mul_terms,
            commitment_slots: binding_layout.commitment_slots,
            linear_term_has_explicit_coefficient: binding_layout
                .linear_coefficient_indices
                .iter()
                .map(|idx| idx.is_some())
                .collect(),
            orbweaver_terminal_pi_len,
            orbweaver_aggregated_proof_count: if orbweaver_template_data.is_some() {
                ORBWEAVER_AGGREGATED_SCALAR_OPENINGS
            } else {
                0
            },
        },
        stats: GermAadpConstraintStats {
            linear_round_checks,
            opening_checks,
            multiplication_gates,
        },
        capsule_digest: capsule.digest(),
    })
}

pub fn arm_germ_aadp_template<R: RngCore>(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
    message: SP1ExtensionField,
    rng: &mut R,
) -> Result<ArmedGermAadpCiphertext, GermError> {
    let template = compile_germ_aadp_template(capsule, residual_plan)?;
    let ciphertext =
        aadp_encrypt_scalar(&template.cs, message, rng).map_err(GermError::AadpEncryptFailed)?;
    Ok(ArmedGermAadpCiphertext { template, ciphertext, capsule_digest: capsule.digest() })
}

/// Materialize the AADP witness from a transcript-bound proof object.
///
/// This function is *outside* the WE security boundary: it assumes a host-side transcript layer
/// already bound the proof object to `commitment_root` and derived the designated challenges from
/// `(x_arm, C)`. The resulting witness is then checked by the tiny algebraic AADP relation.
pub fn materialize_transcript_bound_germ_aadp_witness(
    template: &GermAadpVerifierTemplate,
    capsule: &GermArmCapsule,
    transcript_bound: &TranscriptBoundSp1GermProofObject,
) -> Result<GermAadpWitness, GermError> {
    materialize_transcript_bound_germ_aadp_witness_internal(
        template,
        capsule,
        transcript_bound,
        None,
    )
}

pub fn materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings(
    template: &GermAadpVerifierTemplate,
    capsule: &GermArmCapsule,
    transcript_bound: &TranscriptBoundSp1GermProofObject,
    srs: &OrbweaverOpeningSrs,
) -> Result<GermAadpWitness, GermError> {
    materialize_transcript_bound_germ_aadp_witness_internal(
        template,
        capsule,
        transcript_bound,
        Some(srs),
    )
}

fn materialize_transcript_bound_germ_aadp_witness_internal(
    template: &GermAadpVerifierTemplate,
    capsule: &GermArmCapsule,
    transcript_bound: &TranscriptBoundSp1GermProofObject,
    orbweaver_srs: Option<&OrbweaverOpeningSrs>,
) -> Result<GermAadpWitness, GermError> {
    if template.capsule_digest != capsule.digest() {
        return Err(GermError::TemplateCapsuleMismatch);
    }
    if let Some(srs) = orbweaver_srs {
        verify_transcript_bound_sp1_germ_proof_object_with_orbweaver_terminal_openings(
            transcript_bound,
            capsule,
            srs,
        )?;
    } else {
        verify_transcript_bound_sp1_germ_proof_object(transcript_bound, capsule)?;
    }

    let proof_object = &transcript_bound.proof_object;
    let lin_proof = decode_lin_proof(&proof_object.pi_lin)?;
    let mul_proof = decode_mul_sumcheck(&proof_object.pi_mul)?;
    if usize::from(capsule.sumcheck_rounds) != mul_proof.rounds.len() {
        return Err(GermError::SumcheckRoundsMismatch {
            got: mul_proof.rounds.len(),
            expected: usize::from(capsule.sumcheck_rounds),
        });
    }
    if proof_object.lin_terms.len() != template.layout.expected_linear_terms {
        return Err(GermError::TranscriptShapeMismatch {
            which: "linear terms",
            got: proof_object.lin_terms.len(),
            expected: template.layout.expected_linear_terms,
        });
    }
    if proof_object.mul_terms.len() != template.layout.expected_mul_terms {
        return Err(GermError::TranscriptShapeMismatch {
            which: "multiplicative terms",
            got: proof_object.mul_terms.len(),
            expected: template.layout.expected_mul_terms,
        });
    }

    let mut witness = Vec::<SP1ExtensionField>::with_capacity(template.cs.num_variables);
    let push = |witness: &mut Vec<SP1ExtensionField>, value: SP1ExtensionField| {
        witness.push(value);
    };

    push(&mut witness, ext_one()); // safety bit

    // Commitment coordinates + in-boundary mixed commitment accumulator.
    let mut commitment_coords = [ext_zero(); PACKAGE_AJTAI_ROWS * PACKAGE_AJTAI_RING_DIM];
    let mut coord_idx = 0usize;
    for row in &proof_object.shared_object_commitment {
        for coord in row {
            commitment_coords[coord_idx] = *coord;
            push(&mut witness, *coord);
            coord_idx += 1;
        }
    }
    let mut commitment_mix = ext_zero();
    for (coord_idx, coord) in commitment_coords.iter().enumerate() {
        commitment_mix += commitment_mix_weight(coord_idx) * *coord;
    }
    push(&mut witness, commitment_mix);

    let capsule_digest = capsule.digest();
    let challenge_seed_after_digest_const = algebraic_absorb(
        bytes_to_extension(b"sp1-germ/challenge-seed/v2"),
        bytes_to_extension(&capsule_digest),
        1,
    );
    let challenge_absorb =
        challenge_seed_after_digest_const + commitment_mix + absorb_round_constant(2);
    let challenge_absorb_sq = challenge_absorb * challenge_absorb;
    let challenge_seed = challenge_absorb_sq * (challenge_absorb + ext_one());
    push(&mut witness, challenge_absorb_sq);
    push(&mut witness, challenge_seed);

    // r_lin trace
    let r_lin_absorb_seed =
        bytes_to_extension(b"sp1-germ/r_lin/v2") + challenge_seed + absorb_round_constant(11);
    let r_lin_absorb_seed_sq = r_lin_absorb_seed * r_lin_absorb_seed;
    let r_lin_seed_state = r_lin_absorb_seed_sq * (r_lin_absorb_seed + ext_one());
    push(&mut witness, r_lin_absorb_seed_sq);
    push(&mut witness, r_lin_seed_state);
    let r_lin_absorb_idx = r_lin_seed_state + ext_zero() + absorb_round_constant(13);
    let r_lin_absorb_idx_sq = r_lin_absorb_idx * r_lin_absorb_idx;
    let r_lin_idx = r_lin_absorb_idx_sq * (r_lin_absorb_idx + ext_one());
    push(&mut witness, r_lin_absorb_idx_sq);
    push(&mut witness, r_lin_idx);

    // r_mul trace
    let r_mul_absorb_seed =
        bytes_to_extension(b"sp1-germ/r_mul/v2") + challenge_seed + absorb_round_constant(12);
    let r_mul_absorb_seed_sq = r_mul_absorb_seed * r_mul_absorb_seed;
    let r_mul_seed_state = r_mul_absorb_seed_sq * (r_mul_absorb_seed + ext_one());
    push(&mut witness, r_mul_absorb_seed_sq);
    push(&mut witness, r_mul_seed_state);
    let r_mul_absorb_idx = r_mul_seed_state + ext_from_u32(1) + absorb_round_constant(14);
    let r_mul_absorb_idx_sq = r_mul_absorb_idx * r_mul_absorb_idx;
    let r_mul_idx = r_mul_absorb_idx_sq * (r_mul_absorb_idx + ext_one());
    push(&mut witness, r_mul_absorb_idx_sq);
    push(&mut witness, r_mul_idx);

    // sumcheck-seed trace
    let left_seed_absorb = bytes_to_extension(b"sp1-germ/mul-sumcheck-seed/v2")
        + challenge_seed
        + absorb_round_constant(11);
    let left_seed_absorb_sq = left_seed_absorb * left_seed_absorb;
    let left_seed_state = left_seed_absorb_sq * (left_seed_absorb + ext_one());
    push(&mut witness, left_seed_absorb_sq);
    push(&mut witness, left_seed_state);
    let left_idx_absorb = left_seed_state + ext_zero() + absorb_round_constant(13);
    let left_idx_absorb_sq = left_idx_absorb * left_idx_absorb;
    let left_idx_state = left_idx_absorb_sq * (left_idx_absorb + ext_one());
    push(&mut witness, left_idx_absorb_sq);
    push(&mut witness, left_idx_state);
    let left_extra_absorb = left_idx_state + r_mul_idx + absorb_round_constant(17);
    let left_extra_absorb_sq = left_extra_absorb * left_extra_absorb;
    let left_idx = left_extra_absorb_sq * (left_extra_absorb + ext_one());
    push(&mut witness, left_extra_absorb_sq);
    push(&mut witness, left_idx);

    let right_seed_absorb = bytes_to_extension(b"sp1-germ/mul-sumcheck-seed/v2")
        + challenge_seed
        + absorb_round_constant(12);
    let right_seed_absorb_sq = right_seed_absorb * right_seed_absorb;
    let right_seed_state = right_seed_absorb_sq * (right_seed_absorb + ext_one());
    push(&mut witness, right_seed_absorb_sq);
    push(&mut witness, right_seed_state);
    let right_idx_absorb = right_seed_state + ext_from_u32(1) + absorb_round_constant(14);
    let right_idx_absorb_sq = right_idx_absorb * right_idx_absorb;
    let right_idx_state = right_idx_absorb_sq * (right_idx_absorb + ext_one());
    push(&mut witness, right_idx_absorb_sq);
    push(&mut witness, right_idx_state);
    let right_extra_absorb = right_idx_state + r_mul_idx + absorb_round_constant(18);
    let right_extra_absorb_sq = right_extra_absorb * right_extra_absorb;
    let right_idx = right_extra_absorb_sq * (right_extra_absorb + ext_one());
    push(&mut witness, right_extra_absorb_sq);
    push(&mut witness, right_idx);

    let sumcheck_seed = left_idx + (right_idx * (ext_one() + ext_from_u32(7)));
    push(&mut witness, sumcheck_seed);

    for slot in &template.layout.commitment_slots {
        let slot_value = slot_value_from_proof_object(proof_object, *slot)?;
        push(&mut witness, slot_value);
    }

    if template.layout.expected_linear_terms > 0 {
        let lin_left_seed_absorb = bytes_to_extension(b"sp1-germ/lin-point-seed/v2")
            + challenge_seed
            + absorb_round_constant(13);
        let lin_left_seed_absorb_sq = lin_left_seed_absorb * lin_left_seed_absorb;
        let lin_left_seed_state = lin_left_seed_absorb_sq * (lin_left_seed_absorb + ext_one());
        push(&mut witness, lin_left_seed_absorb_sq);
        push(&mut witness, lin_left_seed_state);
        let lin_left_idx_absorb = lin_left_seed_state + ext_from_u32(2) + absorb_round_constant(15);
        let lin_left_idx_absorb_sq = lin_left_idx_absorb * lin_left_idx_absorb;
        let lin_left_idx_state = lin_left_idx_absorb_sq * (lin_left_idx_absorb + ext_one());
        push(&mut witness, lin_left_idx_absorb_sq);
        push(&mut witness, lin_left_idx_state);
        let lin_left_extra_absorb = lin_left_idx_state + r_lin_idx + absorb_round_constant(19);
        let lin_left_extra_absorb_sq = lin_left_extra_absorb * lin_left_extra_absorb;
        let lin_left_idx = lin_left_extra_absorb_sq * (lin_left_extra_absorb + ext_one());
        push(&mut witness, lin_left_extra_absorb_sq);
        push(&mut witness, lin_left_idx);

        let lin_right_seed_absorb = bytes_to_extension(b"sp1-germ/lin-point-seed/v2")
            + challenge_seed
            + absorb_round_constant(14);
        let lin_right_seed_absorb_sq = lin_right_seed_absorb * lin_right_seed_absorb;
        let lin_right_seed_state = lin_right_seed_absorb_sq * (lin_right_seed_absorb + ext_one());
        push(&mut witness, lin_right_seed_absorb_sq);
        push(&mut witness, lin_right_seed_state);
        let lin_right_idx_absorb =
            lin_right_seed_state + ext_from_u32(3) + absorb_round_constant(16);
        let lin_right_idx_absorb_sq = lin_right_idx_absorb * lin_right_idx_absorb;
        let lin_right_idx_state = lin_right_idx_absorb_sq * (lin_right_idx_absorb + ext_one());
        push(&mut witness, lin_right_idx_absorb_sq);
        push(&mut witness, lin_right_idx_state);
        let lin_right_extra_absorb = lin_right_idx_state + r_lin_idx + absorb_round_constant(20);
        let lin_right_extra_absorb_sq = lin_right_extra_absorb * lin_right_extra_absorb;
        let lin_right_idx = lin_right_extra_absorb_sq * (lin_right_extra_absorb + ext_one());
        push(&mut witness, lin_right_extra_absorb_sq);
        push(&mut witness, lin_right_idx);

        let lin_point_seed = lin_left_idx + (lin_right_idx * (ext_one() + ext_from_u32(7)));
        push(&mut witness, lin_point_seed);

        let lin_nvars = mul_sumcheck_nvars(template.layout.expected_linear_terms);
        let mut lin_point = Vec::with_capacity(lin_nvars);
        for var_idx in 0..lin_nvars {
            let lin_point_seed_absorb = bytes_to_extension(b"sp1-germ/lin-point/v1")
                + lin_point_seed
                + absorb_round_constant((var_idx as u32).wrapping_add(11));
            let lin_point_seed_absorb_sq = lin_point_seed_absorb * lin_point_seed_absorb;
            let lin_point_seed_state =
                lin_point_seed_absorb_sq * (lin_point_seed_absorb + ext_one());
            push(&mut witness, lin_point_seed_absorb_sq);
            push(&mut witness, lin_point_seed_state);
            let lin_point_idx_absorb = lin_point_seed_state
                + ext_from_u32(var_idx as u32)
                + absorb_round_constant((var_idx as u32).wrapping_add(13));
            let lin_point_idx_absorb_sq = lin_point_idx_absorb * lin_point_idx_absorb;
            let lin_point_i = lin_point_idx_absorb_sq * (lin_point_idx_absorb + ext_one());
            push(&mut witness, lin_point_idx_absorb_sq);
            push(&mut witness, lin_point_i);
            lin_point.push(lin_point_i);
        }

        let mut lin_table =
            Vec::with_capacity(template.layout.expected_linear_terms.max(1).next_power_of_two());
        for term_idx in 0..template.layout.expected_linear_terms {
            let term = proof_object.lin_terms[term_idx];
            let product = term.coefficient * term.value;
            if template.layout.linear_term_has_explicit_coefficient[term_idx] {
                push(&mut witness, product);
            }
            lin_table.push(product);
        }
        let padded_linear_terms = template.layout.expected_linear_terms.max(1).next_power_of_two();
        while lin_table.len() < padded_linear_terms {
            lin_table.push(ext_zero());
        }
        for r_i in &lin_point {
            let mut next_table = Vec::with_capacity(lin_table.len() / 2);
            for pair in lin_table.chunks_exact(2) {
                let delta = (pair[1] - pair[0]) * *r_i;
                push(&mut witness, delta);
                next_table.push(pair[0] + delta);
            }
            lin_table = next_table;
        }
        push(&mut witness, lin_proof.folded_residual);
    }

    let inv2 = ext_from_u32(2).try_inverse().expect("2 must be invertible in SP1 extension field");
    let inv6 = ext_from_u32(6).try_inverse().expect("6 must be invertible in SP1 extension field");

    let mut round_state = sumcheck_seed;
    let mut point = Vec::with_capacity(mul_proof.nvars as usize);
    let mut sampled = Vec::with_capacity(mul_proof.rounds.len());
    for round in &mul_proof.rounds {
        for eval in round.evaluations {
            push(&mut witness, eval);
        }

        let r_sc = materialize_challenge_from_seed_trace(
            &mut witness,
            MUL_SUMCHECK_ROUND_CHALLENGE_DOMAIN,
            round_state,
            0,
        );

        sampled.push(r_sc);

        let e = round.evaluations;
        let c3 = (e[3] - (ext_from_u32(3) * e[2]) + (ext_from_u32(3) * e[1]) - e[0]) * inv6;
        let c2 = (e[2] - (ext_from_u32(2) * e[1]) + e[0]) * inv2;
        let d1 = e[1] - e[0];
        let m_a = (r_sc - ext_from_u32(2)) * c3;
        let m_b = (c2 + m_a) * (r_sc - ext_from_u32(1));
        let m_c = (d1 + m_b) * r_sc;
        push(&mut witness, m_a);
        push(&mut witness, m_b);
        push(&mut witness, m_c);

        let claimed_next = e[0] + m_c;
        push(&mut witness, claimed_next);
        round_state =
            materialize_sumcheck_round_state_trace(&mut witness, round_state, &e, claimed_next);
    }

    push(&mut witness, mul_proof.opening.a);
    push(&mut witness, mul_proof.opening.b);
    push(&mut witness, mul_proof.opening.c);
    push(&mut witness, mul_proof.opening.d);

    let opening_lhs_product = mul_proof.opening.a * mul_proof.opening.b;
    let opening_rhs_product = mul_proof.opening.c * mul_proof.opening.d;
    push(&mut witness, opening_lhs_product);
    push(&mut witness, opening_rhs_product);
    if template.layout.orbweaver_terminal_pi_len > 0 {
        let srs = orbweaver_srs.ok_or_else(|| {
            GermError::InvalidOrbweaverSrs("orbweaver template requires SRS".to_string())
        })?;
        let orbweaver_data = build_orbweaver_transcript_template_data(capsule, proof_object, srs)?;
        if template.layout.orbweaver_terminal_pi_len != orbweaver_data.proof_pi_len
            || template.layout.orbweaver_aggregated_proof_count
                != ORBWEAVER_AGGREGATED_SCALAR_OPENINGS
        {
            return Err(GermError::TranscriptShapeMismatch {
                which: "orbweaver aggregated proof shape",
                got: orbweaver_data.proof_pi_len,
                expected: template.layout.orbweaver_terminal_pi_len,
            });
        }
        let proofs = decode_orbweaver_aggregated_scalar_openings(&proof_object.pi_mul_terminal_openings)?;
        let proof_blocks_by_proof = proofs
            .iter()
            .map(packed_scalar_opening_blocks)
            .collect::<Vec<_>>();
        let all_proof_blocks = packed_scalar_opening_family_blocks(&proofs);
        for packed in &all_proof_blocks {
            push(&mut witness, *packed);
        }
        let mut c_flat_ext = ext_zero();
        for (field_multipliers, term) in orbweaver_data
            .c_flat_field_multipliers
            .iter()
            .zip(proof_object.mul_terms.iter())
        {
            let fields = [term.a, term.b, term.c, term.d];
            for field_idx in 0..4 {
                c_flat_ext += field_multipliers[field_idx] * fields[field_idx];
            }
        }
        let opening_values = [mul_proof.opening.a, mul_proof.opening.b, mul_proof.opening.c, mul_proof.opening.d];
        for proof_idx in 0..ORBWEAVER_AGGREGATED_SCALAR_OPENINGS {
            let mut lhs_ext = ext_zero();
            for (packed, coeff) in proof_blocks_by_proof[proof_idx]
                .iter()
                .zip(orbweaver_data.lhs_block_multipliers.iter())
            {
                lhs_ext += *coeff * *packed;
            }
            let mut opening_ext =
                lhs_ext + ((-ext_from_base_field(orbweaver_data.aggregated_vk_values[proof_idx])) * c_flat_ext);
            for field_idx in 0..4 {
                opening_ext +=
                    orbweaver_data.aggregated_output_multipliers[proof_idx][field_idx]
                        * opening_values[field_idx];
            }
            let opening_limbs =
                <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(
                    &opening_ext,
                );
            for limb in opening_limbs {
                push(&mut witness, ext_from_base_field(*limb));
            }
        }

        let mut jl_norm_sum = 0u64;
        for row in &orbweaver_data.jl_rows {
            let row_weights = row
                .iter()
                .map(|coeff| match *coeff {
                    1 => SP1Field::one(),
                    -1 => -SP1Field::one(),
                    _ => SP1Field::zero(),
                })
                .collect::<Vec<_>>();
            let row_multipliers =
                pack_scalar_weights_to_extension_multipliers(row_weights.as_slice());
            let mut projection_ext = ext_zero();
            for (packed, coeff) in all_proof_blocks.iter().zip(row_multipliers.iter()) {
                projection_ext += *coeff * *packed;
            }
            let projection_limbs =
                <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(
                    &projection_ext,
                );
            for limb in projection_limbs {
                push(&mut witness, ext_from_base_field(*limb));
            }
            let square = projection_limbs[0] * projection_limbs[0];
            push(&mut witness, ext_from_base_field(square));
            jl_norm_sum = jl_norm_sum
                .checked_add(u64::from(square.as_canonical_u32()))
                .ok_or_else(|| {
                    GermError::AadpConstraintUnsatisfied(
                        "orbweaver JL squared norm overflow".to_string(),
                    )
                })?;
        }
        let bits = decompose_unsigned_bits(jl_norm_sum, ORBWEAVER_JL_NORM_BITS)
            .map_err(GermError::AadpConstraintUnsatisfied)?;
        for bit in bits {
            push(&mut witness, ext_from_u32(bit as u32));
        }
    }
    let delta = opening_lhs_product - opening_rhs_product;
    if capsule.sumcheck_rounds > 0 {
        for var_idx in 0..usize::from(capsule.sumcheck_rounds) {
            let point_seed_absorb = bytes_to_extension(b"sp1-germ/mul-point/v1")
                + sumcheck_seed
                + absorb_round_constant((var_idx as u32).wrapping_add(11));
            let point_seed_absorb_sq = point_seed_absorb * point_seed_absorb;
            let point_seed_state = point_seed_absorb_sq * (point_seed_absorb + ext_one());
            push(&mut witness, point_seed_absorb_sq);
            push(&mut witness, point_seed_state);
            let point_idx_absorb = point_seed_state
                + ext_from_u32(var_idx as u32)
                + absorb_round_constant((var_idx as u32).wrapping_add(13));
            let point_idx_absorb_sq = point_idx_absorb * point_idx_absorb;
            let point_i = point_idx_absorb_sq * (point_idx_absorb + ext_one());
            push(&mut witness, point_idx_absorb_sq);
            push(&mut witness, point_i);
            point.push(point_i);
        }

        let mut eq_eval = ext_zero();
        for (round_idx, (r_i, s_i)) in point.iter().zip(sampled.iter()).enumerate() {
            let rs = *r_i * *s_i;
            let factor = ext_one() - *r_i - *s_i + (ext_from_u32(2) * rs);
            if round_idx == 0 {
                eq_eval = factor;
            } else {
                eq_eval *= factor;
            }
            push(&mut witness, rs);
            push(&mut witness, factor);
            push(&mut witness, eq_eval);
        }
        push(&mut witness, eq_eval * delta);
        push(&mut witness, delta);
    }

    if witness.len() != template.cs.num_variables {
        return Err(GermError::AadpConstraintUnsatisfied(format!(
            "materialized witness length mismatch: got={} expected={}",
            witness.len(),
            template.cs.num_variables
        )));
    }

    let witness = GermAadpWitness { witness };
    template.check_witness(&witness)?;
    Ok(witness)
}

fn evaluate_linear_relation(
    bundle: &Sp1GermBundle,
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
    r_lin: &SP1ExtensionField,
    domain: &[u8],
) -> Result<GermRelationCheck, GermError> {
    if bundle.pi_lin.is_empty() {
        return Err(GermError::EmptyProof("lin"));
    }
    if bundle.lin_terms.is_empty() {
        return Err(GermError::EmptyTerms("lin"));
    }
    let expected_commitment = compute_transcript_commitment(&bundle.lin_terms, &bundle.mul_terms);
    if expected_commitment != bundle.shared_object_commitment {
        return Err(GermError::CommitmentRootMismatch);
    }
    let expected_commitment_root = compute_commitment_root(&expected_commitment);
    if &expected_commitment_root != commitment_root {
        return Err(GermError::CommitmentRootMismatch);
    }
    let parsed_lin_proof = decode_lin_proof(&bundle.pi_lin)?;
    let expected_term_count = u32::try_from(bundle.lin_terms.len())
        .map_err(|_| GermError::TooManyLinearTerms(bundle.lin_terms.len()))?;
    if parsed_lin_proof.term_count != expected_term_count {
        return Err(GermError::LinProofTranscriptMismatch);
    }
    let folded_residual =
        fold_linear_terms(&bundle.lin_terms, public_values_digest, &expected_commitment, r_lin);
    if parsed_lin_proof.folded_residual != folded_residual {
        return Err(GermError::LinProofTranscriptMismatch);
    }
    if parsed_lin_proof.ajtai_commitment != expected_commitment {
        return Err(GermError::LinProofTranscriptMismatch);
    }
    let terms_digest = digest_linear_terms(&bundle.lin_terms);
    let fingerprint = compute_relation_fingerprint(
        domain,
        public_values_digest,
        commitment_root,
        r_lin,
        &terms_digest,
        &folded_residual,
        &bundle.pi_lin,
    );
    Ok(GermRelationCheck { folded_residual, fingerprint })
}

fn evaluate_multiplicative_relation(
    bundle: &Sp1GermBundle,
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
    r_mul: &SP1ExtensionField,
    domain: &[u8],
) -> Result<GermRelationCheck, GermError> {
    evaluate_multiplicative_relation_internal(
        bundle,
        public_values_digest,
        commitment_root,
        r_mul,
        domain,
        None,
    )
}

fn evaluate_multiplicative_relation_internal(
    bundle: &Sp1GermBundle,
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
    r_mul: &SP1ExtensionField,
    domain: &[u8],
    orbweaver_srs: Option<&OrbweaverOpeningSrs>,
) -> Result<GermRelationCheck, GermError> {
    if bundle.pi_mul.is_empty() {
        return Err(GermError::EmptyProof("mul"));
    }
    if bundle.mul_terms.is_empty() {
        return Err(GermError::EmptyTerms("mul"));
    }
    let expected_commitment = compute_transcript_commitment(&bundle.lin_terms, &bundle.mul_terms);
    if expected_commitment != bundle.shared_object_commitment {
        return Err(GermError::CommitmentRootMismatch);
    }
    let expected_commitment_root = compute_commitment_root(&expected_commitment);
    if &expected_commitment_root != commitment_root {
        return Err(GermError::CommitmentRootMismatch);
    }
    let parsed_sumcheck = decode_mul_sumcheck(&bundle.pi_mul)?;
    let sumcheck_seed = derive_mul_sumcheck_seed(public_values_digest, &expected_commitment, r_mul);
    let folded_residual = fold_multiplicative_terms(&bundle.mul_terms, &sumcheck_seed);
    if !is_zero_ext(&folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    let expected_sumcheck = prove_mul_sumcheck(&sumcheck_seed, &bundle.mul_terms);
    if expected_sumcheck != parsed_sumcheck {
        return Err(GermError::SumcheckTranscriptMismatch);
    }
    verify_mul_sumcheck(&sumcheck_seed, &parsed_sumcheck)?;
    let relation_digest = digest_multiplicative_relation(&bundle.mul_terms, &parsed_sumcheck);
    let proof_bytes = if let Some(srs) = orbweaver_srs {
        let round_challenges = collect_sumcheck_round_challenges(&sumcheck_seed, &parsed_sumcheck)?;
        let weights = eq_table(round_challenges.as_slice());
        let aggregation_coeffs = derive_orbweaver_aggregation_coeffs(
            public_values_digest,
            &bundle.shared_object_commitment,
            bundle.pi_mul.as_slice(),
        );
        let expected_scalar_values = expected_orbweaver_scalar_image_values(&parsed_sumcheck);
        verify_aggregated_scalar_image_openings_from_mul_terms(
            srs,
            bundle.mul_terms.as_slice(),
            weights.as_slice(),
            &aggregation_coeffs,
            &expected_scalar_values,
            &bundle.pi_mul_terminal_openings,
        )
        .map_err(|msg| GermError::MulTerminalOpeningFailed { which: "terminal", msg })?;
        combined_mul_proof_bytes(
            &bundle.pi_mul,
            &bundle.pi_mul_terminal_openings,
            Some(&digest_srs(srs)),
        )
    } else {
        combined_mul_proof_bytes(&bundle.pi_mul, &bundle.pi_mul_terminal_openings, None)
    };
    let fingerprint = compute_relation_fingerprint(
        domain,
        public_values_digest,
        commitment_root,
        r_mul,
        &relation_digest,
        &folded_residual,
        proof_bytes.as_slice(),
    );
    Ok(GermRelationCheck { folded_residual, fingerprint })
}

/// Evaluate globally mixed linear residual at the designated point.
fn fold_linear_terms(
    terms: &[Sp1LinTerm],
    public_values_digest: &[u8; 32],
    commitment: &Sp1PackageCommitment,
    r_lin: &SP1ExtensionField,
) -> SP1ExtensionField {
    let nvars = mul_sumcheck_nvars(terms.len());
    let lin_point_seed = derive_lin_point_seed(public_values_digest, commitment, r_lin);
    let lin_point = derive_lin_point(&lin_point_seed, nvars);
    let padded = terms.len().max(1).next_power_of_two();
    let mut table = vec![ext_zero(); padded];
    for (idx, term) in terms.iter().enumerate() {
        table[idx] = term.coefficient * term.value;
    }
    evaluate_mle_table(table, lin_point.as_slice())
}

/// Evaluate globally mixed multiplicative residual at the designated point.
fn fold_multiplicative_terms(
    terms: &[Sp1MulTerm],
    sumcheck_seed: &SP1ExtensionField,
) -> SP1ExtensionField {
    let (total_checks, a_vals, b_vals, c_vals, d_vals) = build_mul_tables(terms);
    let nvars = mul_sumcheck_nvars(total_checks);
    let point = derive_mul_point(sumcheck_seed, nvars);
    let mut residual_table = Vec::with_capacity(a_vals.len());
    for idx in 0..a_vals.len() {
        residual_table.push((a_vals[idx] * b_vals[idx]) - (c_vals[idx] * d_vals[idx]));
    }
    evaluate_mle_table(residual_table, point.as_slice())
}

fn derive_mul_sumcheck_seed(
    public_values_digest: &[u8; 32],
    commitment: &Sp1PackageCommitment,
    r_mul: &SP1ExtensionField,
) -> SP1ExtensionField {
    let challenge_seed = derive_commitment_bound_seed(public_values_digest, commitment);
    derive_mul_sumcheck_seed_from_challenge_seed(&challenge_seed, r_mul)
}

fn derive_mul_sumcheck_seed_from_challenge_seed(
    challenge_seed: &SP1ExtensionField,
    r_mul: &SP1ExtensionField,
) -> SP1ExtensionField {
    derive_seed_from_challenge_seed(
        b"sp1-germ/mul-sumcheck-seed/v2",
        *challenge_seed,
        core::slice::from_ref(r_mul),
        0,
    )
}

fn derive_seed_from_challenge_seed(
    domain: &[u8],
    challenge_seed: SP1ExtensionField,
    extras: &[SP1ExtensionField],
    seed_idx: u32,
) -> SP1ExtensionField {
    let left = derive_algebraic_challenge(domain, challenge_seed, extras, seed_idx);
    let right =
        derive_algebraic_challenge(domain, challenge_seed, extras, seed_idx.wrapping_add(1));
    left + (right * (ext_one() + ext_from_u32(7)))
}

fn derive_lin_point_seed(
    public_values_digest: &[u8; 32],
    commitment: &Sp1PackageCommitment,
    r_lin: &SP1ExtensionField,
) -> SP1ExtensionField {
    let challenge_seed = derive_commitment_bound_seed(public_values_digest, commitment);
    derive_seed_from_challenge_seed(
        b"sp1-germ/lin-point-seed/v2",
        challenge_seed,
        core::slice::from_ref(r_lin),
        2,
    )
}

fn derive_lin_point(seed: &SP1ExtensionField, nvars: usize) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(nvars);
    for var_idx in 0..nvars {
        out.push(derive_algebraic_challenge(b"sp1-germ/lin-point/v1", *seed, &[], var_idx as u32));
    }
    out
}

fn derive_mul_point(seed: &SP1ExtensionField, nvars: usize) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(nvars);
    for var_idx in 0..nvars {
        out.push(derive_algebraic_challenge(b"sp1-germ/mul-point/v1", *seed, &[], var_idx as u32));
    }
    out
}

fn derive_sumcheck_round_challenge(round_state: &SP1ExtensionField) -> SP1ExtensionField {
    derive_algebraic_challenge(MUL_SUMCHECK_ROUND_CHALLENGE_DOMAIN, *round_state, &[], 0)
}

fn update_sumcheck_round_state(
    prev_state: &SP1ExtensionField,
    evals: &[SP1ExtensionField; 4],
    claim: &SP1ExtensionField,
) -> SP1ExtensionField {
    let mut state =
        algebraic_absorb(bytes_to_extension(MUL_SUMCHECK_ROUND_STATE_DOMAIN), *prev_state, 11);
    state = algebraic_absorb(state, evals[0], 13);
    state = algebraic_absorb(state, evals[1], 15);
    state = algebraic_absorb(state, evals[2], 17);
    state = algebraic_absorb(state, evals[3], 19);
    algebraic_absorb(state, *claim, 21)
}

fn interpolate_0123(
    evals: &[SP1ExtensionField; 4],
    x: SP1ExtensionField,
) -> Result<SP1ExtensionField, GermError> {
    let xs = [ext_from_u32(0), ext_from_u32(1), ext_from_u32(2), ext_from_u32(3)];
    let mut acc = ext_zero();
    for i in 0..4 {
        let mut num = ext_one();
        let mut den = ext_one();
        for j in 0..4 {
            if i == j {
                continue;
            }
            num *= x - xs[j];
            den *= xs[i] - xs[j];
        }
        let den_inv = den.try_inverse().ok_or(GermError::SumcheckInterpolationDenominatorZero)?;
        acc += evals[i] * (num * den_inv);
    }
    Ok(acc)
}

fn linear_form_constant(value: SP1ExtensionField) -> AadpLinearForm<SP1ExtensionField> {
    AadpLinearForm { constant: value, terms: Vec::new() }
}

fn linear_form_single_var(var_idx: usize) -> AadpLinearForm<SP1ExtensionField> {
    AadpLinearForm { constant: ext_zero(), terms: vec![(var_idx, ext_one())] }
}

fn linear_form_add_forms(
    lhs: &AadpLinearForm<SP1ExtensionField>,
    rhs: &AadpLinearForm<SP1ExtensionField>,
) -> AadpLinearForm<SP1ExtensionField> {
    let mut terms = lhs.terms.clone();
    terms.extend(rhs.terms.iter().copied());
    AadpLinearForm { constant: lhs.constant + rhs.constant, terms }
}

fn linear_form_sub_forms(
    lhs: &AadpLinearForm<SP1ExtensionField>,
    rhs: &AadpLinearForm<SP1ExtensionField>,
) -> AadpLinearForm<SP1ExtensionField> {
    let mut terms = lhs.terms.clone();
    terms.extend(rhs.terms.iter().map(|(idx, coeff)| (*idx, -*coeff)));
    AadpLinearForm { constant: lhs.constant - rhs.constant, terms }
}

fn linear_form_scale(
    form: &AadpLinearForm<SP1ExtensionField>,
    scale: SP1ExtensionField,
) -> AadpLinearForm<SP1ExtensionField> {
    AadpLinearForm {
        constant: form.constant * scale,
        terms: form.terms.iter().map(|(idx, coeff)| (*idx, *coeff * scale)).collect(),
    }
}

fn add_mul_equals_var(
    constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>,
    a: AadpLinearForm<SP1ExtensionField>,
    b: AadpLinearForm<SP1ExtensionField>,
    out_idx: usize,
) {
    constraints.push(AadpMulConstraint {
        a,
        b,
        c: linear_form_constant(ext_one()),
        d: linear_form_single_var(out_idx),
    });
}

fn absorb_round_constant(round_idx: u32) -> SP1ExtensionField {
    ext_from_u32(round_idx.wrapping_mul(17).wrapping_add(1))
}

fn absorb_linear_form(
    prev_idx: Option<usize>,
    prev_const: SP1ExtensionField,
    value_idx: Option<usize>,
    value_const: SP1ExtensionField,
    round_idx: u32,
) -> AadpLinearForm<SP1ExtensionField> {
    let mut terms = Vec::new();
    if let Some(idx) = prev_idx {
        terms.push((idx, ext_one()));
    }
    if let Some(idx) = value_idx {
        terms.push((idx, ext_one()));
    }
    AadpLinearForm { constant: prev_const + value_const + absorb_round_constant(round_idx), terms }
}

fn add_absorb_constraints<A: FnMut() -> usize>(
    constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>,
    alloc: &mut A,
    prev_idx: Option<usize>,
    prev_const: SP1ExtensionField,
    value_idx: Option<usize>,
    value_const: SP1ExtensionField,
    round_idx: u32,
) -> (usize, usize) {
    let absorb_form = absorb_linear_form(prev_idx, prev_const, value_idx, value_const, round_idx);
    let sq_idx = alloc();
    add_mul_equals_var(constraints, absorb_form.clone(), absorb_form.clone(), sq_idx);
    let next_idx = alloc();
    add_mul_equals_var(
        constraints,
        linear_form_single_var(sq_idx),
        AadpLinearForm { constant: absorb_form.constant + ext_one(), terms: absorb_form.terms },
        next_idx,
    );
    (sq_idx, next_idx)
}

fn add_linear_zero_constraint(
    constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>,
    form: AadpLinearForm<SP1ExtensionField>,
) {
    constraints.push(AadpMulConstraint {
        a: form,
        b: linear_form_constant(ext_one()),
        c: linear_form_constant(ext_zero()),
        d: linear_form_constant(ext_one()),
    });
}

fn add_bit_constraint(constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>, var_idx: usize) {
    constraints.push(AadpMulConstraint {
        a: linear_form_single_var(var_idx),
        b: linear_form_single_var(var_idx),
        c: linear_form_constant(ext_one()),
        d: linear_form_single_var(var_idx),
    });
}

fn add_challenge_from_seed_constraints<A: FnMut() -> usize>(
    constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>,
    alloc: &mut A,
    domain: &[u8],
    seed_idx: usize,
    challenge_index: u32,
) -> usize {
    let (_seed_sq_idx, seed_state_idx) = add_absorb_constraints(
        constraints,
        alloc,
        None,
        bytes_to_extension(domain),
        Some(seed_idx),
        ext_zero(),
        challenge_index.wrapping_add(11),
    );
    let (_idx_sq_idx, challenge_idx_var) = add_absorb_constraints(
        constraints,
        alloc,
        Some(seed_state_idx),
        ext_zero(),
        None,
        ext_from_u32(challenge_index),
        challenge_index.wrapping_add(13),
    );
    challenge_idx_var
}

fn add_sumcheck_round_state_constraints<A: FnMut() -> usize>(
    constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>,
    alloc: &mut A,
    prev_state_idx: usize,
    eval_indices: [usize; 4],
    claim_idx: usize,
) -> usize {
    let (_prev_sq_idx, after_prev_idx) = add_absorb_constraints(
        constraints,
        alloc,
        None,
        bytes_to_extension(MUL_SUMCHECK_ROUND_STATE_DOMAIN),
        Some(prev_state_idx),
        ext_zero(),
        11,
    );
    let (_eval0_sq_idx, after_eval0_idx) = add_absorb_constraints(
        constraints,
        alloc,
        Some(after_prev_idx),
        ext_zero(),
        Some(eval_indices[0]),
        ext_zero(),
        13,
    );
    let (_eval1_sq_idx, after_eval1_idx) = add_absorb_constraints(
        constraints,
        alloc,
        Some(after_eval0_idx),
        ext_zero(),
        Some(eval_indices[1]),
        ext_zero(),
        15,
    );
    let (_eval2_sq_idx, after_eval2_idx) = add_absorb_constraints(
        constraints,
        alloc,
        Some(after_eval1_idx),
        ext_zero(),
        Some(eval_indices[2]),
        ext_zero(),
        17,
    );
    let (_eval3_sq_idx, after_eval3_idx) = add_absorb_constraints(
        constraints,
        alloc,
        Some(after_eval2_idx),
        ext_zero(),
        Some(eval_indices[3]),
        ext_zero(),
        19,
    );
    let (_claim_sq_idx, next_state_idx) = add_absorb_constraints(
        constraints,
        alloc,
        Some(after_eval3_idx),
        ext_zero(),
        Some(claim_idx),
        ext_zero(),
        21,
    );
    next_state_idx
}

fn materialize_absorb_trace(
    witness: &mut Vec<SP1ExtensionField>,
    prev_state: Option<SP1ExtensionField>,
    prev_const: SP1ExtensionField,
    value: Option<SP1ExtensionField>,
    value_const: SP1ExtensionField,
    round_idx: u32,
) -> SP1ExtensionField {
    let absorb = prev_state.unwrap_or_else(ext_zero)
        + prev_const
        + value.unwrap_or_else(ext_zero)
        + value_const
        + absorb_round_constant(round_idx);
    let absorb_sq = absorb * absorb;
    let next_state = absorb_sq * (absorb + ext_one());
    witness.push(absorb_sq);
    witness.push(next_state);
    next_state
}

fn materialize_challenge_from_seed_trace(
    witness: &mut Vec<SP1ExtensionField>,
    domain: &[u8],
    seed: SP1ExtensionField,
    challenge_index: u32,
) -> SP1ExtensionField {
    let seed_state = materialize_absorb_trace(
        witness,
        None,
        bytes_to_extension(domain),
        Some(seed),
        ext_zero(),
        challenge_index.wrapping_add(11),
    );
    materialize_absorb_trace(
        witness,
        Some(seed_state),
        ext_zero(),
        None,
        ext_from_u32(challenge_index),
        challenge_index.wrapping_add(13),
    )
}

fn materialize_sumcheck_round_state_trace(
    witness: &mut Vec<SP1ExtensionField>,
    prev_state: SP1ExtensionField,
    evals: &[SP1ExtensionField; 4],
    claim: SP1ExtensionField,
) -> SP1ExtensionField {
    let after_prev = materialize_absorb_trace(
        witness,
        None,
        bytes_to_extension(MUL_SUMCHECK_ROUND_STATE_DOMAIN),
        Some(prev_state),
        ext_zero(),
        11,
    );
    let after_eval0 = materialize_absorb_trace(
        witness,
        Some(after_prev),
        ext_zero(),
        Some(evals[0]),
        ext_zero(),
        13,
    );
    let after_eval1 = materialize_absorb_trace(
        witness,
        Some(after_eval0),
        ext_zero(),
        Some(evals[1]),
        ext_zero(),
        15,
    );
    let after_eval2 = materialize_absorb_trace(
        witness,
        Some(after_eval1),
        ext_zero(),
        Some(evals[2]),
        ext_zero(),
        17,
    );
    let after_eval3 = materialize_absorb_trace(
        witness,
        Some(after_eval2),
        ext_zero(),
        Some(evals[3]),
        ext_zero(),
        19,
    );
    materialize_absorb_trace(witness, Some(after_eval3), ext_zero(), Some(claim), ext_zero(), 21)
}

fn eq_table(point: &[SP1ExtensionField]) -> Vec<SP1ExtensionField> {
    if point.is_empty() {
        return vec![ext_one()];
    }
    let nvars = point.len();
    let size = 1usize << nvars;
    let mut out = vec![ext_zero(); size];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut acc = ext_one();
        for (var_idx, r_i) in point.iter().enumerate() {
            let bit = (idx >> var_idx) & 1;
            acc *= if bit == 0 { ext_one() - *r_i } else { *r_i };
        }
        *slot = acc;
    }
    out
}

fn fold_mle_table(table: &[SP1ExtensionField], r: SP1ExtensionField) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(table.len() / 2);
    for pair in table.chunks_exact(2) {
        out.push(pair[0] + (pair[1] - pair[0]) * r);
    }
    out
}

fn evaluate_mle_table(
    mut table: Vec<SP1ExtensionField>,
    point: &[SP1ExtensionField],
) -> SP1ExtensionField {
    if point.is_empty() {
        return table[0];
    }
    for r in point {
        table = fold_mle_table(table.as_slice(), *r);
    }
    table[0]
}

fn mul_sumcheck_nvars(total_checks: usize) -> usize {
    if total_checks <= 1 {
        0
    } else {
        total_checks.next_power_of_two().trailing_zeros() as usize
    }
}

fn build_mul_tables(
    terms: &[Sp1MulTerm],
) -> (
    usize,
    Vec<SP1ExtensionField>,
    Vec<SP1ExtensionField>,
    Vec<SP1ExtensionField>,
    Vec<SP1ExtensionField>,
) {
    let total_checks = terms.len();
    let padded = total_checks.max(1).next_power_of_two();
    let mut a_vals = vec![ext_zero(); padded];
    let mut b_vals = vec![ext_zero(); padded];
    let mut c_vals = vec![ext_zero(); padded];
    let mut d_vals = vec![ext_zero(); padded];
    for (idx, term) in terms.iter().enumerate() {
        a_vals[idx] = term.a;
        b_vals[idx] = term.b;
        c_vals[idx] = term.c;
        d_vals[idx] = term.d;
    }
    (total_checks, a_vals, b_vals, c_vals, d_vals)
}

fn prove_mul_sumcheck(seed: &SP1ExtensionField, terms: &[Sp1MulTerm]) -> Sp1MulSumcheckProof {
    let (total_checks, mut a_vals, mut b_vals, mut c_vals, mut d_vals) = build_mul_tables(terms);
    let nvars = mul_sumcheck_nvars(total_checks);
    if nvars == 0 {
        return Sp1MulSumcheckProof {
            nvars: 0,
            rounds: Vec::new(),
            opening: Sp1MulTerm::new(a_vals[0], b_vals[0], c_vals[0], d_vals[0]),
        };
    }
    let point = derive_mul_point(seed, nvars);
    let mut eq = eq_table(point.as_slice());
    let mut rounds = Vec::with_capacity(nvars);
    let ts = [ext_from_u32(0), ext_from_u32(1), ext_from_u32(2), ext_from_u32(3)];
    let mut round_state = *seed;
    for _round_idx in 0..nvars {
        let mut evals = [ext_zero(), ext_zero(), ext_zero(), ext_zero()];
        for idx in 0..(a_vals.len() / 2) {
            let i0 = 2 * idx;
            let i1 = i0 + 1;
            let a0 = a_vals[i0];
            let a1 = a_vals[i1];
            let b0 = b_vals[i0];
            let b1 = b_vals[i1];
            let c0 = c_vals[i0];
            let c1 = c_vals[i1];
            let d0 = d_vals[i0];
            let d1 = d_vals[i1];
            let eq0 = eq[i0];
            let eq1 = eq[i1];
            for (t_idx, t) in ts.iter().copied().enumerate() {
                let a_t = a0 + (a1 - a0) * t;
                let b_t = b0 + (b1 - b0) * t;
                let c_t = c0 + (c1 - c0) * t;
                let d_t = d0 + (d1 - d0) * t;
                let eq_t = eq0 + (eq1 - eq0) * t;
                evals[t_idx] += eq_t * ((a_t * b_t) - (c_t * d_t));
            }
        }
        let round = Sp1MulSumcheckRound { evaluations: evals };
        rounds.push(round);
        let r_sc = derive_sumcheck_round_challenge(&round_state);
        let claim_next = interpolate_0123(&evals, r_sc)
            .expect("sumcheck interpolation over points 0,1,2,3 should be well-defined");
        round_state = update_sumcheck_round_state(&round_state, &evals, &claim_next);
        a_vals = fold_mle_table(a_vals.as_slice(), r_sc);
        b_vals = fold_mle_table(b_vals.as_slice(), r_sc);
        c_vals = fold_mle_table(c_vals.as_slice(), r_sc);
        d_vals = fold_mle_table(d_vals.as_slice(), r_sc);
        eq = fold_mle_table(eq.as_slice(), r_sc);
    }
    Sp1MulSumcheckProof {
        nvars: nvars as u16,
        rounds,
        opening: Sp1MulTerm::new(a_vals[0], b_vals[0], c_vals[0], d_vals[0]),
    }
}

fn verify_mul_sumcheck(
    seed: &SP1ExtensionField,
    proof: &Sp1MulSumcheckProof,
) -> Result<(), GermError> {
    let nvars = proof.nvars as usize;
    if proof.rounds.len() != nvars {
        return Err(GermError::SumcheckRoundsMismatch { got: proof.rounds.len(), expected: nvars });
    }
    if nvars == 0 {
        let residual = (proof.opening.a * proof.opening.b) - (proof.opening.c * proof.opening.d);
        if !is_zero_ext(&residual) {
            return Err(GermError::SumcheckFinalResidualNonZero);
        }
        return Ok(());
    }

    let point = derive_mul_point(seed, nvars);
    let mut claimed = ext_zero();
    let mut sampled = Vec::with_capacity(nvars);
    let mut round_state = *seed;
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        let evals = round.evaluations;
        if evals[0] + evals[1] != claimed {
            return Err(GermError::SumcheckIdentityFailed(round_idx));
        }
        let r_sc = derive_sumcheck_round_challenge(&round_state);
        sampled.push(r_sc);
        claimed = interpolate_0123(&evals, r_sc)?;
        round_state = update_sumcheck_round_state(&round_state, &evals, &claimed);
    }

    let mut eq_eval = ext_one();
    for (r_i, s_i) in point.iter().zip(sampled.iter()) {
        eq_eval *= (ext_one() - *r_i) * (ext_one() - *s_i) + (*r_i * *s_i);
    }
    let final_residual = claimed
        - (eq_eval * ((proof.opening.a * proof.opening.b) - (proof.opening.c * proof.opening.d)));
    if !is_zero_ext(&final_residual) {
        return Err(GermError::SumcheckFinalResidualNonZero);
    }
    Ok(())
}

fn build_orbweaver_terminal_openings_internal(
    bundle: &Sp1GermBundle,
    public_values_digest: &[u8; 32],
    r_mul: &SP1ExtensionField,
    srs: &OrbweaverOpeningSrs,
) -> Result<crate::bundle::Sp1MulTerminalOpeningProofs, GermError> {
    let commitment_root = compute_commitment_root(&bundle.shared_object_commitment);
    let _ = evaluate_multiplicative_relation(
        bundle,
        public_values_digest,
        &commitment_root,
        r_mul,
        b"sp1-germ/mul_bind/v1",
    )?;

    let mul_proof = decode_mul_sumcheck(&bundle.pi_mul)?;
    let sumcheck_seed =
        derive_mul_sumcheck_seed(public_values_digest, &bundle.shared_object_commitment, r_mul);
    let round_challenges = collect_sumcheck_round_challenges(&sumcheck_seed, &mul_proof)?;
    let weights = eq_table(round_challenges.as_slice());
    let aggregation_coeffs = derive_orbweaver_aggregation_coeffs(
        public_values_digest,
        &bundle.shared_object_commitment,
        bundle.pi_mul.as_slice(),
    );
    let openings = build_aggregated_scalar_image_openings_from_mul_terms(
        srs,
        bundle.mul_terms.as_slice(),
        weights.as_slice(),
        &aggregation_coeffs,
    )
    .map_err(|msg| GermError::MulTerminalOpeningFailed { which: "terminal", msg })?;
    let expected_scalar_values = expected_orbweaver_scalar_image_values(&mul_proof);
    let proofs = decode_orbweaver_aggregated_scalar_openings(&openings)?;
    for (which, coeffs, proof) in [
        ("agg0", &aggregation_coeffs[0], &proofs[0]),
        ("agg1", &aggregation_coeffs[1], &proofs[1]),
        ("agg2", &aggregation_coeffs[2], &proofs[2]),
        ("agg3", &aggregation_coeffs[3], &proofs[3]),
    ] {
        if proof.opened_value != aggregate_scalar_image_value(&expected_scalar_values, coeffs) {
            return Err(GermError::MulTerminalOpeningMismatch(which));
        }
    }
    Ok(openings)
}

fn combined_mul_proof_bytes(
    pi_mul: &[u8],
    openings: &Sp1MulTerminalOpeningProofs,
    srs_digest: Option<&[u8; 32]>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        pi_mul.len()
            + openings.a.len()
            + openings.b.len()
            + openings.c.len()
            + openings.d.len()
            + 56,
    );
    out.extend_from_slice(&(pi_mul.len() as u32).to_le_bytes());
    out.extend_from_slice(pi_mul);
    for bytes in [&openings.a, &openings.b, &openings.c, &openings.d] {
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    if let Some(digest) = srs_digest {
        out.extend_from_slice(&(digest.len() as u32).to_le_bytes());
        out.extend_from_slice(digest);
    } else {
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    out
}

fn collect_sumcheck_round_challenges(
    seed: &SP1ExtensionField,
    proof: &Sp1MulSumcheckProof,
) -> Result<Vec<SP1ExtensionField>, GermError> {
    let nvars = proof.nvars as usize;
    if proof.rounds.len() != nvars {
        return Err(GermError::SumcheckRoundsMismatch { got: proof.rounds.len(), expected: nvars });
    }
    if nvars == 0 {
        return Ok(Vec::new());
    }

    let mut claimed = ext_zero();
    let mut sampled = Vec::with_capacity(nvars);
    let mut round_state = *seed;
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        let evals = round.evaluations;
        if evals[0] + evals[1] != claimed {
            return Err(GermError::SumcheckIdentityFailed(round_idx));
        }
        let r_sc = derive_sumcheck_round_challenge(&round_state);
        sampled.push(r_sc);
        claimed = interpolate_0123(&evals, r_sc)?;
        round_state = update_sumcheck_round_state(&round_state, &evals, &claimed);
    }
    Ok(sampled)
}

fn build_weighted_mul_opening_form(
    weights: &[SP1ExtensionField],
    field_exprs: &[[AffineWitnessExpr; 4]],
    field: MultiplicativeTermField,
) -> AadpLinearForm<SP1ExtensionField> {
    let field_idx = match field {
        MultiplicativeTermField::A => 0,
        MultiplicativeTermField::B => 1,
        MultiplicativeTermField::C => 2,
        MultiplicativeTermField::D => 3,
    };
    let mut form = linear_form_constant(ext_zero());
    for (weight, exprs) in weights.iter().zip(field_exprs.iter()) {
        let expr = exprs[field_idx];
        form.constant += *weight * expr.constant;
        if let Some(var_idx) = expr.var_idx {
            form.terms.push((var_idx, *weight * expr.scale));
        }
    }
    form
}

fn encode_lin_proof(proof: &Sp1LinProof) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 16 + (PACKAGE_AJTAI_ROWS * PACKAGE_AJTAI_RING_DIM * 16));
    out.extend_from_slice(&proof.term_count.to_le_bytes());
    write_extension(&mut out, &proof.folded_residual);
    for row in &proof.ajtai_commitment {
        for coeff in row {
            write_extension(&mut out, coeff);
        }
    }
    out
}

fn decode_lin_proof(bytes: &[u8]) -> Result<Sp1LinProof, GermError> {
    let mut cursor = 0usize;
    let term_count = read_u32_lin(bytes, &mut cursor)?;
    let folded_residual = read_extension_lin(bytes, &mut cursor)?;
    let mut ajtai_commitment = [[ext_zero(); PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS];
    for row in ajtai_commitment.iter_mut().take(PACKAGE_AJTAI_ROWS) {
        for coeff in row.iter_mut().take(PACKAGE_AJTAI_RING_DIM) {
            *coeff = read_extension_lin(bytes, &mut cursor)?;
        }
    }
    if cursor != bytes.len() {
        return Err(GermError::MalformedLinProof);
    }
    Ok(Sp1LinProof { term_count, folded_residual, ajtai_commitment })
}

fn encode_mul_sumcheck(proof: &Sp1MulSumcheckProof) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&proof.nvars.to_le_bytes());
    out.extend_from_slice(&(proof.rounds.len() as u32).to_le_bytes());
    for round in &proof.rounds {
        for eval in &round.evaluations {
            write_extension(&mut out, eval);
        }
    }
    write_extension(&mut out, &proof.opening.a);
    write_extension(&mut out, &proof.opening.b);
    write_extension(&mut out, &proof.opening.c);
    write_extension(&mut out, &proof.opening.d);
    out
}

fn decode_mul_sumcheck(bytes: &[u8]) -> Result<Sp1MulSumcheckProof, GermError> {
    let mut cursor = 0usize;
    let nvars = read_u16(bytes, &mut cursor)?;
    let rounds_len = read_u32(bytes, &mut cursor)? as usize;
    let mut rounds = Vec::with_capacity(rounds_len);
    for _ in 0..rounds_len {
        let evals = [
            read_extension(bytes, &mut cursor)?,
            read_extension(bytes, &mut cursor)?,
            read_extension(bytes, &mut cursor)?,
            read_extension(bytes, &mut cursor)?,
        ];
        rounds.push(Sp1MulSumcheckRound { evaluations: evals });
    }
    let opening = Sp1MulTerm::new(
        read_extension(bytes, &mut cursor)?,
        read_extension(bytes, &mut cursor)?,
        read_extension(bytes, &mut cursor)?,
        read_extension(bytes, &mut cursor)?,
    );
    if cursor != bytes.len() {
        return Err(GermError::MalformedMulProof);
    }
    Ok(Sp1MulSumcheckProof { nvars, rounds, opening })
}

fn digest_linear_terms(terms: &[Sp1LinTerm]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/lin-terms/v1");
    h.update((terms.len() as u64).to_le_bytes());
    for term in terms {
        hash_extension(&mut h, &term.coefficient);
        hash_extension(&mut h, &term.value);
    }
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn digest_multiplicative_terms(terms: &[Sp1MulTerm]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/mul-terms/v1");
    h.update((terms.len() as u64).to_le_bytes());
    for term in terms {
        hash_extension(&mut h, &term.a);
        hash_extension(&mut h, &term.b);
        hash_extension(&mut h, &term.c);
        hash_extension(&mut h, &term.d);
    }
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn digest_mul_sumcheck(proof: &Sp1MulSumcheckProof) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/mul-sumcheck/v1");
    h.update(proof.nvars.to_le_bytes());
    h.update((proof.rounds.len() as u64).to_le_bytes());
    for round in &proof.rounds {
        for eval in &round.evaluations {
            hash_extension(&mut h, eval);
        }
    }
    hash_extension(&mut h, &proof.opening.a);
    hash_extension(&mut h, &proof.opening.b);
    hash_extension(&mut h, &proof.opening.c);
    hash_extension(&mut h, &proof.opening.d);
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn digest_multiplicative_relation(terms: &[Sp1MulTerm], proof: &Sp1MulSumcheckProof) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/mul-relation/v1");
    h.update(digest_multiplicative_terms(terms));
    h.update(digest_mul_sumcheck(proof));
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn compute_relation_fingerprint(
    domain: &[u8],
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
    challenge: &SP1ExtensionField,
    relation_digest: &[u8; 32],
    folded_residual: &SP1ExtensionField,
    proof: &[u8],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(public_values_digest);
    h.update(commitment_root);
    hash_extension(&mut h, challenge);
    h.update(relation_digest);
    hash_extension(&mut h, folded_residual);
    h.update((proof.len() as u64).to_le_bytes());
    h.update(proof);
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn hash_extension(h: &mut Sha256, value: &SP1ExtensionField) {
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value) {
        h.update(limb.as_canonical_u32().to_le_bytes());
    }
}

fn algebraic_absorb(
    state: SP1ExtensionField,
    value: SP1ExtensionField,
    round_idx: u32,
) -> SP1ExtensionField {
    let round = ext_from_u32(round_idx.wrapping_mul(17).wrapping_add(1));
    let mixed = state + value + round;
    (mixed * mixed) * (mixed + ext_one())
}

fn derive_algebraic_challenge(
    domain: &[u8],
    seed: SP1ExtensionField,
    extras: &[SP1ExtensionField],
    index: u32,
) -> SP1ExtensionField {
    let mut state = bytes_to_extension(domain);
    state = algebraic_absorb(state, seed, index.wrapping_add(11));
    state = algebraic_absorb(state, ext_from_u32(index), index.wrapping_add(13));
    for (extra_idx, extra) in extras.iter().enumerate() {
        state = algebraic_absorb(
            state,
            *extra,
            index.wrapping_add((extra_idx as u32).wrapping_mul(3)).wrapping_add(17),
        );
    }
    state
}

fn bytes_to_extension(bytes: &[u8]) -> SP1ExtensionField {
    let mut state = ext_one();
    for (idx, chunk) in bytes.chunks(4).enumerate() {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let absorbed = ext_from_wrapped_u32(u32::from_le_bytes(word));
        state = algebraic_absorb(state, absorbed, (idx as u32).wrapping_add(23));
    }
    state
}

fn derive_orbweaver_jl_rows(seed: &[u8; 32], coords_total: usize) -> Vec<Vec<i8>> {
    let mut out = vec![vec![0i8; coords_total]; ORBWEAVER_JL_TOTAL_PROJECTIONS];
    for (row_idx, row) in out.iter_mut().enumerate() {
        let mut filled = 0usize;
        let mut block = 0u32;
        while filled < row.len() {
            let mut h = Sha256::new();
            h.update(b"sp1-germ/orbweaver-jl-row/v2");
            h.update(seed);
            h.update((row_idx as u32).to_le_bytes());
            h.update(block.to_le_bytes());
            let digest = h.finalize();
            for byte in digest {
                if filled >= row.len() {
                    break;
                }
                row[filled] = match byte % 5 {
                    0 => -1,
                    1 => 1,
                    _ => 0,
                };
                filled += 1;
            }
            block = block.wrapping_add(1);
        }
        if row.iter().all(|coeff| *coeff == 0) && !row.is_empty() {
            row[0] = 1;
        }
    }
    out
}

fn write_extension(out: &mut Vec<u8>, value: &SP1ExtensionField) {
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value) {
        out.extend_from_slice(&limb.as_canonical_u32().to_le_bytes());
    }
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, GermError> {
    let end = cursor.saturating_add(2);
    let chunk = bytes.get(*cursor..end).ok_or(GermError::MalformedMulProof)?;
    *cursor = end;
    Ok(u16::from_le_bytes([chunk[0], chunk[1]]))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, GermError> {
    let end = cursor.saturating_add(4);
    let chunk = bytes.get(*cursor..end).ok_or(GermError::MalformedMulProof)?;
    *cursor = end;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_u32_lin(bytes: &[u8], cursor: &mut usize) -> Result<u32, GermError> {
    let end = cursor.saturating_add(4);
    let chunk = bytes.get(*cursor..end).ok_or(GermError::MalformedLinProof)?;
    *cursor = end;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_extension(bytes: &[u8], cursor: &mut usize) -> Result<SP1ExtensionField, GermError> {
    let limbs = [
        SP1Field::from_wrapped_u32(read_u32(bytes, cursor)?),
        SP1Field::from_wrapped_u32(read_u32(bytes, cursor)?),
        SP1Field::from_wrapped_u32(read_u32(bytes, cursor)?),
        SP1Field::from_wrapped_u32(read_u32(bytes, cursor)?),
    ];
    Ok(SP1ExtensionField::from_base_slice(&limbs))
}

fn read_extension_lin(bytes: &[u8], cursor: &mut usize) -> Result<SP1ExtensionField, GermError> {
    let limbs = [
        SP1Field::from_wrapped_u32(read_u32_lin(bytes, cursor)?),
        SP1Field::from_wrapped_u32(read_u32_lin(bytes, cursor)?),
        SP1Field::from_wrapped_u32(read_u32_lin(bytes, cursor)?),
        SP1Field::from_wrapped_u32(read_u32_lin(bytes, cursor)?),
    ];
    Ok(SP1ExtensionField::from_base_slice(&limbs))
}

fn is_zero_ext(value: &SP1ExtensionField) -> bool {
    <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value)
        .iter()
        .all(|x| x.as_canonical_u32() == 0)
}

fn ext_from_u32(x: u32) -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::from_canonical_u32(x),
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
    ])
}

fn ext_from_base_field(x: SP1Field) -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[x, SP1Field::zero(), SP1Field::zero(), SP1Field::zero()])
}

fn ext_basis_u() -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::zero(),
        SP1Field::one(),
        SP1Field::zero(),
        SP1Field::zero(),
    ])
}

fn ext_basis_u_squared() -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::one(),
        SP1Field::zero(),
    ])
}

fn ext_basis_u_cubed() -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::one(),
    ])
}

fn ext_from_wrapped_u32(x: u32) -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::from_wrapped_u32(x),
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
    ])
}

fn ext_zero() -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
    ])
}

fn ext_one() -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::one(),
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
    ])
}

fn decompose_unsigned_bits(value: u64, bits: usize) -> Result<Vec<u8>, String> {
    if bits == 0 {
        return Err("unsigned decomposition requires at least 1 bit".to_string());
    }
    if bits < 64 && value >= (1u64 << bits) {
        return Err(format!(
            "unsigned decomposition overflow: value={} range=[0,{})",
            value,
            1u64 << bits
        ));
    }
    let mut out = Vec::with_capacity(bits);
    let mut rem = value;
    for _ in 0..bits {
        out.push((rem & 1) as u8);
        rem >>= 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{Sp1LinTerm, Sp1MulTerm};
    use crate::orbweaver_opening::OrbweaverOpeningSrs;
    use rand::{rngs::StdRng, SeedableRng};
    use slop_algebra::{AbstractExtensionField, AbstractField};

    fn ext_from_word(x: u32) -> SP1ExtensionField {
        let limbs: [SP1Field; 4] =
            core::array::from_fn(|i| SP1Field::from_canonical_u32(x + i as u32));
        SP1ExtensionField::from_base_slice(&limbs)
    }

    fn test_public_values() -> GermPublicValues {
        GermPublicValues {
            statement_digest: [7u8; 32],
            descriptor_digest: [11u8; 32],
            verifier_shape_digest: [13u8; 32],
            share_index: 2,
            share_domain_separator: [17u8; 32],
        }
    }

    fn alternate_public_values() -> GermPublicValues {
        GermPublicValues {
            statement_digest: [7u8; 32],
            descriptor_digest: [11u8; 32],
            verifier_shape_digest: [13u8; 32],
            share_index: 3,
            share_domain_separator: [17u8; 32],
        }
    }

    fn test_residual_plan() -> GermResidualPlan {
        GermResidualPlan::new(
            [19u8; 32],
            GermVerifierStage::Compressed,
            1,
            PACKAGE_AJTAI_ROWS as u16,
            PACKAGE_AJTAI_RING_DIM as u16,
            vec![LinearResidualDescriptor::Explicit],
            vec![
                MultiplicativeResidualDescriptor::Explicit,
                MultiplicativeResidualDescriptor::Explicit,
            ],
        )
    }

    fn toy_orbweaver_srs(max_terms: usize) -> OrbweaverOpeningSrs {
        let witness_width = max_terms * 16;
        let a0 = vec![crate::koala_ring::KoalaRing64::one()];
        let v_scalar = SP1Field::from_canonical_u32(7);
        let v = crate::koala_ring::KoalaRing64::from_scalar(v_scalar);
        let v_inv_scalar = v_scalar.try_inverse().expect("invertible");
        let v_inv = crate::koala_ring::KoalaRing64::from_scalar(v_inv_scalar);
        let mut u0_positive = vec![vec![crate::koala_ring::KoalaRing64::zero()]; witness_width + 1];
        let mut u0_negative = vec![vec![crate::koala_ring::KoalaRing64::zero()]; witness_width + 1];
        let mut cur_pos = v;
        let mut cur_neg = v_inv;
        for idx in 1..=witness_width {
            u0_positive[idx] = vec![cur_pos];
            u0_negative[idx] = vec![cur_neg];
            cur_pos *= v;
            cur_neg *= v_inv;
        }
        OrbweaverOpeningSrs { a0, v, u0_positive, u0_negative }
    }

    #[test]
    fn bind_and_verify_roundtrip() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, challenges) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");

        let lin = verify_lin(&bundle, &public_values, &commitment_root).expect("lin verify");
        let mul = verify_mul(&bundle, &public_values, &commitment_root).expect("mul verify");
        assert!(is_zero_ext(&lin.folded_residual));
        assert!(is_zero_ext(&mul.folded_residual));

        let derived = derive_challenges(&public_values, &bundle.shared_object_commitment);
        assert_eq!(derived, challenges);
    }

    #[test]
    fn compile_aadp_template_roundtrip() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let (commitment_root, _) =
            bind_bundle_to_capsule(&mut bundle, &capsule).expect("bind should succeed");
        let template = compile_germ_aadp_template(&capsule, &residual_plan)
            .expect("aadp template compile should succeed");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let witness =
            materialize_transcript_bound_germ_aadp_witness(&template, &capsule, &transcript_bound)
                .expect("materialize witness should succeed");

        assert!(
            template.stats.linear_round_checks >= 3,
            "expected at least lin+sumcheck+final checks"
        );
        assert!(template.stats.multiplication_gates >= 2);
        template.check_witness(&witness).expect("template witness must satisfy constraints");
    }

    #[test]
    fn compile_transcript_bound_aadp_template_roundtrip() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let (commitment_root, _) =
            bind_bundle_to_capsule(&mut bundle, &capsule).expect("bind should succeed");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);

        let generic_template =
            compile_germ_aadp_template(&capsule, &residual_plan).expect("generic template compile");
        let transcript_template = compile_transcript_bound_germ_aadp_template(
            &capsule,
            &residual_plan,
            &transcript_bound,
        )
        .expect("transcript-bound template compile");
        let witness = materialize_transcript_bound_germ_aadp_witness(
            &transcript_template,
            &capsule,
            &transcript_bound,
        )
        .expect("materialize witness should succeed");

        assert_eq!(
            transcript_template.stats.opening_checks,
            generic_template.stats.opening_checks + 4
        );
        assert_eq!(
            transcript_template.cs.constraints.len(),
            generic_template.cs.constraints.len() + 4
        );
        transcript_template
            .check_witness(&witness)
            .expect("witness must satisfy transcript-bound template");
    }

    #[test]
    fn transcript_bound_template_rejects_rebound_proof_object() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);

        let mut bundle_a = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let (commitment_root_a, _) =
            bind_bundle_to_capsule(&mut bundle_a, &capsule).expect("bind bundle_a");
        let transcript_a = TranscriptBoundSp1GermProofObject::new(bundle_a, commitment_root_a);
        let transcript_template =
            compile_transcript_bound_germ_aadp_template(&capsule, &residual_plan, &transcript_a)
                .expect("compile transcript-bound template");

        let a2 = ext_from_word(17);
        let b2 = ext_from_word(19);
        let c2 = ext_from_word(23);
        let mut bundle_b = Sp1GermBundle::new(
            b"shared-object-v2".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![
                Sp1MulTerm::new(a2, b2, ext_one(), a2 * b2),
                Sp1MulTerm::new(b2, c2, ext_one(), b2 * c2),
            ],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let (commitment_root_b, _) =
            bind_bundle_to_capsule(&mut bundle_b, &capsule).expect("bind bundle_b");
        let transcript_b = TranscriptBoundSp1GermProofObject::new(bundle_b, commitment_root_b);

        let err = materialize_transcript_bound_germ_aadp_witness(
            &transcript_template,
            &capsule,
            &transcript_b,
        )
        .unwrap_err();
        assert!(matches!(err, GermError::AadpConstraintUnsatisfied(_)));
    }

    #[test]
    fn materialization_rejects_tampered_mul_round() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let (commitment_root, _) =
            bind_bundle_to_capsule(&mut bundle, &capsule).expect("bind should succeed");
        let template =
            compile_germ_aadp_template(&capsule, &residual_plan).expect("template compile");

        let mut mul_proof = decode_mul_sumcheck(&bundle.pi_mul).expect("decode mul proof");
        let first_round = mul_proof.rounds.first_mut().expect("sumcheck should have one round");
        first_round.evaluations[0] += ext_one();
        bundle.pi_mul = encode_mul_sumcheck(&mul_proof);

        let err = materialize_transcript_bound_germ_aadp_witness(
            &template,
            &capsule,
            &TranscriptBoundSp1GermProofObject::new(bundle, commitment_root),
        )
        .unwrap_err();
        assert_eq!(err, GermError::SumcheckTranscriptMismatch);
    }

    #[test]
    fn materialization_rejects_tampered_mul_opening() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let (commitment_root, _) =
            bind_bundle_to_capsule(&mut bundle, &capsule).expect("bind should succeed");
        let template =
            compile_germ_aadp_template(&capsule, &residual_plan).expect("template compile");

        let mut mul_proof = decode_mul_sumcheck(&bundle.pi_mul).expect("decode mul proof");
        mul_proof.opening.a += ext_one();
        bundle.pi_mul = encode_mul_sumcheck(&mul_proof);

        let err = materialize_transcript_bound_germ_aadp_witness(
            &template,
            &capsule,
            &TranscriptBoundSp1GermProofObject::new(bundle, commitment_root),
        )
        .unwrap_err();
        assert_eq!(err, GermError::SumcheckTranscriptMismatch);
    }

    #[test]
    fn template_rejects_public_context_mismatch() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let (commitment_root, _) =
            bind_bundle_to_capsule(&mut bundle, &capsule).expect("bind should succeed");
        let template =
            compile_germ_aadp_template(&capsule, &residual_plan).expect("template compile");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let mismatched_capsule = alternate_public_values().arm_capsule(&residual_plan);
        let err = materialize_transcript_bound_germ_aadp_witness(
            &template,
            &mismatched_capsule,
            &transcript_bound,
        )
        .unwrap_err();
        assert_eq!(err, GermError::TemplateCapsuleMismatch);
    }

    #[test]
    fn armed_relation_rejects_tampered_witness() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let msg = ext_from_word(123);
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let (commitment_root, _) =
            bind_bundle_to_capsule(&mut bundle, &capsule).expect("bind should succeed");
        let mut rng = StdRng::seed_from_u64(42);
        let armed = arm_germ_aadp_template(&capsule, &residual_plan, msg, &mut rng)
            .expect("arming template should succeed");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let witness = materialize_transcript_bound_germ_aadp_witness(
            &armed.template,
            &capsule,
            &transcript_bound,
        )
        .expect("materialize witness should succeed");

        armed.template.check_witness(&witness).expect("witness must satisfy template constraints");
        let got =
            armed.decap_checked(&witness).expect("aadp decrypt with valid witness should succeed");
        assert_eq!(got, msg);

        let mut tampered_witness = witness.witness.clone();
        tampered_witness[0] += ext_one();
        let err = armed.decap_checked(&GermAadpWitness { witness: tampered_witness }).unwrap_err();
        assert!(matches!(err, GermError::AadpWitnessRejected(_)));
    }

    #[test]
    fn orbweaver_terminal_opening_roundtrip() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let srs = crate::orbweaver_opening::generate_local_dev_srs(bundle.mul_terms.len() * 16);
        let (commitment_root, _) =
            bind_bundle_with_orbweaver_terminal_openings(&mut bundle, &public_values, &srs)
                .expect("bind with orbweaver terminal openings");
        assert!(!bundle.pi_mul_terminal_openings.a.is_empty());
        assert!(!bundle.pi_mul_terminal_openings.b.is_empty());
        assert!(!bundle.pi_mul_terminal_openings.c.is_empty());
        assert!(!bundle.pi_mul_terminal_openings.d.is_empty());
        let mul_check = verify_mul_with_orbweaver_terminal_openings(
            &bundle,
            &public_values,
            &commitment_root,
            &srs,
        )
        .expect("verify mul with orbweaver terminal openings");
        assert!(is_zero_ext(&mul_check.folded_residual));
    }

    #[test]
    fn orbweaver_terminal_opening_detects_tamper() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let srs = crate::orbweaver_opening::generate_local_dev_srs(bundle.mul_terms.len() * 16);
        let (commitment_root, _) =
            bind_bundle_with_orbweaver_terminal_openings(&mut bundle, &public_values, &srs)
                .expect("bind with orbweaver terminal openings");
        bundle.pi_mul_terminal_openings.a[0] ^= 0x01;
        let err = verify_mul_with_orbweaver_terminal_openings(
            &bundle,
            &public_values,
            &commitment_root,
            &srs,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                GermError::MalformedMulTerminalOpening { .. }
                    | GermError::MulTerminalOpeningMismatch(_)
                    | GermError::MulTerminalOpeningFailed { .. }
            ),
            "unexpected orbweaver terminal opening error: {err:?}"
        );
    }

    #[test]
    fn orbweaver_template_materialization_roundtrip() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let c = ext_from_word(13);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b), Sp1MulTerm::new(b, c, ext_one(), b * c)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let srs = crate::orbweaver_opening::generate_local_dev_srs(bundle.mul_terms.len() * 16);
        let (commitment_root, _) = bind_bundle_to_capsule_with_orbweaver_terminal_openings(
            &mut bundle,
            &capsule,
            &srs,
        )
        .expect("bind with orbweaver terminal openings");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let template = compile_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings(
            &capsule,
            &residual_plan,
            &transcript_bound,
            &srs,
        )
        .expect("compile transcript-bound orbweaver template");
        let witness = materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings(
            &template,
            &capsule,
            &transcript_bound,
            &srs,
        )
        .expect("materialize orbweaver witness");
        template
            .check_witness(&witness)
            .expect("orbweaver witness must satisfy template");
    }

    #[test]
    fn verify_rejects_public_context_mismatch() {
        let a = ext_from_word(7);
        let b = ext_from_word(11);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, _) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let mismatched_public_values = alternate_public_values();

        let lin_err = verify_lin(&bundle, &mismatched_public_values, &commitment_root).unwrap_err();
        assert_eq!(lin_err, GermError::LinProofTranscriptMismatch);

        let mul_err = verify_mul(&bundle, &mismatched_public_values, &commitment_root).unwrap_err();
        assert_eq!(mul_err, GermError::BindingTagMismatch("mul"));
    }

    #[test]
    fn verify_detects_tag_tamper() {
        let a = ext_from_word(3);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, a, ext_one(), a * a)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, _challenges) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let mut lin_proof = decode_lin_proof(&bundle.pi_lin).expect("decode linear proof");
        lin_proof.folded_residual += ext_one();
        bundle.pi_lin = encode_lin_proof(&lin_proof);

        let err = verify_lin(&bundle, &public_values, &commitment_root).unwrap_err();
        assert_eq!(err, GermError::LinProofTranscriptMismatch);
    }

    #[test]
    fn verify_rejects_tampered_ajtai_commitment() {
        let a = ext_from_word(3);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, a, ext_one(), a * a)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, _challenges) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let mut lin_proof = decode_lin_proof(&bundle.pi_lin).expect("decode linear proof");
        lin_proof.ajtai_commitment[0][0] += ext_one();
        bundle.pi_lin = encode_lin_proof(&lin_proof);

        let err = verify_lin(&bundle, &public_values, &commitment_root).unwrap_err();
        assert_eq!(err, GermError::LinProofTranscriptMismatch);
    }

    #[test]
    fn verify_rejects_malformed_linear_proof_bytes() {
        let a = ext_from_word(3);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, a, ext_one(), a * a)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, _) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let _ = bundle.pi_lin.pop();

        let err = verify_lin(&bundle, &public_values, &commitment_root).unwrap_err();
        assert_eq!(err, GermError::MalformedLinProof);
    }

    #[test]
    fn verify_rejects_nonzero_residual() {
        let a = ext_from_word(5);
        let b = ext_from_word(8);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, challenges) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        bundle.mul_terms[0].d = bundle.mul_terms[0].d + ext_one();
        let sumcheck_seed = derive_mul_sumcheck_seed(
            &public_values.digest(),
            &bundle.shared_object_commitment,
            &challenges.r_mul,
        );
        let sumcheck = prove_mul_sumcheck(&sumcheck_seed, &bundle.mul_terms);
        bundle.pi_mul = encode_mul_sumcheck(&sumcheck);

        let err = verify_mul(&bundle, &public_values, &commitment_root).unwrap_err();
        assert_eq!(err, GermError::NonZeroResidual("mul"));
    }

    #[test]
    fn verify_rejects_tampered_sumcheck() {
        let a = ext_from_word(9);
        let b = ext_from_word(12);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_zero())],
            vec![Sp1MulTerm::new(a, b, ext_one(), a * b)],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();
        let (commitment_root, _challenges) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let last = bundle.pi_mul.last_mut().expect("pi_mul should be non-empty");
        *last ^= 0x01;

        let err = verify_mul(&bundle, &public_values, &commitment_root).unwrap_err();
        assert_eq!(err, GermError::SumcheckTranscriptMismatch);
    }

    #[test]
    fn bind_rejects_invalid_terms() {
        let a = ext_from_word(2);
        let b = ext_from_word(4);
        let mut bundle = Sp1GermBundle::new(
            b"shared-object-v1".to_vec(),
            vec![Sp1LinTerm::new(ext_one(), ext_one())],
            vec![Sp1MulTerm::new(a, b, ext_one(), ext_zero())],
            b"lin-proof".to_vec(),
            b"mul-proof".to_vec(),
        );
        let public_values = test_public_values();

        let err = bind_bundle(&mut bundle, &public_values).unwrap_err();
        assert_eq!(err, GermError::NonZeroResidual("lin"));
    }
}

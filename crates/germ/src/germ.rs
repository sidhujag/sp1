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
    Sp1MulSumcheckRound, Sp1MulTerm, Sp1PackageCommitment, TranscriptBoundSp1GermProofObject,
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

#[derive(Debug, Clone)]
struct CommitmentBindingLayout {
    commitment_slots: Vec<CommitmentWitnessSlot>,
    commitment_forms: Vec<AadpLinearForm<SP1ExtensionField>>,
    expected_linear_terms: usize,
    expected_mul_terms: usize,
    linear_coefficient_indices: Vec<Option<usize>>,
    linear_value_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct GermAadpWitnessLayout {
    pub sumcheck_rounds: usize,
    expected_linear_terms: usize,
    expected_mul_terms: usize,
    commitment_slots: Vec<CommitmentWitnessSlot>,
    linear_term_has_explicit_coefficient: Vec<bool>,
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

fn linear_descriptor_term_count(descriptor: &LinearResidualDescriptor) -> usize {
    match descriptor {
        LinearResidualDescriptor::PublicValuesPadding { count, .. } => *count,
        _ => 1,
    }
}

fn expected_mul_term_count(plan: &GermResidualPlan) -> Result<usize, GermError> {
    let target = 1usize
        .checked_shl(u32::from(plan.sumcheck_rounds))
        .ok_or_else(|| {
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
                    commitment_slots.push(CommitmentWitnessSlot::LinearValue {
                        term_idx: linear_term_idx,
                    });
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
                for field in [
                    MultiplicativeTermField::A,
                    MultiplicativeTermField::B,
                    MultiplicativeTermField::C,
                    MultiplicativeTermField::D,
                ] {
                    let field_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                        term_idx: mul_term_idx,
                        field,
                    });
                    add_commitment_message_affine_entry(
                        forms.as_mut_slice(),
                        &seed,
                        column_idx,
                        Some(field_idx),
                        ext_one(),
                        ext_zero(),
                    );
                    column_idx += 1;
                }
            }
            MultiplicativeResidualDescriptor::DegreeBitBooleanity { .. } => {
                let (chip_name, bit_number) = match descriptor {
                    MultiplicativeResidualDescriptor::DegreeBitBooleanity { chip_name, bit_idx } => {
                        (chip_name.clone(), *bit_idx)
                    }
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
            }
            MultiplicativeResidualDescriptor::DegreeHeightProduct { .. } => {
                let (chip_name, bit_number) = match descriptor {
                    MultiplicativeResidualDescriptor::DegreeHeightProduct { chip_name, bit_idx } => {
                        (chip_name.clone(), *bit_idx)
                    }
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
                let field_b_idx =
                    if let Some(existing_idx) = degree_bit_slots.get(&(chip_name.clone(), 0usize)) {
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
            }
            MultiplicativeResidualDescriptor::GkrDenominatorInverse { .. } => {
                for field in [MultiplicativeTermField::A, MultiplicativeTermField::B] {
                    let field_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                        term_idx: mul_term_idx,
                        field,
                    });
                    add_commitment_message_affine_entry(
                        forms.as_mut_slice(),
                        &seed,
                        column_idx,
                        Some(field_idx),
                        ext_one(),
                        ext_zero(),
                    );
                    column_idx += 1;
                }
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
            }
            MultiplicativeResidualDescriptor::GkrRoundFinalEval { .. } => {
                for field in [
                    MultiplicativeTermField::A,
                    MultiplicativeTermField::B,
                    MultiplicativeTermField::C,
                ] {
                    let field_idx = alloc();
                    commitment_slots.push(CommitmentWitnessSlot::MultiplicativeField {
                        term_idx: mul_term_idx,
                        field,
                    });
                    add_commitment_message_affine_entry(
                        forms.as_mut_slice(),
                        &seed,
                        column_idx,
                        Some(field_idx),
                        ext_one(),
                        ext_zero(),
                    );
                    column_idx += 1;
                }
                add_commitment_message_affine_entry(
                    forms.as_mut_slice(),
                    &seed,
                    column_idx,
                    None,
                    ext_zero(),
                    ext_one(),
                );
                column_idx += 1;
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
    }

    Ok(CommitmentBindingLayout {
        commitment_slots,
        commitment_forms: forms,
        expected_linear_terms: linear_term_idx,
        expected_mul_terms,
        linear_coefficient_indices,
        linear_value_indices,
    })
}

fn slot_value_from_proof_object(
    proof_object: &Sp1GermBundle,
    slot: CommitmentWitnessSlot,
) -> Result<SP1ExtensionField, GermError> {
    match slot {
        CommitmentWitnessSlot::LinearCoefficient { term_idx } => {
            proof_object
                .lin_terms
                .get(term_idx)
                .map(|term| term.coefficient)
                .ok_or(GermError::TranscriptShapeMismatch {
                    which: "linear commitment slot",
                    got: proof_object.lin_terms.len(),
                    expected: term_idx + 1,
                })
        }
        CommitmentWitnessSlot::LinearValue { term_idx } => proof_object
            .lin_terms
            .get(term_idx)
            .map(|term| term.value)
            .ok_or(GermError::TranscriptShapeMismatch {
                which: "linear commitment slot",
                got: proof_object.lin_terms.len(),
                expected: term_idx + 1,
            }),
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
    let mut commitment_mix = ext_zero();
    let mut coord_idx = 0usize;
    for row in commitment {
        for coord in row {
            commitment_mix += commitment_mix_weight(coord_idx) * *coord;
            coord_idx += 1;
        }
    }
    let after_digest = algebraic_absorb(
        bytes_to_extension(b"sp1-germ/challenge-seed/v2"),
        bytes_to_extension(arming_digest),
        1,
    );
    algebraic_absorb(after_digest, commitment_mix, 2)
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
    let check = evaluate_multiplicative_relation(
        bundle,
        &public_values_digest,
        commitment_root,
        &challenges.r_mul,
        b"sp1-germ/mul_bind/v1",
    )?;
    if !is_zero_ext(&check.folded_residual) {
        return Err(GermError::NonZeroResidual("mul"));
    }
    if check.fingerprint != bundle.mul_binding_tag {
        return Err(GermError::BindingTagMismatch("mul"));
    }
    Ok(check)
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

pub fn compile_germ_aadp_template(
    capsule: &GermArmCapsule,
    residual_plan: &GermResidualPlan,
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
        add_linear_zero_constraint(
            &mut constraints,
            form,
        );
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

        let mut lin_table_forms =
            Vec::<AadpLinearForm<SP1ExtensionField>>::with_capacity(binding_layout.expected_linear_terms.max(1).next_power_of_two());
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
                next_forms.push(linear_form_add_forms(&pair[0], &linear_form_single_var(delta_idx)));
            }
            lin_table_forms = next_forms;
        }
        let lin_claim_idx = alloc();
        add_linear_zero_constraint(
            &mut constraints,
            linear_form_sub_forms(&linear_form_single_var(lin_claim_idx), &lin_table_forms[0]),
        );
        linear_round_checks += 1;
        add_linear_zero_constraint(
            &mut constraints,
            linear_form_single_var(lin_claim_idx),
        );
        linear_round_checks += 1;
    }

    // Packed sumcheck verifier with schedule-fixed number of rounds.
    let mut claimed_prev_idx: Option<usize> = None;
    let mut round_challenge_indices = Vec::with_capacity(usize::from(capsule.sumcheck_rounds));
    let round_domain = bytes_to_extension(b"sp1-germ/sumcheck-round-challenge/v2");
    for round_idx in 0..usize::from(capsule.sumcheck_rounds) {
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

        // r_sc = Challenge(sumcheck_seed, idx=round_idx).
        let (_r_sc_seed_sq_idx, r_sc_seed_state_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            None,
            round_domain,
            Some(sumcheck_seed_idx),
            ext_zero(),
            (round_idx as u32).wrapping_add(11),
        );
        let (_r_sc_idx_sq_idx, r_sc_idx) = add_absorb_constraints(
            &mut constraints,
            &mut alloc,
            Some(r_sc_seed_state_idx),
            ext_zero(),
            None,
            ext_from_u32(round_idx as u32),
            (round_idx as u32).wrapping_add(13),
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
    if template.capsule_digest != capsule.digest() {
        return Err(GermError::TemplateCapsuleMismatch);
    }
    verify_transcript_bound_sp1_germ_proof_object(transcript_bound, capsule)?;

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
        let lin_right_idx_absorb = lin_right_seed_state + ext_from_u32(3) + absorb_round_constant(16);
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
            let lin_point_seed_state = lin_point_seed_absorb_sq * (lin_point_seed_absorb + ext_one());
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

        let mut lin_table = Vec::with_capacity(template.layout.expected_linear_terms.max(1).next_power_of_two());
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

    let mut point = Vec::with_capacity(mul_proof.nvars as usize);
    let mut sampled = Vec::with_capacity(mul_proof.rounds.len());
    for (round_idx, round) in mul_proof.rounds.iter().enumerate() {
        for eval in round.evaluations {
            push(&mut witness, eval);
        }

        // r_sc derivation trace.
        let r_sc_seed_absorb = bytes_to_extension(b"sp1-germ/sumcheck-round-challenge/v2")
            + sumcheck_seed
            + absorb_round_constant((round_idx as u32).wrapping_add(11));
        let r_sc_seed_absorb_sq = r_sc_seed_absorb * r_sc_seed_absorb;
        let r_sc_seed_state = r_sc_seed_absorb_sq * (r_sc_seed_absorb + ext_one());
        push(&mut witness, r_sc_seed_absorb_sq);
        push(&mut witness, r_sc_seed_state);
        let r_sc_idx_absorb = r_sc_seed_state
            + ext_from_u32(round_idx as u32)
            + absorb_round_constant((round_idx as u32).wrapping_add(13));
        let r_sc_idx_absorb_sq = r_sc_idx_absorb * r_sc_idx_absorb;
        let r_sc = r_sc_idx_absorb_sq * (r_sc_idx_absorb + ext_one());
        push(&mut witness, r_sc_idx_absorb_sq);
        push(&mut witness, r_sc);

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
    }

    push(&mut witness, mul_proof.opening.a);
    push(&mut witness, mul_proof.opening.b);
    push(&mut witness, mul_proof.opening.c);
    push(&mut witness, mul_proof.opening.d);
    let opening_lhs_product = mul_proof.opening.a * mul_proof.opening.b;
    let opening_rhs_product = mul_proof.opening.c * mul_proof.opening.d;
    push(&mut witness, opening_lhs_product);
    push(&mut witness, opening_rhs_product);
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
    let fingerprint = compute_relation_fingerprint(
        domain,
        public_values_digest,
        commitment_root,
        r_mul,
        &relation_digest,
        &folded_residual,
        &bundle.pi_mul,
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

fn derive_sumcheck_round_challenge(
    seed: &SP1ExtensionField,
    round_idx: usize,
) -> SP1ExtensionField {
    derive_algebraic_challenge(
        b"sp1-germ/sumcheck-round-challenge/v2",
        *seed,
        &[],
        round_idx as u32,
    )
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
    for round_idx in 0..nvars {
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
        let r_sc = derive_sumcheck_round_challenge(seed, round_idx);
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
    for (round_idx, round) in proof.rounds.iter().enumerate() {
        let evals = round.evaluations;
        if evals[0] + evals[1] != claimed {
            return Err(GermError::SumcheckIdentityFailed(round_idx));
        }
        let r_sc = derive_sumcheck_round_challenge(seed, round_idx);
        sampled.push(r_sc);
        claimed = interpolate_0123(&evals, r_sc)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::{Sp1LinTerm, Sp1MulTerm};
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
        let (commitment_root, _) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
        let template = compile_germ_aadp_template(&capsule, &residual_plan)
            .expect("aadp template compile should succeed");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let witness = materialize_transcript_bound_germ_aadp_witness(
            &template,
            &capsule,
            &transcript_bound,
        )
        .expect("materialize witness should succeed");

        assert!(
            template.stats.linear_round_checks >= 3,
            "expected at least lin+sumcheck+final checks"
        );
        assert!(template.stats.multiplication_gates >= 2);
        template.check_witness(&witness).expect("template witness must satisfy constraints");
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
        let (commitment_root, _) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
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
        let (commitment_root, _) =
            bind_bundle(&mut bundle, &public_values).expect("bind should succeed");
        let msg = ext_from_word(123);
        let residual_plan = test_residual_plan();
        let capsule = public_values.arm_capsule(&residual_plan);
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

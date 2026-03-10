//! SP1-native GERM relation checks.
//!
//! This module follows the "two global vanishing checks" shape:
//! - `V_lin(r_lin) = 0` for globally mixed linear residuals
//! - `V_mul(r_mul) = 0` for globally mixed multiplicative residuals
//!
//! `r_lin` and `r_mul` are Fiat-Shamir designated points derived from
//! public arming context + commitment root.

use rand::RngCore;
use sha2::{Digest, Sha256};
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use sp1_primitives::{SP1ExtensionField, SP1Field};

use crate::ajtai::{
    derive_lin_ajtai_seed, derive_package_ajtai_seed, lin_ajtai_commitment,
    lin_ajtai_matrix_entry_coeff, package_ajtai_commitment, package_message_from_bytes,
    package_opening_projection_residuals, LIN_AJTAI_RING_DIM, LIN_AJTAI_ROWS,
    PACKAGE_OPENING_PROJECTIONS,
};
use crate::aadp::{
    aadp_encrypt_scalar, AadpCiphertext, AadpConstraintSystem, AadpLinearForm, AadpMulConstraint,
};
use crate::bundle::{
    GermArmCapsule, GermResidualPlan, GermVerifierStage, Sp1GermBundle, Sp1PackageCommitment,
    Sp1LinProof, Sp1LinTerm, Sp1MulSumcheckProof, Sp1MulSumcheckRound, Sp1MulTerm,
    TranscriptBoundSp1GermProofObject,
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
pub struct GermAadpWitnessLayout {
    pub sumcheck_rounds: usize,
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
        self.cs
            .check_witness(witness.as_slice())
            .map_err(GermError::AadpWitnessRejected)
    }

    pub fn decap_checked(
        &self,
        ciphertext: &AadpCiphertext<SP1ExtensionField>,
        witness: &GermAadpWitness,
    ) -> Result<SP1ExtensionField, GermError> {
        self.check_witness(witness)?;
        ciphertext
            .decrypt_scalar(witness.as_slice())
            .map_err(GermError::AadpDecryptFailed)
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
            Self::SumcheckFinalResidualNonZero => write!(f, "final sumcheck opening residual non-zero"),
            Self::SumcheckInterpolationDenominatorZero => {
                write!(f, "sumcheck interpolation denominator was zero")
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

fn compute_shared_object_commitment(shared_object: &[u8]) -> Sp1PackageCommitment {
    let seed = derive_package_ajtai_seed();
    let msg = package_message_from_bytes(shared_object);
    package_ajtai_commitment(&seed, msg.as_slice())
}

#[must_use]
pub fn derive_challenges(
    public_values: &GermPublicValues,
    commitment_root: &[u8; 32],
) -> GermChallenges {
    derive_challenges_from_arming_digest(&public_values.digest(), commitment_root)
}

#[must_use]
pub fn derive_challenges_from_capsule(
    capsule: &GermArmCapsule,
    commitment_root: &[u8; 32],
) -> GermChallenges {
    derive_challenges_from_arming_digest(&capsule.digest(), commitment_root)
}

#[must_use]
pub fn derive_challenges_from_arming_digest(
    arming_digest: &[u8; 32],
    commitment_root: &[u8; 32],
) -> GermChallenges {
    GermChallenges {
        r_lin: nonzero_challenge(challenge_from_commitment(
            b"sp1-germ/r_lin/v1",
            arming_digest,
            commitment_root,
        )),
        r_mul: nonzero_challenge(challenge_from_commitment(
            b"sp1-germ/r_mul/v1",
            arming_digest,
            commitment_root,
        )),
    }
}

pub fn bind_bundle_to_capsule(
    bundle: &mut Sp1GermBundle,
    capsule: &GermArmCapsule,
) -> Result<([u8; 32], GermChallenges), GermError> {
    bundle.shared_object_commitment = compute_shared_object_commitment(&bundle.shared_object);
    let commitment_root = compute_commitment_root(&bundle.shared_object_commitment);
    let arming_digest = capsule.digest();
    let challenges = derive_challenges_from_capsule(capsule, &commitment_root);
    let lin_term_count = u32::try_from(bundle.lin_terms.len())
        .map_err(|_| GermError::TooManyLinearTerms(bundle.lin_terms.len()))?;
    let lin_folded_residual = fold_linear_terms(
        &bundle.lin_terms,
        &arming_digest,
        &commitment_root,
        &challenges.r_lin,
    );
    let lin_ajtai_seed =
        derive_lin_ajtai_seed(&arming_digest, &commitment_root, &challenges.r_lin);
    let lin_proof = Sp1LinProof {
        term_count: lin_term_count,
        folded_residual: lin_folded_residual,
        ajtai_commitment: lin_ajtai_commitment(
            &lin_ajtai_seed,
            lin_term_count,
            &lin_folded_residual,
        ),
        package_opening_projection: package_opening_projection_residuals(
            &challenges.r_lin,
            &compute_shared_object_commitment(&bundle.shared_object),
            &bundle.shared_object_commitment,
        ),
    };
    bundle.pi_lin = encode_lin_proof(&lin_proof);
    let sumcheck_seed =
        derive_mul_sumcheck_seed(&arming_digest, &commitment_root, &challenges.r_mul);
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
    bundle.shared_object_commitment = compute_shared_object_commitment(&bundle.shared_object);
    let commitment_root = compute_commitment_root(&bundle.shared_object_commitment);
    let public_values_digest = public_values.digest();
    let challenges = derive_challenges(public_values, &commitment_root);
    let lin_term_count = u32::try_from(bundle.lin_terms.len())
        .map_err(|_| GermError::TooManyLinearTerms(bundle.lin_terms.len()))?;
    let lin_folded_residual = fold_linear_terms(
        &bundle.lin_terms,
        &public_values_digest,
        &commitment_root,
        &challenges.r_lin,
    );
    let lin_ajtai_seed =
        derive_lin_ajtai_seed(&public_values_digest, &commitment_root, &challenges.r_lin);
    let lin_proof = Sp1LinProof {
        term_count: lin_term_count,
        folded_residual: lin_folded_residual,
        ajtai_commitment: lin_ajtai_commitment(
            &lin_ajtai_seed,
            lin_term_count,
            &lin_folded_residual,
        ),
        package_opening_projection: package_opening_projection_residuals(
            &challenges.r_lin,
            &compute_shared_object_commitment(&bundle.shared_object),
            &bundle.shared_object_commitment,
        ),
    };
    bundle.pi_lin = encode_lin_proof(&lin_proof);
    let sumcheck_seed =
        derive_mul_sumcheck_seed(&public_values_digest, &commitment_root, &challenges.r_mul);
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
    let challenges = derive_challenges(public_values, commitment_root);
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
    let challenges = derive_challenges(public_values, commitment_root);
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
    let challenges = derive_challenges_from_capsule(capsule, &transcript_bound.commitment_root);
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
) -> Result<GermAadpVerifierTemplate, GermError> {
    if !matches!(capsule.verifier_stage, GermVerifierStage::Compressed) {
        return Err(GermError::TemplateCapsuleMismatch);
    }

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
        AadpLinearForm {
            constant: -ext_one(),
            terms: vec![(safety_bit_idx, ext_one())],
        },
    );
    linear_round_checks += 1;
    multiplication_gates += 1;

    // Linear folded claim and term count are witness-side under the pre-proof template.
    let lin_claim_idx = alloc();
    let lin_term_count_idx = alloc();
    add_linear_zero_constraint(
        &mut constraints,
        AadpLinearForm {
            constant: ext_zero(),
            terms: vec![(lin_claim_idx, ext_one())],
        },
    );
    linear_round_checks += 1;

    // Ajtai opening checks are verified against witness-side transcript coefficients.
    for _row in 0..LIN_AJTAI_ROWS {
        for _coeff_idx in 0..LIN_AJTAI_RING_DIM {
            let commitment_row_idx = alloc();
            let a0_idx = alloc();
            let a1_idx = alloc();
            let prod0_idx = alloc();
            let prod1_idx = alloc();

            constraints.push(AadpMulConstraint {
                a: linear_form_single_var(a0_idx),
                b: linear_form_single_var(lin_claim_idx),
                c: linear_form_constant(ext_one()),
                d: linear_form_single_var(prod0_idx),
            });
            constraints.push(AadpMulConstraint {
                a: linear_form_single_var(a1_idx),
                b: linear_form_single_var(lin_term_count_idx),
                c: linear_form_constant(ext_one()),
                d: linear_form_single_var(prod1_idx),
            });
            multiplication_gates += 2;

            add_linear_zero_constraint(
                &mut constraints,
                AadpLinearForm {
                    constant: ext_zero(),
                    terms: vec![
                        (commitment_row_idx, ext_one()),
                        (prod0_idx, -ext_one()),
                        (prod1_idx, -ext_one()),
                    ],
                },
            );
            opening_checks += 1;
        }
    }

    // Batched opening projection checks authenticate the common package commitment `C`.
    for _ in 0..PACKAGE_OPENING_PROJECTIONS {
        let projection_idx = alloc();
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: vec![(projection_idx, ext_one())],
            },
        );
        opening_checks += 1;
    }

    // Packed sumcheck verifier with schedule-fixed number of rounds.
    let mut claimed_prev_idx: Option<usize> = None;
    for _round_idx in 0..usize::from(capsule.sumcheck_rounds) {
        let eval_indices = [alloc(), alloc(), alloc(), alloc()];

        let mut identity_terms = vec![(eval_indices[0], ext_one()), (eval_indices[1], ext_one())];
        if let Some(prev_idx) = claimed_prev_idx {
            identity_terms.push((prev_idx, -ext_one()));
        }
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: identity_terms,
            },
        );
        linear_round_checks += 1;

        let coeff_indices = [alloc(), alloc(), alloc(), alloc()];
        let prod_indices = [alloc(), alloc(), alloc(), alloc()];
        let claimed_next_idx = alloc();
        for (coeff_idx, (eval_idx, prod_idx)) in coeff_indices
            .iter()
            .zip(eval_indices.iter().zip(prod_indices.iter()))
        {
            constraints.push(AadpMulConstraint {
                a: linear_form_single_var(*coeff_idx),
                b: linear_form_single_var(*eval_idx),
                c: linear_form_constant(ext_one()),
                d: linear_form_single_var(*prod_idx),
            });
            multiplication_gates += 1;
        }
        let mut interp_terms = vec![(claimed_next_idx, ext_one())];
        for prod_idx in prod_indices {
            interp_terms.push((prod_idx, -ext_one()));
        }
        add_linear_zero_constraint(
            &mut constraints,
            AadpLinearForm {
                constant: ext_zero(),
                terms: interp_terms,
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
        let eq_eval_idx = alloc();
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
                terms: vec![
                    (claimed_last_idx, ext_one()),
                    (final_scaled_idx, -ext_one()),
                ],
            },
        );
        linear_round_checks += 1;
    }

    Ok(GermAadpVerifierTemplate {
        cs: AadpConstraintSystem { num_variables, constraints },
        layout: GermAadpWitnessLayout {
            sumcheck_rounds: usize::from(capsule.sumcheck_rounds),
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
    message: SP1ExtensionField,
    rng: &mut R,
) -> Result<ArmedGermAadpCiphertext, GermError> {
    let template = compile_germ_aadp_template(capsule)?;
    let ciphertext =
        aadp_encrypt_scalar(&template.cs, message, rng).map_err(GermError::AadpEncryptFailed)?;
    Ok(ArmedGermAadpCiphertext {
        template,
        ciphertext,
        capsule_digest: capsule.digest(),
    })
}

/// Materialize the AADP witness from a transcript-bound proof object.
///
/// This function is *outside* the WE security boundary: it assumes a host-side transcript layer
/// already bound the proof object to `commitment_root` and derived the designated challenges from
/// `(x_arm, root(C))`. The resulting witness is then checked by the tiny algebraic AADP relation.
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
    let commitment_root = &transcript_bound.commitment_root;
    let lin_proof = decode_lin_proof(&proof_object.pi_lin)?;
    let mul_proof = decode_mul_sumcheck(&proof_object.pi_mul)?;
    if usize::from(capsule.sumcheck_rounds) != mul_proof.rounds.len() {
        return Err(GermError::SumcheckRoundsMismatch {
            got: mul_proof.rounds.len(),
            expected: usize::from(capsule.sumcheck_rounds),
        });
    }

    let challenges = derive_challenges_from_capsule(capsule, commitment_root);
    let arming_digest = capsule.digest();
    let lin_ajtai_seed =
        derive_lin_ajtai_seed(&arming_digest, commitment_root, &challenges.r_lin);
    let sumcheck_seed =
        derive_mul_sumcheck_seed(&arming_digest, commitment_root, &challenges.r_mul);

    let mut witness = Vec::<SP1ExtensionField>::with_capacity(template.cs.num_variables);
    let push = |witness: &mut Vec<SP1ExtensionField>, value: SP1ExtensionField| {
        witness.push(value);
    };

    push(&mut witness, ext_one()); // safety bit
    push(&mut witness, lin_proof.folded_residual);
    push(&mut witness, ext_from_u32(lin_proof.term_count));

    for row in 0..LIN_AJTAI_ROWS {
        for coeff_idx in 0..LIN_AJTAI_RING_DIM {
            let commitment = lin_proof.ajtai_commitment[row][coeff_idx];
            let a0 = lin_ajtai_matrix_entry_coeff(&lin_ajtai_seed, row, 0, coeff_idx);
            let a1 = lin_ajtai_matrix_entry_coeff(&lin_ajtai_seed, row, 1, coeff_idx);
            push(&mut witness, commitment);
            push(&mut witness, a0);
            push(&mut witness, a1);
            push(&mut witness, a0 * lin_proof.folded_residual);
            push(&mut witness, a1 * ext_from_u32(lin_proof.term_count));
        }
    }
    for residual in lin_proof.package_opening_projection {
        push(&mut witness, residual);
    }

    let point = derive_mul_point(&sumcheck_seed, mul_proof.nvars as usize);
    let mut sampled = Vec::with_capacity(mul_proof.rounds.len());
    for (round_idx, round) in mul_proof.rounds.iter().enumerate() {
        for eval in round.evaluations {
            push(&mut witness, eval);
        }
        let r_sc =
            derive_sumcheck_round_challenge(&sumcheck_seed, &mul_proof.rounds[..=round_idx], round_idx);
        sampled.push(r_sc);
        let coeffs = lagrange_coefficients_0123(r_sc)?;
        for coeff in coeffs {
            push(&mut witness, coeff);
        }
        for (coeff, eval) in coeffs.iter().zip(round.evaluations.iter()) {
            push(&mut witness, *coeff * *eval);
        }
        let claimed_next = interpolate_0123(&round.evaluations, r_sc)?;
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
        let mut eq_eval = ext_one();
        for (r_i, s_i) in point.iter().zip(sampled.iter()) {
            eq_eval *= (ext_one() - *r_i) * (ext_one() - *s_i) + (*r_i * *s_i);
        }
        push(&mut witness, eq_eval);
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
    let expected_commitment = compute_shared_object_commitment(&bundle.shared_object);
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
    let folded_residual = fold_linear_terms(
        &bundle.lin_terms,
        public_values_digest,
        commitment_root,
        r_lin,
    );
    if parsed_lin_proof.folded_residual != folded_residual {
        return Err(GermError::LinProofTranscriptMismatch);
    }
    let lin_ajtai_seed = derive_lin_ajtai_seed(public_values_digest, commitment_root, r_lin);
    let expected_ajtai_commitment =
        lin_ajtai_commitment(
            &lin_ajtai_seed,
            expected_term_count,
            &folded_residual,
        );
    if parsed_lin_proof.ajtai_commitment != expected_ajtai_commitment {
        return Err(GermError::LinProofTranscriptMismatch);
    }
    let expected_package_projection = package_opening_projection_residuals(
        r_lin,
        &expected_commitment,
        &bundle.shared_object_commitment,
    );
    if parsed_lin_proof.package_opening_projection != expected_package_projection {
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
    let expected_commitment = compute_shared_object_commitment(&bundle.shared_object);
    if expected_commitment != bundle.shared_object_commitment {
        return Err(GermError::CommitmentRootMismatch);
    }
    let expected_commitment_root = compute_commitment_root(&expected_commitment);
    if &expected_commitment_root != commitment_root {
        return Err(GermError::CommitmentRootMismatch);
    }
    let parsed_sumcheck = decode_mul_sumcheck(&bundle.pi_mul)?;
    let sumcheck_seed = derive_mul_sumcheck_seed(public_values_digest, commitment_root, r_mul);
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
    commitment_root: &[u8; 32],
    r_lin: &SP1ExtensionField,
) -> SP1ExtensionField {
    let nvars = mul_sumcheck_nvars(terms.len());
    let lin_point_seed = derive_lin_point_seed(public_values_digest, commitment_root, r_lin);
    let lin_point = derive_lin_point(&lin_point_seed, nvars);
    let padded = terms.len().max(1).next_power_of_two();
    let mut table = vec![ext_zero(); padded];
    for (idx, term) in terms.iter().enumerate() {
        table[idx] = term.coefficient * term.value;
    }
    evaluate_mle_table(table, lin_point.as_slice())
}

/// Evaluate globally mixed multiplicative residual at the designated point.
fn fold_multiplicative_terms(terms: &[Sp1MulTerm], sumcheck_seed: &[u8; 32]) -> SP1ExtensionField {
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
    commitment_root: &[u8; 32],
    r_mul: &SP1ExtensionField,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/mul-sumcheck-seed/v1");
    h.update(public_values_digest);
    h.update(commitment_root);
    hash_extension(&mut h, r_mul);
    h.finalize().into()
}

fn derive_lin_point_seed(
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
    r_lin: &SP1ExtensionField,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/lin-point-seed/v1");
    h.update(public_values_digest);
    h.update(commitment_root);
    hash_extension(&mut h, r_lin);
    h.finalize().into()
}

fn derive_lin_point(seed: &[u8; 32], nvars: usize) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(nvars);
    for var_idx in 0..nvars {
        let mut h = Sha256::new();
        h.update(b"sp1-germ/lin-point/v1");
        h.update(seed);
        h.update((var_idx as u32).to_le_bytes());
        let digest: [u8; 32] = h.finalize().into();
        out.push(extension_from_digest(&digest));
    }
    out
}

fn derive_mul_point(seed: &[u8; 32], nvars: usize) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(nvars);
    for var_idx in 0..nvars {
        let mut h = Sha256::new();
        h.update(b"sp1-germ/mul-point/v1");
        h.update(seed);
        h.update((var_idx as u32).to_le_bytes());
        let digest: [u8; 32] = h.finalize().into();
        out.push(extension_from_digest(&digest));
    }
    out
}

fn derive_sumcheck_round_challenge(
    seed: &[u8; 32],
    rounds: &[Sp1MulSumcheckRound],
    round_idx: usize,
) -> SP1ExtensionField {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/sumcheck-round-challenge/v1");
    h.update(seed);
    h.update((round_idx as u32).to_le_bytes());
    for round in rounds {
        for eval in &round.evaluations {
            hash_extension(&mut h, eval);
        }
    }
    let digest: [u8; 32] = h.finalize().into();
    extension_from_digest(&digest)
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
        let den_inv = den
            .try_inverse()
            .ok_or(GermError::SumcheckInterpolationDenominatorZero)?;
        acc += evals[i] * (num * den_inv);
    }
    Ok(acc)
}

fn lagrange_coefficients_0123(
    x: SP1ExtensionField,
) -> Result<[SP1ExtensionField; 4], GermError> {
    let xs = [ext_from_u32(0), ext_from_u32(1), ext_from_u32(2), ext_from_u32(3)];
    let mut out = [ext_zero(), ext_zero(), ext_zero(), ext_zero()];
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
        let den_inv = den
            .try_inverse()
            .ok_or(GermError::SumcheckInterpolationDenominatorZero)?;
        out[i] = num * den_inv;
    }
    Ok(out)
}

fn linear_form_constant(value: SP1ExtensionField) -> AadpLinearForm<SP1ExtensionField> {
    AadpLinearForm {
        constant: value,
        terms: Vec::new(),
    }
}

fn linear_form_single_var(var_idx: usize) -> AadpLinearForm<SP1ExtensionField> {
    AadpLinearForm {
        constant: ext_zero(),
        terms: vec![(var_idx, ext_one())],
    }
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

fn add_bit_constraint(
    constraints: &mut Vec<AadpMulConstraint<SP1ExtensionField>>,
    var_idx: usize,
) {
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

fn evaluate_mle_table(mut table: Vec<SP1ExtensionField>, point: &[SP1ExtensionField]) -> SP1ExtensionField {
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

fn prove_mul_sumcheck(seed: &[u8; 32], terms: &[Sp1MulTerm]) -> Sp1MulSumcheckProof {
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
        let r_sc = derive_sumcheck_round_challenge(seed, rounds.as_slice(), round_idx);
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

fn verify_mul_sumcheck(seed: &[u8; 32], proof: &Sp1MulSumcheckProof) -> Result<(), GermError> {
    let nvars = proof.nvars as usize;
    if proof.rounds.len() != nvars {
        return Err(GermError::SumcheckRoundsMismatch {
            got: proof.rounds.len(),
            expected: nvars,
        });
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
        let r_sc = derive_sumcheck_round_challenge(seed, &proof.rounds[..=round_idx], round_idx);
        sampled.push(r_sc);
        claimed = interpolate_0123(&evals, r_sc)?;
    }

    let mut eq_eval = ext_one();
    for (r_i, s_i) in point.iter().zip(sampled.iter()) {
        eq_eval *= (ext_one() - *r_i) * (ext_one() - *s_i) + (*r_i * *s_i);
    }
    let final_residual =
        claimed - (eq_eval * ((proof.opening.a * proof.opening.b) - (proof.opening.c * proof.opening.d)));
    if !is_zero_ext(&final_residual) {
        return Err(GermError::SumcheckFinalResidualNonZero);
    }
    Ok(())
}

fn encode_lin_proof(proof: &Sp1LinProof) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        4 + 16 + (LIN_AJTAI_ROWS * LIN_AJTAI_RING_DIM * 16) + (PACKAGE_OPENING_PROJECTIONS * 16),
    );
    out.extend_from_slice(&proof.term_count.to_le_bytes());
    write_extension(&mut out, &proof.folded_residual);
    for row in &proof.ajtai_commitment {
        for coeff in row {
            write_extension(&mut out, coeff);
        }
    }
    for residual in &proof.package_opening_projection {
        write_extension(&mut out, residual);
    }
    out
}

fn decode_lin_proof(bytes: &[u8]) -> Result<Sp1LinProof, GermError> {
    let mut cursor = 0usize;
    let term_count = read_u32_lin(bytes, &mut cursor)?;
    let folded_residual = read_extension_lin(bytes, &mut cursor)?;
    let mut ajtai_commitment = [[ext_zero(); LIN_AJTAI_RING_DIM]; LIN_AJTAI_ROWS];
    for row in ajtai_commitment.iter_mut().take(LIN_AJTAI_ROWS) {
        for coeff in row.iter_mut().take(LIN_AJTAI_RING_DIM) {
            *coeff = read_extension_lin(bytes, &mut cursor)?;
        }
    }
    let mut package_opening_projection = [ext_zero(); PACKAGE_OPENING_PROJECTIONS];
    for residual in package_opening_projection.iter_mut().take(PACKAGE_OPENING_PROJECTIONS) {
        *residual = read_extension_lin(bytes, &mut cursor)?;
    }
    if cursor != bytes.len() {
        return Err(GermError::MalformedLinProof);
    }
    Ok(Sp1LinProof {
        term_count,
        folded_residual,
        ajtai_commitment,
        package_opening_projection,
    })
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
    Ok(Sp1MulSumcheckProof {
        nvars,
        rounds,
        opening,
    })
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

fn digest_multiplicative_relation(
    terms: &[Sp1MulTerm],
    proof: &Sp1MulSumcheckProof,
) -> [u8; 32] {
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

fn challenge_from_commitment(
    domain: &[u8],
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
) -> SP1ExtensionField {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(public_values_digest);
    h.update(commitment_root);
    let digest: [u8; 32] = h.finalize().into();
    extension_from_digest(&digest)
}

fn nonzero_challenge(challenge: SP1ExtensionField) -> SP1ExtensionField {
    if is_zero_ext(&challenge) {
        ext_one()
    } else {
        challenge
    }
}

fn hash_extension(h: &mut Sha256, value: &SP1ExtensionField) {
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value) {
        h.update(limb.as_canonical_u32().to_le_bytes());
    }
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

fn extension_from_digest(digest: &[u8; 32]) -> SP1ExtensionField {
    let limbs: [SP1Field; 4] = core::array::from_fn(|i| {
        let start = i * 4;
        let bytes = [
            digest[start],
            digest[start + 1],
            digest[start + 2],
            digest[start + 3],
        ];
        SP1Field::from_wrapped_u32(u32::from_le_bytes(bytes))
    });
    SP1ExtensionField::from_base_slice(&limbs)
}

fn ext_from_u32(x: u32) -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[
        SP1Field::from_canonical_u32(x),
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
        GermResidualPlan {
            schedule_descriptor_digest: [19u8; 32],
            residual_plan_digest: [23u8; 32],
            verifier_stage: GermVerifierStage::Compressed,
            sumcheck_rounds: 1,
            linear_opening_rows: LIN_AJTAI_ROWS as u16,
            linear_opening_ring_dim: LIN_AJTAI_RING_DIM as u16,
        }
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

        let derived = derive_challenges(&public_values, &commitment_root);
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
        let capsule = public_values.arm_capsule(&test_residual_plan());
        let template = compile_germ_aadp_template(&capsule).expect("aadp template compile should succeed");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let witness = materialize_transcript_bound_germ_aadp_witness(
            &template,
            &capsule,
            &transcript_bound,
            &public_values,
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
        let capsule = public_values.arm_capsule(&test_residual_plan());
        let template = compile_germ_aadp_template(&capsule).expect("template compile");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let err = materialize_transcript_bound_germ_aadp_witness(
            &template,
            &capsule,
            &transcript_bound,
            &alternate_public_values(),
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
        let capsule = public_values.arm_capsule(&test_residual_plan());
        let mut rng = StdRng::seed_from_u64(42);
        let armed =
            arm_germ_aadp_template(&capsule, msg, &mut rng).expect("arming template should succeed");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        let witness = materialize_transcript_bound_germ_aadp_witness(
            &armed.template,
            &capsule,
            &transcript_bound,
            &public_values,
        )
        .expect("materialize witness should succeed");

        armed.template.check_witness(&witness).expect("witness must satisfy template constraints");
        let got = armed
            .decap_checked(&witness)
            .expect("aadp decrypt with valid witness should succeed");
        assert_eq!(got, msg);

        let mut tampered_witness = witness.witness.clone();
        tampered_witness[0] += ext_one();
        let err = armed
            .decap_checked(&GermAadpWitness { witness: tampered_witness })
            .unwrap_err();
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

        let lin_err =
            verify_lin(&bundle, &mismatched_public_values, &commitment_root).unwrap_err();
        assert_eq!(lin_err, GermError::LinProofTranscriptMismatch);

        let mul_err =
            verify_mul(&bundle, &mismatched_public_values, &commitment_root).unwrap_err();
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
            &commitment_root,
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

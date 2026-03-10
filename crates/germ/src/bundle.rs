use slop_algebra::{AbstractExtensionField, AbstractField};
use sp1_primitives::{SP1ExtensionField, SP1Field};

pub type Sp1PackageCommitment = [[SP1ExtensionField; 4]; 4];

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GermVerifierStage {
    Compressed,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GermResidualPlan {
    pub schedule_descriptor_digest: [u8; 32],
    pub residual_plan_digest: [u8; 32],
    pub verifier_stage: GermVerifierStage,
    pub sumcheck_rounds: u16,
    pub linear_opening_rows: u16,
    pub linear_opening_ring_dim: u16,
}

impl GermResidualPlan {
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};

        let mut h = Sha256::new();
        h.update(b"sp1-germ/residual-plan/v1");
        h.update(self.schedule_descriptor_digest);
        h.update(self.residual_plan_digest);
        h.update(match self.verifier_stage {
            GermVerifierStage::Compressed => b"compressed".as_slice(),
        });
        h.update(self.sumcheck_rounds.to_le_bytes());
        h.update(self.linear_opening_rows.to_le_bytes());
        h.update(self.linear_opening_ring_dim.to_le_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GermArmCapsule {
    pub public_values_digest: [u8; 32],
    pub schedule_descriptor_digest: [u8; 32],
    pub residual_plan_digest: [u8; 32],
    pub verifier_stage: GermVerifierStage,
    pub sumcheck_rounds: u16,
}

impl GermArmCapsule {
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};

        let mut h = Sha256::new();
        h.update(b"sp1-germ/arm-capsule/v1");
        h.update(self.public_values_digest);
        h.update(self.schedule_descriptor_digest);
        h.update(self.residual_plan_digest);
        h.update(match self.verifier_stage {
            GermVerifierStage::Compressed => b"compressed".as_slice(),
        });
        h.update(self.sumcheck_rounds.to_le_bytes());
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sp1LinTerm {
    pub coefficient: SP1ExtensionField,
    pub value: SP1ExtensionField,
}

impl Sp1LinTerm {
    #[must_use]
    pub fn new(coefficient: SP1ExtensionField, value: SP1ExtensionField) -> Self {
        Self { coefficient, value }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sp1LinProof {
    pub term_count: u32,
    pub folded_residual: SP1ExtensionField,
    /// Ajtai-style ring commitment rows for the opened linear proof message.
    pub ajtai_commitment: [[SP1ExtensionField; 4]; 4],
    /// Batched projection residuals authenticating the shared-object commitment `C`.
    pub package_opening_projection: [SP1ExtensionField; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sp1MulTerm {
    pub a: SP1ExtensionField,
    pub b: SP1ExtensionField,
    pub c: SP1ExtensionField,
    pub d: SP1ExtensionField,
}

impl Sp1MulTerm {
    #[must_use]
    pub fn new(a: SP1ExtensionField, b: SP1ExtensionField, c: SP1ExtensionField, d: SP1ExtensionField) -> Self {
        Self { a, b, c, d }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sp1MulSumcheckRound {
    pub evaluations: [SP1ExtensionField; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sp1MulSumcheckProof {
    pub nvars: u16,
    pub rounds: Vec<Sp1MulSumcheckRound>,
    pub opening: Sp1MulTerm,
}

impl Default for Sp1MulSumcheckProof {
    fn default() -> Self {
        Self {
            nvars: 0,
            rounds: Vec::new(),
            opening: Sp1MulTerm::new(
                SP1ExtensionField::from_base_slice(&[
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                ]),
                SP1ExtensionField::from_base_slice(&[
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                ]),
                SP1ExtensionField::from_base_slice(&[
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                ]),
                SP1ExtensionField::from_base_slice(&[
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                    SP1Field::zero(),
                ]),
            ),
        }
    }
}

/// SP1-native in-memory carrier used by the GERM+AADP flow.
///
/// The caller is responsible for extracting these fields from a concrete proof object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sp1GermBundle {
    /// Canonical bytes for the shared object `O` to be committed into `C`.
    pub shared_object: Vec<u8>,
    /// Deterministic Ajtai-style commitment object for `shared_object`.
    pub shared_object_commitment: Sp1PackageCommitment,
    /// Terms for linear relation checks.
    pub lin_terms: Vec<Sp1LinTerm>,
    /// Terms for multiplicative relation checks.
    pub mul_terms: Vec<Sp1MulTerm>,
    /// Proof bytes for the linear opening path.
    ///
    /// In the current SP1-native GERM layer this is the canonical encoding of
    /// `Sp1LinProof` (claim value + Ajtai opening payload).
    pub pi_lin: Vec<u8>,
    /// Proof bytes for the multiplicative opening path.
    ///
    /// In the current SP1-native GERM layer this is the canonical encoding of
    /// `Sp1MulSumcheckProof`.
    pub pi_mul: Vec<u8>,
    /// Binding tag checked by `verify_lin`.
    pub lin_binding_tag: [u8; 32],
    /// Binding tag checked by `verify_mul`.
    pub mul_binding_tag: [u8; 32],
}

/// Realized post-proof object for the SP1-only GERM bridge.
///
/// This is intentionally separate from `GermArmCapsule`: the capsule is fixed before proving,
/// while this object is materialized only after the SP1 proof has been produced and exported.
pub type Sp1GermProofObject = Sp1GermBundle;

/// Host-side transcript-bound proof artifact.
///
/// This wrapper makes the current security split explicit:
/// - `proof_object` is the committed/exported object consumed by the tiny verifier relation
/// - `commitment_root` is derived from that object by host-side transcript logic
///
/// This wrapper is intentionally *outside* the AADP/WE boundary. It represents the
/// transcript-bound preprocessing step that later feeds witness materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptBoundSp1GermProofObject {
    pub proof_object: Sp1GermProofObject,
    pub commitment_root: [u8; 32],
}

impl TranscriptBoundSp1GermProofObject {
    #[must_use]
    pub fn new(proof_object: Sp1GermProofObject, commitment_root: [u8; 32]) -> Self {
        Self { proof_object, commitment_root }
    }
}

impl Sp1GermBundle {
    #[must_use]
    pub fn new(
        shared_object: Vec<u8>,
        lin_terms: Vec<Sp1LinTerm>,
        mul_terms: Vec<Sp1MulTerm>,
        pi_lin: Vec<u8>,
        pi_mul: Vec<u8>,
    ) -> Self {
        Self {
            shared_object,
            shared_object_commitment: [[SP1ExtensionField::from_base_slice(&[
                SP1Field::zero(),
                SP1Field::zero(),
                SP1Field::zero(),
                SP1Field::zero(),
            ]); 4]; 4],
            lin_terms,
            mul_terms,
            pi_lin,
            pi_mul,
            lin_binding_tag: [0u8; 32],
            mul_binding_tag: [0u8; 32],
        }
    }

    /// Convenience constructor for migration from raw opening vectors.
    ///
    /// - each linear opening is interpreted as a term with coefficient 1
    /// - multiplicative openings are interpreted as quadruples `(a, b, c, d)`
    pub fn from_openings(
        shared_object: Vec<u8>,
        lin_openings: Vec<SP1ExtensionField>,
        mul_openings: Vec<SP1ExtensionField>,
        pi_lin: Vec<u8>,
        pi_mul: Vec<u8>,
    ) -> Result<Self, String> {
        let one = SP1ExtensionField::from_base_slice(&[
            SP1Field::one(),
            SP1Field::zero(),
            SP1Field::zero(),
            SP1Field::zero(),
        ]);
        let lin_terms = lin_openings.into_iter().map(|value| Sp1LinTerm::new(one, value)).collect();

        if mul_openings.len() % 4 != 0 {
            return Err(format!(
                "mul_openings length must be divisible by 4, got {}",
                mul_openings.len()
            ));
        }
        let mut mul_terms = Vec::with_capacity(mul_openings.len() / 4);
        for chunk in mul_openings.chunks_exact(4) {
            mul_terms.push(Sp1MulTerm::new(chunk[0], chunk[1], chunk[2], chunk[3]));
        }
        Ok(Self::new(shared_object, lin_terms, mul_terms, pi_lin, pi_mul))
    }
}

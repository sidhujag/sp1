//! Ajtai-style seeded linear commitment helpers for `pi_lin`.
//!
//! This module keeps the seeded-matrix binding shape aligned with the Ajtai flow:
//! - derive a matrix seed from `(x, C, r_lin)`,
//! - commit to `(folded_residual, term_count)`.
//!
//! To match Ajtai semantics, the commitment rows are represented as ring elements
//! over `SP1ExtensionField` with fixed coefficient dimension.

use sha2::{Digest, Sha256};
use slop_algebra::{AbstractExtensionField, AbstractField, PrimeField32};
use sp1_primitives::{SP1ExtensionField, SP1Field};

pub(crate) const LIN_AJTAI_ROWS: usize = 4;
pub(crate) const LIN_AJTAI_RING_DIM: usize = 4;
const LIN_AJTAI_WIDTH: usize = 2;
pub(crate) const PACKAGE_AJTAI_ROWS: usize = 4;
pub(crate) const PACKAGE_AJTAI_RING_DIM: usize = 4;
pub(crate) const PACKAGE_OPENING_PROJECTIONS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AjtaiRingElem {
    coeffs: [SP1ExtensionField; LIN_AJTAI_RING_DIM],
}

impl AjtaiRingElem {
    fn from_seed(seed: &[u8; 32], row: usize, col: usize) -> Self {
        let coeffs = core::array::from_fn(|coeff_idx| {
            let mut h = Sha256::new();
            h.update(b"sp1-germ/lin-ajtai-ring-entry/v1");
            h.update(seed);
            h.update((row as u32).to_le_bytes());
            h.update((col as u32).to_le_bytes());
            h.update((coeff_idx as u32).to_le_bytes());
            let digest: [u8; 32] = h.finalize().into();
            extension_from_digest(&digest)
        });
        Self { coeffs }
    }
}

#[derive(Clone, Debug)]
enum AjtaiMatrix {
    Explicit([[AjtaiRingElem; LIN_AJTAI_WIDTH]; LIN_AJTAI_ROWS]),
    Seeded {
        seed: [u8; 32],
    },
}

#[derive(Clone, Debug)]
struct AjtaiCommitmentScheme {
    matrix: AjtaiMatrix,
}

impl AjtaiCommitmentScheme {
    /// Create a new scheme using the provided Ajtai matrix.
    pub fn new(matrix: [[AjtaiRingElem; LIN_AJTAI_WIDTH]; LIN_AJTAI_ROWS]) -> Self {
        Self {
            matrix: AjtaiMatrix::Explicit(matrix),
        }
    }

    /// Create a scheme with an implicitly-defined pseudorandom Ajtai matrix.
    pub fn seeded(seed: [u8; 32]) -> Self {
        Self {
            matrix: AjtaiMatrix::Seeded { seed },
        }
    }

    fn matrix_entry(&self, row: usize, col: usize) -> AjtaiRingElem {
        match &self.matrix {
            AjtaiMatrix::Explicit(matrix) => matrix[row][col],
            AjtaiMatrix::Seeded { seed } => AjtaiRingElem::from_seed(seed, row, col),
        }
    }

    fn commit_const_coeff_base_fast(
        &self,
        f0: &[SP1ExtensionField; LIN_AJTAI_WIDTH],
    ) -> [[SP1ExtensionField; LIN_AJTAI_RING_DIM]; LIN_AJTAI_ROWS] {
        let mut acc = [[ext_zero(); LIN_AJTAI_RING_DIM]; LIN_AJTAI_ROWS];
        for (col, fj0) in f0.iter().enumerate() {
            if is_zero_ext(fj0) {
                continue;
            }
            for (row, row_acc) in acc.iter_mut().enumerate().take(LIN_AJTAI_ROWS) {
                let aij = self.matrix_entry(row, col);
                for (coeff_idx, coeff_acc) in row_acc.iter_mut().enumerate().take(LIN_AJTAI_RING_DIM) {
                    *coeff_acc += aij.coeffs[coeff_idx] * *fj0;
                }
            }
        }
        acc
    }
}

pub(crate) fn derive_lin_ajtai_seed(
    public_values_digest: &[u8; 32],
    commitment_root: &[u8; 32],
    r_lin: &SP1ExtensionField,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/lin-ajtai-seed/v1");
    h.update(public_values_digest);
    h.update(commitment_root);
    hash_extension(&mut h, r_lin);
    h.finalize().into()
}

pub(crate) fn lin_ajtai_matrix_entry_coeff(
    seed: &[u8; 32],
    row: usize,
    col: usize,
    coeff_idx: usize,
) -> SP1ExtensionField {
    AjtaiCommitmentScheme::seeded(*seed).matrix_entry(row, col).coeffs[coeff_idx]
}

pub(crate) fn lin_ajtai_commitment(
    seed: &[u8; 32],
    term_count: u32,
    folded_residual: &SP1ExtensionField,
) -> [[SP1ExtensionField; LIN_AJTAI_RING_DIM]; LIN_AJTAI_ROWS] {
    let msg = [*folded_residual, ext_from_u32(term_count)];
    let seeded_scheme = AjtaiCommitmentScheme::seeded(*seed);
    let explicit = core::array::from_fn(|row| {
        core::array::from_fn(|col| seeded_scheme.matrix_entry(row, col))
    });
    AjtaiCommitmentScheme::new(explicit).commit_const_coeff_base_fast(&msg)
}

pub(crate) fn derive_package_ajtai_seed() -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/package-ajtai-seed/v1");
    h.finalize().into()
}

pub(crate) fn package_message_from_bytes(bytes: &[u8]) -> Vec<SP1ExtensionField> {
    bytes.chunks(16)
        .map(|chunk| {
            let limbs: [SP1Field; 4] = core::array::from_fn(|i| {
                let start = i * 4;
                let mut word = [0u8; 4];
                if start < chunk.len() {
                    let end = core::cmp::min(start + 4, chunk.len());
                    word[..(end - start)].copy_from_slice(&chunk[start..end]);
                }
                SP1Field::from_wrapped_u32(u32::from_le_bytes(word))
            });
            SP1ExtensionField::from_base_slice(&limbs)
        })
        .collect()
}

pub(crate) fn package_ajtai_commitment(
    seed: &[u8; 32],
    message: &[SP1ExtensionField],
) -> [[SP1ExtensionField; PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS] {
    let mut acc = [[ext_zero(); PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS];
    let scheme = AjtaiCommitmentScheme::seeded(*seed);
    for (col, value) in message.iter().enumerate() {
        if is_zero_ext(value) {
            continue;
        }
        for (row, row_acc) in acc.iter_mut().enumerate().take(PACKAGE_AJTAI_ROWS) {
            let aij = scheme.matrix_entry(row, col);
            for (coeff_idx, coeff_acc) in row_acc.iter_mut().enumerate().take(PACKAGE_AJTAI_RING_DIM) {
                *coeff_acc += aij.coeffs[coeff_idx] * *value;
            }
        }
    }
    acc
}

pub(crate) fn package_opening_projection_residuals(
    designated_challenge: &SP1ExtensionField,
    expected: &[[SP1ExtensionField; PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS],
    claimed: &[[SP1ExtensionField; PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS],
) -> [SP1ExtensionField; PACKAGE_OPENING_PROJECTIONS] {
    let expected_coords = flatten_commitment_rows(expected);
    let claimed_coords = flatten_commitment_rows(claimed);
    let mut out = [ext_zero(); PACKAGE_OPENING_PROJECTIONS];
    for projection_idx in 0..PACKAGE_OPENING_PROJECTIONS {
        let mut acc = ext_zero();
        for (coord_idx, (exp, got)) in expected_coords.iter().zip(claimed_coords.iter()).enumerate() {
            let weight = projection_weight(designated_challenge, projection_idx, coord_idx);
            acc += weight * (*exp - *got);
        }
        out[projection_idx] = acc;
    }
    out
}

fn flatten_commitment_rows(
    rows: &[[SP1ExtensionField; PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS],
) -> Vec<SP1ExtensionField> {
    let mut out = Vec::with_capacity(PACKAGE_AJTAI_ROWS * PACKAGE_AJTAI_RING_DIM);
    for row in rows {
        out.extend(row.iter().copied());
    }
    out
}

fn projection_weight(
    designated_challenge: &SP1ExtensionField,
    projection_idx: usize,
    coord_idx: usize,
) -> SP1ExtensionField {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/package-opening-projection/v1");
    hash_extension(&mut h, designated_challenge);
    h.update((projection_idx as u32).to_le_bytes());
    h.update((coord_idx as u32).to_le_bytes());
    let digest: [u8; 32] = h.finalize().into();
    extension_from_digest(&digest)
}

fn hash_extension(h: &mut Sha256, value: &SP1ExtensionField) {
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value) {
        h.update(limb.as_canonical_u32().to_le_bytes());
    }
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

fn is_zero_ext(value: &SP1ExtensionField) -> bool {
    <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value)
        .iter()
        .all(|x| x.as_canonical_u32() == 0)
}

//! Ajtai-style seeded commitment helpers for shared-object commitment `C`.
//!
//! This module only provides the package commitment path used to build `C` from
//! canonical `shared_object` bytes. Linear `pi_lin` Ajtai message commitment
//! helpers were removed once the relation switched to deterministic in-boundary
//! coordinate equality against `C`.

use sha2::{Digest, Sha256};
use slop_algebra::{AbstractExtensionField, AbstractField, PrimeField32};
use sp1_primitives::{SP1ExtensionField, SP1Field};

pub(crate) const PACKAGE_AJTAI_ROWS: usize = 4;
pub(crate) const PACKAGE_AJTAI_RING_DIM: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AjtaiRingElem {
    coeffs: [SP1ExtensionField; PACKAGE_AJTAI_RING_DIM],
}

impl AjtaiRingElem {
    fn from_seed(seed: &[u8; 32], row: usize, col: usize) -> Self {
        let coeffs = core::array::from_fn(|coeff_idx| {
            let mut h = Sha256::new();
            // Keep the historical domain separator to avoid changing commitment values.
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

pub(crate) fn derive_package_ajtai_seed() -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/package-ajtai-seed/v1");
    h.finalize().into()
}

pub(crate) fn package_ajtai_matrix_entry(
    seed: &[u8; 32],
    row: usize,
    col: usize,
) -> [SP1ExtensionField; PACKAGE_AJTAI_RING_DIM] {
    AjtaiRingElem::from_seed(seed, row, col).coeffs
}

pub(crate) fn package_ajtai_commitment(
    seed: &[u8; 32],
    message: &[SP1ExtensionField],
) -> [[SP1ExtensionField; PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS] {
    let mut acc = [[ext_zero(); PACKAGE_AJTAI_RING_DIM]; PACKAGE_AJTAI_ROWS];
    for (col, value) in message.iter().enumerate() {
        if is_zero_ext(value) {
            continue;
        }
        for (row, row_acc) in acc.iter_mut().enumerate().take(PACKAGE_AJTAI_ROWS) {
            let aij = package_ajtai_matrix_entry(seed, row, col);
            for (coeff_idx, coeff_acc) in
                row_acc.iter_mut().enumerate().take(PACKAGE_AJTAI_RING_DIM)
            {
                *coeff_acc += aij[coeff_idx] * *value;
            }
        }
    }
    acc
}

fn extension_from_digest(digest: &[u8; 32]) -> SP1ExtensionField {
    let limbs: [SP1Field; 4] = core::array::from_fn(|i| {
        let start = i * 4;
        let bytes = [digest[start], digest[start + 1], digest[start + 2], digest[start + 3]];
        SP1Field::from_wrapped_u32(u32::from_le_bytes(bytes))
    });
    SP1ExtensionField::from_base_slice(&limbs)
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

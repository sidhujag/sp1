//! Linearization helpers for `SP1ExtensionField` over the KoalaBear base field.
//!
//! The Orbweaver opening layer we are building is currently native over the base
//! field, while the SP1 GERM witness and terminal maps live in the quartic
//! extension `F[u]/(u^4 - 3)`. This module exposes the exact base-field matrix
//! representation of multiplication by an extension-field scalar so we can
//! derive faithful base-field linear maps for terminal openings.

use slop_algebra::{AbstractExtensionField, AbstractField};
use sp1_primitives::{SP1ExtensionField, SP1Field};

/// The SP1 extension is `F[u] / (u^4 - 3)`.
#[inline]
fn sp1_ext_w() -> SP1Field {
    SP1Field::from_canonical_u32(3)
}

/// Return the `4 x 4` base-field matrix `M_a` such that
/// `coeffs(a * x) = M_a * coeffs(x)`.
#[must_use]
pub fn sp1_extension_mul_matrix(a: &SP1ExtensionField) -> [[SP1Field; 4]; 4] {
    let limbs = <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(a);
    let a0 = limbs[0];
    let a1 = limbs[1];
    let a2 = limbs[2];
    let a3 = limbs[3];
    let w = sp1_ext_w();
    [[a0, w * a3, w * a2, w * a1], [a1, a0, w * a3, w * a2], [a2, a1, a0, w * a3], [a3, a2, a1, a0]]
}

/// Apply the base-field multiplication matrix for `a` to an input extension limb vector.
#[must_use]
pub fn apply_sp1_extension_mul_matrix(
    matrix: &[[SP1Field; 4]; 4],
    x_limbs: &[SP1Field; 4],
) -> [SP1Field; 4] {
    let mut out = [SP1Field::zero(); 4];
    for row in 0..4 {
        let mut acc = SP1Field::zero();
        for col in 0..4 {
            acc += matrix[row][col] * x_limbs[col];
        }
        out[row] = acc;
    }
    out
}

/// Linearize extension weights into 4 base-field forms over a limb-flattened witness.
///
/// If the witness is flattened as
/// `[x_0[0], x_0[1], x_0[2], x_0[3], x_1[0], ...]`,
/// then the returned vectors `forms[row]` have length `4 * weights.len()` and satisfy:
///
/// `sum_j forms[row][j] * flat_x[j] = coeff_row(sum_i weights[i] * x_i)`.
#[must_use]
pub fn linearize_extension_weights_to_base_forms(
    weights: &[SP1ExtensionField],
) -> [Vec<SP1Field>; 4] {
    let mut forms = core::array::from_fn(|_| Vec::with_capacity(weights.len() * 4));
    for weight in weights {
        let matrix = sp1_extension_mul_matrix(weight);
        for row in 0..4 {
            forms[row].extend(matrix[row]);
        }
    }
    forms
}

#[must_use]
pub fn flatten_extension_limbs(values: &[SP1ExtensionField]) -> Vec<SP1Field> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext(a0: u32, a1: u32, a2: u32, a3: u32) -> SP1ExtensionField {
        SP1ExtensionField::from_base_slice(&[
            SP1Field::from_canonical_u32(a0),
            SP1Field::from_canonical_u32(a1),
            SP1Field::from_canonical_u32(a2),
            SP1Field::from_canonical_u32(a3),
        ])
    }

    #[test]
    fn multiplication_matrix_matches_extension_product() {
        let a = ext(2, 3, 5, 7);
        let x = ext(11, 13, 17, 19);
        let matrix = sp1_extension_mul_matrix(&a);
        let x_limbs: [SP1Field; 4] =
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&x)
                .try_into()
                .expect("4 limbs");
        let got_limbs = apply_sp1_extension_mul_matrix(&matrix, &x_limbs);
        let got = SP1ExtensionField::from_base_slice(&got_limbs);
        assert_eq!(got, a * x);
    }

    #[test]
    fn linearized_forms_match_extension_linear_combination() {
        let weights = [ext(2, 3, 0, 1), ext(5, 7, 11, 13)];
        let inputs = [ext(17, 19, 23, 29), ext(31, 37, 41, 43)];
        let flat_inputs = flatten_extension_limbs(&inputs);
        let forms = linearize_extension_weights_to_base_forms(&weights);
        let expected = weights[0] * inputs[0] + weights[1] * inputs[1];
        let expected_limbs =
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&expected);

        for row in 0..4 {
            let mut acc = SP1Field::zero();
            for (coeff, x) in forms[row].iter().zip(flat_inputs.iter()) {
                acc += *coeff * *x;
            }
            assert_eq!(acc, expected_limbs[row]);
        }
    }
}

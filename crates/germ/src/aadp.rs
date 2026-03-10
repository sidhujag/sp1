//! Arithmetic ADP witness-encryption backend.
//!
//! This is the paper-style AADP core (constraint-system compilation into randomized matrix
//! ciphertexts) rather than a symmetric lock wrapper.
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::RngCore;
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use sp1_primitives::{SP1ExtensionField, SP1Field};

pub trait AadpField:
    Copy
    + Clone
    + core::fmt::Debug
    + Send
    + Sync
    + Eq
    + PartialEq
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn rand(rng: &mut impl RngCore) -> Self;
    fn from_u64(v: u64) -> Self;
    fn from_u128(v: u128) -> Self;
    fn inverse(self) -> Option<Self>;
    fn is_zero(&self) -> bool;
    fn to_le_words_u64(&self) -> Vec<u64>;
}

const SP1_EXT_LIMBS: usize = 4;

fn sp1_ext_from_u128(mut value: u128) -> SP1ExtensionField {
    let radix = SP1Field::ORDER_U32 as u128;
    let limbs: [SP1Field; SP1_EXT_LIMBS] = core::array::from_fn(|_| {
        let digit = (value % radix) as u32;
        value /= radix;
        SP1Field::from_canonical_u32(digit)
    });
    SP1ExtensionField::from_base_slice(&limbs)
}

fn sp1_ext_to_u128(value: &SP1ExtensionField) -> Result<u128, String> {
    let radix = SP1Field::ORDER_U32 as u128;
    let mut acc = 0u128;
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value)
        .iter()
        .rev()
    {
        let digit = limb.as_canonical_u32() as u128;
        acc = acc
            .checked_mul(radix)
            .ok_or_else(|| "SP1ExtensionField limb pack overflow".to_string())?;
        acc = acc
            .checked_add(digit)
            .ok_or_else(|| "SP1ExtensionField limb pack overflow".to_string())?;
    }
    Ok(acc)
}

impl AadpField for SP1ExtensionField {
    fn zero() -> Self {
        SP1ExtensionField::from_base_slice(&[
            SP1Field::zero(),
            SP1Field::zero(),
            SP1Field::zero(),
            SP1Field::zero(),
        ])
    }

    fn one() -> Self {
        SP1ExtensionField::from_base_slice(&[
            SP1Field::one(),
            SP1Field::zero(),
            SP1Field::zero(),
            SP1Field::zero(),
        ])
    }

    fn rand(rng: &mut impl RngCore) -> Self {
        let limbs: [SP1Field; SP1_EXT_LIMBS] =
            core::array::from_fn(|_| SP1Field::from_wrapped_u32(rng.next_u32()));
        SP1ExtensionField::from_base_slice(&limbs)
    }

    fn from_u64(v: u64) -> Self {
        sp1_ext_from_u128(v as u128)
    }

    fn from_u128(v: u128) -> Self {
        sp1_ext_from_u128(v)
    }

    fn inverse(self) -> Option<Self> {
        self.try_inverse()
    }

    fn is_zero(&self) -> bool {
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(self)
            .iter()
            .all(|x| x.as_canonical_u32() == 0)
    }

    fn to_le_words_u64(&self) -> Vec<u64> {
        let packed = sp1_ext_to_u128(self).unwrap_or(0);
        vec![packed as u64, (packed >> 64) as u64]
    }
}

/// Sparse linear form over witness variables, with an explicit constant term.
#[derive(Clone, Debug, Default)]
pub struct AadpLinearForm<F: AadpField> {
    pub constant: F,
    pub terms: Vec<(usize, F)>,
}

impl<F: AadpField> AadpLinearForm<F> {
    pub fn eval(&self, witness: &[F]) -> Result<F, String> {
        let mut acc = self.constant;
        for &(idx, coeff) in &self.terms {
            let value = witness
                .get(idx)
                .ok_or_else(|| format!("linear form witness index out of range: idx={idx}"))?;
            acc += coeff * *value;
        }
        Ok(acc)
    }
}

/// One arithmetic constraint `a(x) * b(x) = c(x) * d(x)`.
#[derive(Clone, Debug)]
pub struct AadpMulConstraint<F: AadpField> {
    pub a: AadpLinearForm<F>,
    pub b: AadpLinearForm<F>,
    pub c: AadpLinearForm<F>,
    pub d: AadpLinearForm<F>,
}

impl<F: AadpField> AadpMulConstraint<F> {
    pub fn eval_holds(&self, witness: &[F]) -> Result<bool, String> {
        let a = self.a.eval(witness)?;
        let b = self.b.eval(witness)?;
        let c = self.c.eval(witness)?;
        let d = self.d.eval(witness)?;
        Ok(a * b == c * d)
    }
}

/// Arithmetic constraint system consumed by AADP.
#[derive(Clone, Debug, Default)]
pub struct AadpConstraintSystem<F: AadpField> {
    pub num_variables: usize,
    pub constraints: Vec<AadpMulConstraint<F>>,
}

impl<F: AadpField> AadpConstraintSystem<F> {
    #[must_use]
    pub fn matrix_dim(&self) -> usize {
        self.constraints
            .len()
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .unwrap_or(1)
    }

    pub fn check_witness(&self, witness: &[F]) -> Result<(), String> {
        if witness.len() != self.num_variables {
            return Err(format!(
                "AADP witness length mismatch: got={} expected={}",
                witness.len(),
                self.num_variables
            ));
        }
        for (i, constraint) in self.constraints.iter().enumerate() {
            if !constraint.eval_holds(witness)? {
                return Err(format!("AADP witness violates constraint {i}"));
            }
        }
        Ok(())
    }
}

/// One randomized AADP ciphertext.
///
/// `matrices[0]` is the constant matrix `M0` and `matrices[i + 1]` corresponds to witness variable `x_i`.
#[derive(Clone, Debug)]
pub struct AadpCiphertext<F: AadpField> {
    pub num_variables: usize,
    pub dim: usize,
    pub matrices: Vec<Vec<F>>,
}

/// Byte-wise AADP encryption helper.
#[derive(Clone, Debug)]
pub struct AadpByteCiphertext<F: AadpField> {
    pub parts: Vec<AadpCiphertext<F>>,
}

impl<F: AadpField> AadpCiphertext<F> {
    pub fn evaluate(&self, witness: &[F]) -> Result<Vec<F>, String> {
        if witness.len() != self.num_variables {
            return Err(format!(
                "AADP evaluate witness length mismatch: got={} expected={}",
                witness.len(),
                self.num_variables
            ));
        }
        if self.matrices.len() != self.num_variables + 1 {
            return Err("AADP ciphertext matrix count mismatch".to_string());
        }
        let mut out = self
            .matrices
            .first()
            .cloned()
            .ok_or_else(|| "AADP ciphertext has no constant matrix".to_string())?;
        let nn = self
            .dim
            .checked_mul(self.dim)
            .ok_or_else(|| "AADP evaluate dim^2 overflow".to_string())?;
        if out.len() != nn {
            return Err("AADP constant matrix size mismatch".to_string());
        }
        for (i, &x) in witness.iter().enumerate() {
            let m = self
                .matrices
                .get(i + 1)
                .ok_or_else(|| format!("AADP missing witness matrix {i}"))?;
            if m.len() != nn {
                return Err(format!("AADP witness matrix size mismatch at index {i}"));
            }
            for j in 0..nn {
                out[j] += m[j] * x;
            }
        }
        Ok(out)
    }

    pub fn decrypt_scalar(&self, witness: &[F]) -> Result<F, String> {
        let eval = self.evaluate(witness)?;
        let det_eval = determinant(self.dim, eval.as_slice())?;
        let coeff = if self.dim == 1 {
            F::one()
        } else {
            let mut minor = Vec::with_capacity((self.dim - 1) * (self.dim - 1));
            for r in 0..(self.dim - 1) {
                for c in 0..(self.dim - 1) {
                    minor.push(eval[r * self.dim + c]);
                }
            }
            determinant(self.dim - 1, minor.as_slice())?
        };
        if coeff.is_zero() {
            return Err("AADP decrypt failed: bottom-right cofactor is zero".to_string());
        }
        let inv = coeff
            .inverse()
            .ok_or_else(|| "AADP decrypt failed: missing inverse for nonzero cofactor".to_string())?;
        Ok(det_eval * inv)
    }

    pub fn decrypt_u128(&self, witness: &[F]) -> Result<u128, String> {
        field_to_u128(self.decrypt_scalar(witness)?)
    }
}

pub fn aadp_encrypt_u128<F: AadpField, R: RngCore>(
    cs: &AadpConstraintSystem<F>,
    msg: u128,
    rng: &mut R,
) -> Result<AadpCiphertext<F>, String> {
    aadp_encrypt_scalar(cs, F::from_u128(msg), rng)
}

pub fn aadp_encrypt_bytes<F: AadpField, R: RngCore>(
    cs: &AadpConstraintSystem<F>,
    msg: &[u8],
    rng: &mut R,
) -> Result<AadpByteCiphertext<F>, String> {
    let mut parts = Vec::with_capacity(msg.len());
    for &b in msg {
        parts.push(aadp_encrypt_scalar(cs, F::from_u64(b as u64), rng)?);
    }
    Ok(AadpByteCiphertext { parts })
}

/// Encrypt one field element under the paper's AADP matrix construction.
pub fn aadp_encrypt_scalar<F: AadpField, R: RngCore>(
    cs: &AadpConstraintSystem<F>,
    msg: F,
    rng: &mut R,
) -> Result<AadpCiphertext<F>, String> {
    if cs.constraints.is_empty() {
        return Err("AADP requires at least one constraint".to_string());
    }
    let dim = cs.matrix_dim();
    let nn = dim
        .checked_mul(dim)
        .ok_or_else(|| "AADP dim^2 overflow".to_string())?;
    let mut matrices = vec![vec![F::zero(); nn]; cs.num_variables + 1];

    for constraint in &cs.constraints {
        let l = random_matrix::<F, R>(dim, 4, rng);
        let r = random_matrix::<F, R>(4, dim, rng);

        let b_a = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::A)?;
        let b_b = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::B)?;
        let b_c = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::C)?;
        let b_d = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::D)?;
        let b_xi = basis_matrix_contribution::<F>(dim, &l, &r, AadpBasis::Xi)?;

        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.a, b_a.as_slice())?;
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.b, b_b.as_slice())?;
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.c, b_c.as_slice())?;
        add_linear_form_to_matrices(matrices.as_mut_slice(), &constraint.d, b_d.as_slice())?;

        let mut xi = AadpLinearForm::<F> {
            constant: F::rand(rng),
            terms: Vec::with_capacity(cs.num_variables),
        };
        for idx in 0..cs.num_variables {
            xi.terms.push((idx, F::rand(rng)));
        }
        add_linear_form_to_matrices(matrices.as_mut_slice(), &xi, b_xi.as_slice())?;
    }

    let last = dim - 1;
    matrices[0][last * dim + last] += msg;

    Ok(AadpCiphertext { num_variables: cs.num_variables, dim, matrices })
}

#[derive(Clone, Copy)]
enum AadpBasis {
    A,
    B,
    C,
    D,
    Xi,
}

fn add_linear_form_to_matrices<F: AadpField>(
    matrices: &mut [Vec<F>],
    form: &AadpLinearForm<F>,
    basis_matrix: &[F],
) -> Result<(), String> {
    if matrices.is_empty() {
        return Err("AADP matrices are empty".to_string());
    }
    if matrices[0].len() != basis_matrix.len() {
        return Err("AADP basis matrix size mismatch".to_string());
    }
    for (dst, src) in matrices[0].iter_mut().zip(basis_matrix.iter()) {
        *dst += *src * form.constant;
    }
    for &(idx, coeff) in &form.terms {
        let dst = matrices
            .get_mut(idx + 1)
            .ok_or_else(|| format!("AADP variable index out of range: idx={idx}"))?;
        if dst.len() != basis_matrix.len() {
            return Err(format!("AADP variable matrix size mismatch at idx={idx}"));
        }
        for (d, s) in dst.iter_mut().zip(basis_matrix.iter()) {
            *d += *s * coeff;
        }
    }
    Ok(())
}

fn basis_matrix_contribution<F: AadpField>(
    dim: usize,
    l: &[F],
    r: &[F],
    basis: AadpBasis,
) -> Result<Vec<F>, String> {
    if l.len() != dim * 4 || r.len() != 4 * dim {
        return Err("AADP L/R shape mismatch".to_string());
    }
    let mut out = vec![F::zero(); dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            let v = match basis {
                AadpBasis::A => l[row * 4] * r[col] + l[row * 4 + 3] * r[3 * dim + col],
                AadpBasis::B => l[row * 4 + 1] * r[dim + col] + l[row * 4 + 2] * r[2 * dim + col],
                AadpBasis::C => l[row * 4] * r[dim + col] + l[row * 4 + 2] * r[3 * dim + col],
                AadpBasis::D => l[row * 4 + 1] * r[col] + l[row * 4 + 3] * r[2 * dim + col],
                AadpBasis::Xi => {
                    -l[row * 4] * r[2 * dim + col] + l[row * 4 + 1] * r[3 * dim + col]
                }
            };
            out[row * dim + col] = v;
        }
    }
    Ok(out)
}

fn random_matrix<F: AadpField, R: RngCore>(rows: usize, cols: usize, rng: &mut R) -> Vec<F> {
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        out.push(F::rand(rng));
    }
    out
}

fn determinant<F: AadpField>(dim: usize, data: &[F]) -> Result<F, String> {
    if dim == 0 {
        return Ok(F::one());
    }
    if data.len() != dim * dim {
        return Err(format!(
            "AADP determinant size mismatch: len={} expected={}",
            data.len(),
            dim * dim
        ));
    }
    let mut a = data.to_vec();
    let mut det = F::one();
    for i in 0..dim {
        let mut pivot = i;
        while pivot < dim && a[pivot * dim + i].is_zero() {
            pivot += 1;
        }
        if pivot == dim {
            return Ok(F::zero());
        }
        if pivot != i {
            for c in 0..dim {
                a.swap(i * dim + c, pivot * dim + c);
            }
            det = -det;
        }
        let pivot_val = a[i * dim + i];
        det *= pivot_val;
        let inv = pivot_val
            .inverse()
            .ok_or_else(|| "AADP pivot inverse unexpectedly missing".to_string())?;
        for row in (i + 1)..dim {
            let factor = a[row * dim + i] * inv;
            if factor.is_zero() {
                continue;
            }
            for col in i..dim {
                let idx = row * dim + col;
                let src = a[i * dim + col];
                a[idx] -= factor * src;
            }
        }
    }
    Ok(det)
}

fn field_to_u128<F: AadpField>(x: F) -> Result<u128, String> {
    let limbs = x.to_le_words_u64();
    if limbs.len() > 2 && limbs[2..].iter().any(|&w| w != 0) {
        return Err("AADP decrypted field element does not fit in u128".to_string());
    }
    let lo = *limbs.first().unwrap_or(&0u64) as u128;
    let hi = *limbs.get(1).unwrap_or(&0u64) as u128;
    Ok(lo | (hi << 64))
}

impl<F: AadpField> AadpByteCiphertext<F> {
    pub fn decrypt_bytes(&self, witness: &[F]) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(self.parts.len());
        for (i, part) in self.parts.iter().enumerate() {
            let x = part.decrypt_scalar(witness)?;
            let limbs = x.to_le_words_u64();
            if limbs.len() > 1 && limbs[1..].iter().any(|&w| w != 0) {
                return Err(format!("AADP decrypted byte part does not fit in u8 at index {i}"));
            }
            let b = *limbs.first().unwrap_or(&0u64);
            if b > 255 {
                return Err(format!("AADP decrypted byte part out of range at index {i}: {b}"));
            }
            out.push(b as u8);
        }
        Ok(out)
    }
}

/// Canonical SP1 field type used by this crate's AADP path.
pub type Sp1AadpField = SP1ExtensionField;

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};
    use sp1_primitives::{SP1ExtensionField, SP1Field};

    use super::*;

    fn ext_zero() -> SP1ExtensionField {
        <SP1ExtensionField as AadpField>::zero()
    }

    fn ext_one() -> SP1ExtensionField {
        <SP1ExtensionField as AadpField>::one()
    }

    fn ext_from_u32(v: u32) -> SP1ExtensionField {
        SP1ExtensionField::from_base_slice(&[
            SP1Field::from_canonical_u32(v),
            SP1Field::zero(),
            SP1Field::zero(),
            SP1Field::zero(),
        ])
    }

    #[test]
    fn test_aadp_encrypt_decrypt_scalar_simple_projective_safe_bitcheck() {
        let cs = AadpConstraintSystem::<Sp1AadpField> {
            num_variables: 1,
            constraints: vec![AadpMulConstraint {
                a: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
                b: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
                c: AadpLinearForm { constant: ext_one(), terms: Vec::new() },
                d: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
            }],
        };
        let witness = vec![ext_one()];
        cs.check_witness(witness.as_slice()).expect("witness holds");

        let mut rng = StdRng::seed_from_u64(42);
        let msg = ext_from_u32(77);
        let ct = aadp_encrypt_scalar(&cs, msg, &mut rng).expect("aadp encrypt");
        let got = ct.decrypt_scalar(witness.as_slice()).expect("aadp decrypt");
        assert_eq!(got, msg);
    }

    #[test]
    fn test_aadp_encrypt_decrypt_u128() {
        let cs = AadpConstraintSystem::<Sp1AadpField> {
            num_variables: 1,
            constraints: vec![AadpMulConstraint {
                a: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
                b: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
                c: AadpLinearForm { constant: ext_one(), terms: Vec::new() },
                d: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
            }],
        };
        let witness = vec![ext_one()];
        let mut rng = StdRng::seed_from_u64(99);
        let ct = aadp_encrypt_u128(&cs, 123u128, &mut rng).expect("aadp encrypt");
        let got = ct.decrypt_u128(witness.as_slice()).expect("aadp decrypt");
        assert_eq!(got, 123u128);
    }

    #[test]
    fn test_aadp_encrypt_decrypt_bytes() {
        let cs = AadpConstraintSystem::<Sp1AadpField> {
            num_variables: 1,
            constraints: vec![AadpMulConstraint {
                a: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
                b: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
                c: AadpLinearForm { constant: ext_one(), terms: Vec::new() },
                d: AadpLinearForm { constant: ext_zero(), terms: vec![(0, ext_one())] },
            }],
        };
        let witness = vec![ext_one()];
        let mut rng = StdRng::seed_from_u64(1234);
        let msg = [0u8, 1, 2, 3, 250, 251, 252, 253, 254, 255];
        let ct = aadp_encrypt_bytes(&cs, &msg, &mut rng).expect("aadp encrypt");
        let got = ct.decrypt_bytes(witness.as_slice()).expect("aadp decrypt");
        assert_eq!(got, msg);
    }
}

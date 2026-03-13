//! Native KoalaBear coefficient-form cyclotomic ring helpers.
//!
//! This module is the first step toward a native Orbweaver-style opening layer
//! over the same prime field family used by `sp1-germ`.

use core::array::from_fn;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use slop_algebra::{AbstractField, PrimeField32};
use sp1_primitives::SP1Field;

pub const KOALA_RING64_DIM: usize = 64;

/// Coefficient-form ring `F[X]/(X^64 + 1)` over the SP1 base field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KoalaRing64 {
    coeffs: [SP1Field; KOALA_RING64_DIM],
}

impl Default for KoalaRing64 {
    fn default() -> Self {
        Self::zero()
    }
}

impl KoalaRing64 {
    #[must_use]
    pub fn zero() -> Self {
        Self { coeffs: [SP1Field::zero(); KOALA_RING64_DIM] }
    }

    #[must_use]
    pub fn one() -> Self {
        Self::from_scalar(SP1Field::one())
    }

    #[must_use]
    pub fn from_scalar(scalar: SP1Field) -> Self {
        let mut coeffs = [SP1Field::zero(); KOALA_RING64_DIM];
        coeffs[0] = scalar;
        Self { coeffs }
    }

    #[must_use]
    pub fn from_coeffs(coeffs: [SP1Field; KOALA_RING64_DIM]) -> Self {
        Self { coeffs }
    }

    #[must_use]
    pub fn monomial(degree: usize, coeff: SP1Field) -> Self {
        assert!(degree < KOALA_RING64_DIM, "monomial degree out of range");
        let mut coeffs = [SP1Field::zero(); KOALA_RING64_DIM];
        coeffs[degree] = coeff;
        Self { coeffs }
    }

    #[must_use]
    pub fn coeffs(&self) -> &[SP1Field; KOALA_RING64_DIM] {
        &self.coeffs
    }

    pub fn coeffs_mut(&mut self) -> &mut [SP1Field; KOALA_RING64_DIM] {
        &mut self.coeffs
    }

    /// Centered `l_infinity` norm using the canonical coefficient representatives.
    #[must_use]
    pub fn l_inf_norm_u32(&self) -> u32 {
        self.coeffs.iter().map(centered_coeff_abs_u32).max().unwrap_or(0)
    }

    #[must_use]
    pub fn dot(lhs: &[Self], rhs: &[Self]) -> Result<Self, String> {
        if lhs.len() != rhs.len() {
            return Err(format!(
                "KoalaRing64 dot length mismatch: lhs={} rhs={}",
                lhs.len(),
                rhs.len()
            ));
        }
        let mut acc = Self::zero();
        for (x, y) in lhs.iter().zip(rhs.iter()) {
            acc += *x * *y;
        }
        Ok(acc)
    }

    #[must_use]
    fn mul_negacyclic(self, rhs: Self) -> Self {
        let mut coeffs = [SP1Field::zero(); KOALA_RING64_DIM];
        for i in 0..KOALA_RING64_DIM {
            for j in 0..KOALA_RING64_DIM {
                let deg = i + j;
                let term = self.coeffs[i] * rhs.coeffs[j];
                if deg < KOALA_RING64_DIM {
                    coeffs[deg] += term;
                } else {
                    coeffs[deg - KOALA_RING64_DIM] -= term;
                }
            }
        }
        Self { coeffs }
    }
}

impl Add for KoalaRing64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { coeffs: from_fn(|idx| self.coeffs[idx] + rhs.coeffs[idx]) }
    }
}

impl AddAssign for KoalaRing64 {
    fn add_assign(&mut self, rhs: Self) {
        for idx in 0..KOALA_RING64_DIM {
            self.coeffs[idx] += rhs.coeffs[idx];
        }
    }
}

impl Sub for KoalaRing64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { coeffs: from_fn(|idx| self.coeffs[idx] - rhs.coeffs[idx]) }
    }
}

impl SubAssign for KoalaRing64 {
    fn sub_assign(&mut self, rhs: Self) {
        for idx in 0..KOALA_RING64_DIM {
            self.coeffs[idx] -= rhs.coeffs[idx];
        }
    }
}

impl Neg for KoalaRing64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { coeffs: from_fn(|idx| -self.coeffs[idx]) }
    }
}

impl Mul for KoalaRing64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_negacyclic(rhs)
    }
}

impl MulAssign for KoalaRing64 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul<SP1Field> for KoalaRing64 {
    type Output = Self;

    fn mul(self, rhs: SP1Field) -> Self::Output {
        Self { coeffs: from_fn(|idx| self.coeffs[idx] * rhs) }
    }
}

impl MulAssign<SP1Field> for KoalaRing64 {
    fn mul_assign(&mut self, rhs: SP1Field) {
        for coeff in &mut self.coeffs {
            *coeff *= rhs;
        }
    }
}

impl Sum for KoalaRing64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Self::zero();
        for item in iter {
            acc += item;
        }
        acc
    }
}

impl<'a> Sum<&'a Self> for KoalaRing64 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut acc = Self::zero();
        for item in iter {
            acc += *item;
        }
        acc
    }
}

impl Product for KoalaRing64 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Self::one();
        for item in iter {
            acc *= item;
        }
        acc
    }
}

impl<'a> Product<&'a Self> for KoalaRing64 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        let mut acc = Self::one();
        for item in iter {
            acc *= *item;
        }
        acc
    }
}

fn centered_coeff_abs_u32(value: &SP1Field) -> u32 {
    let canonical = value.as_canonical_u32();
    let modulus = SP1Field::ORDER_U32;
    let neg = modulus - canonical;
    canonical.min(neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(x: u32) -> SP1Field {
        SP1Field::from_canonical_u32(x)
    }

    #[test]
    fn test_negacyclic_wrap() {
        let x63 = KoalaRing64::monomial(63, SP1Field::one());
        let x1 = KoalaRing64::monomial(1, SP1Field::one());
        let got = x63 * x1;
        let expected = KoalaRing64::from_scalar(-SP1Field::one());
        assert_eq!(got, expected);
    }

    #[test]
    fn test_distributive_small_product() {
        let a = KoalaRing64::from_scalar(scalar(3)) + KoalaRing64::monomial(1, scalar(5));
        let b = KoalaRing64::from_scalar(scalar(7)) + KoalaRing64::monomial(2, scalar(11));
        let c = KoalaRing64::from_scalar(scalar(13));
        assert_eq!(a * (b + c), (a * b) + (a * c));
    }

    #[test]
    fn test_l_inf_norm_centered() {
        let minus_one = -SP1Field::one();
        let ring = KoalaRing64::monomial(17, minus_one);
        assert_eq!(ring.l_inf_norm_u32(), 1);
    }

    #[test]
    fn test_dot() {
        let lhs = [KoalaRing64::from_scalar(scalar(2)), KoalaRing64::monomial(3, scalar(4))];
        let rhs = [KoalaRing64::from_scalar(scalar(5)), KoalaRing64::monomial(1, scalar(6))];
        let got = KoalaRing64::dot(&lhs, &rhs).expect("dot");
        let expected = (lhs[0] * rhs[0]) + (lhs[1] * rhs[1]);
        assert_eq!(got, expected);
    }
}

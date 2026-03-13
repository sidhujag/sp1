//! Minimal Orbweaver opening-layer algebra over the native Koala ring backend.
//!
//! This module intentionally implements only the *opening side* equations:
//! - `vk_f = PreVerify(f)`
//! - `⟨a0, π0⟩ = vk_f * c - y`
//!
//! It does not yet implement the trapdoor-backed SRS generation or opening prover.

use std::path::Path;

use sha2::{Digest, Sha256};
use slop_algebra::AbstractExtensionField;
use slop_algebra::{AbstractField, Field, PrimeField32};
use sp1_primitives::{SP1ExtensionField, SP1Field};

use crate::{
    bundle::{Sp1MulTerm, Sp1MulTerminalOpeningProofs},
    ext_linear::{flatten_extension_limbs, linearize_extension_weights_to_base_forms},
    koala_ring::KoalaRing64,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverOpeningSrs {
    /// Public opening-check vector `a0`.
    pub a0: Vec<KoalaRing64>,
    /// Public scalar/ring element `v`.
    pub v: KoalaRing64,
    /// Opening-layer preimages `u_{0,i}` for positive powers of `v`.
    ///
    /// The entry at index `i` stores the ring vector whose inner product with `a0`
    /// equals `v^i`. Index `0` is unused by the opening proof.
    pub u0_positive: Vec<Vec<KoalaRing64>>,
    /// Opening-layer preimages `u_{0,-i}` for negative powers of `v`.
    ///
    /// The entry at index `i` stores the ring vector whose inner product with `a0`
    /// equals `v^{-i}`. Index `0` is unused by the opening proof.
    pub u0_negative: Vec<Vec<KoalaRing64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverOpeningProof {
    pub pi0: Vec<KoalaRing64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverOpeningVerificationKey {
    pub value: KoalaRing64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbweaverTerminalField {
    A,
    B,
    C,
    D,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverTerminalValueProof {
    pub opened_value: SP1ExtensionField,
    pub limb_proofs: [OrbweaverOpeningProof; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverScalarOpeningProof {
    pub opened_value: SP1Field,
    pub pi0: Vec<KoalaRing64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverExtensionOpeningSrs {
    pub a0: Vec<SP1ExtensionField>,
    pub alpha: SP1ExtensionField,
    pub u0_positive: Vec<Vec<SP1ExtensionField>>,
    pub u0_negative: Vec<Vec<SP1ExtensionField>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbweaverExtensionOpeningProof {
    pub opened_value: SP1ExtensionField,
    pub pi0: Vec<SP1ExtensionField>,
}

pub fn terminal_value_proof_to_extension_image(
    proof: &OrbweaverTerminalValueProof,
) -> Result<OrbweaverExtensionOpeningProof, String> {
    let len = proof.limb_proofs[0].pi0.len();
    if proof.limb_proofs.iter().any(|limb| limb.pi0.len() != len) {
        return Err("Orbweaver limb proof length mismatch".to_string());
    }
    let mut pi0 = Vec::with_capacity(len);
    for idx in 0..len {
        let limbs = [
            scalar_ring_element_to_extension(&proof.limb_proofs[0].pi0[idx])?,
            scalar_ring_element_to_extension(&proof.limb_proofs[1].pi0[idx])?,
            scalar_ring_element_to_extension(&proof.limb_proofs[2].pi0[idx])?,
            scalar_ring_element_to_extension(&proof.limb_proofs[3].pi0[idx])?,
        ];
        let base_limbs = [
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&limbs[0])[0],
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&limbs[1])[0],
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&limbs[2])[0],
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&limbs[3])[0],
        ];
        pi0.push(SP1ExtensionField::from_base_slice(&base_limbs));
    }
    Ok(OrbweaverExtensionOpeningProof { opened_value: proof.opened_value, pi0 })
}

#[must_use]
pub fn srs_max_supported_width(srs: &OrbweaverOpeningSrs) -> usize {
    srs.u0_positive.len().saturating_sub(1).min(srs.u0_negative.len().saturating_sub(1))
}

pub fn validate_srs(srs: &OrbweaverOpeningSrs, width: usize) -> Result<(), String> {
    if srs.a0.is_empty() {
        return Err("Orbweaver SRS has empty a0".to_string());
    }
    if width > srs_max_supported_width(srs) {
        return Err(format!(
            "Orbweaver SRS width unsupported: requested={} supported={}",
            width,
            srs_max_supported_width(srs)
        ));
    }
    let positive = derive_positive_powers(srs, width)?;
    let negative = derive_negative_powers(srs, width)?;
    let mut expected_pos = srs.v;
    let expected_neg_base = if width > 0 { negative[0] } else { KoalaRing64::one() };
    let mut expected_neg = expected_neg_base;
    for idx in 0..width {
        if positive[idx] != expected_pos {
            return Err(format!("Orbweaver positive power check failed at index {}", idx + 1));
        }
        if negative[idx] != expected_neg {
            return Err(format!("Orbweaver negative power check failed at index {}", idx + 1));
        }
        expected_pos *= srs.v;
        expected_neg *= expected_neg_base;
    }
    Ok(())
}

pub fn scalarize_srs_to_extension(
    srs: &OrbweaverOpeningSrs,
    width: usize,
) -> Result<OrbweaverExtensionOpeningSrs, String> {
    validate_srs(srs, width)?;
    let alpha = scalar_ring_element_to_extension(&srs.v)?;
    let a0 = srs.a0.iter().map(scalar_ring_element_to_extension).collect::<Result<Vec<_>, _>>()?;
    let u0_positive = srs
        .u0_positive
        .iter()
        .take(width + 1)
        .map(|vec| vec.iter().map(scalar_ring_element_to_extension).collect::<Result<Vec<_>, _>>())
        .collect::<Result<Vec<_>, _>>()?;
    let u0_negative = srs
        .u0_negative
        .iter()
        .take(width + 1)
        .map(|vec| vec.iter().map(scalar_ring_element_to_extension).collect::<Result<Vec<_>, _>>())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OrbweaverExtensionOpeningSrs { a0, alpha, u0_positive, u0_negative })
}

const LOCAL_DEV_GADGET_BASE_U32: u32 = 1 << 8;
const LOCAL_DEV_GADGET_DIGITS: usize = 4;

/// Deterministic development-only SRS for exercising the Orbweaver opening path.
///
/// This is **not** a secure trapdoor-generated setup. However, unlike the earlier
/// one-slot toy path, it uses a genuine short-preimage gadget basis:
/// each opening preimage is a vector of small base-`2^8` digits under a fixed
/// scalar gadget vector. This keeps the proof coordinates small and matches the
/// intended "verifier image of a structured ring proof" shape more closely.
#[must_use]
pub fn generate_local_dev_srs(width: usize) -> OrbweaverOpeningSrs {
    let v_scalar = SP1Field::from_canonical_u32(7);
    let v = KoalaRing64::from_scalar(v_scalar);
    let v_inv_scalar = v_scalar.try_inverse().expect("nonzero local dev SRS generator");
    let v_inv = KoalaRing64::from_scalar(v_inv_scalar);
    let mut u0_positive = vec![vec![KoalaRing64::zero(); LOCAL_DEV_GADGET_DIGITS]; width + 1];
    let mut u0_negative = vec![vec![KoalaRing64::zero(); LOCAL_DEV_GADGET_DIGITS]; width + 1];
    let mut cur_pos = v;
    let mut cur_neg = v_inv;
    for idx in 1..=width {
        u0_positive[idx] = local_dev_short_preimage_for_scalar(cur_pos.coeffs()[0]);
        u0_negative[idx] = local_dev_short_preimage_for_scalar(cur_neg.coeffs()[0]);
        cur_pos *= v;
        cur_neg *= v_inv;
    }
    OrbweaverOpeningSrs { a0: local_dev_gadget_vector(), v, u0_positive, u0_negative }
}

pub fn encode_srs(srs: &OrbweaverOpeningSrs) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(srs.a0.len() as u32).to_le_bytes());
    for ring in &srs.a0 {
        write_ring64(&mut out, ring);
    }
    write_ring64(&mut out, &srs.v);
    encode_ring_vector_family(&mut out, srs.u0_positive.as_slice());
    encode_ring_vector_family(&mut out, srs.u0_negative.as_slice());
    out
}

pub fn decode_srs(bytes: &[u8]) -> Result<OrbweaverOpeningSrs, String> {
    let mut cursor = 0usize;
    let a0_len = read_u32(bytes, &mut cursor)? as usize;
    let mut a0 = Vec::with_capacity(a0_len);
    for _ in 0..a0_len {
        a0.push(read_ring64(bytes, &mut cursor)?);
    }
    let v = read_ring64(bytes, &mut cursor)?;
    let u0_positive = decode_ring_vector_family(bytes, &mut cursor)?;
    let u0_negative = decode_ring_vector_family(bytes, &mut cursor)?;
    if cursor != bytes.len() {
        return Err("Orbweaver SRS had trailing bytes".to_string());
    }
    Ok(OrbweaverOpeningSrs { a0, v, u0_positive, u0_negative })
}

#[must_use]
pub fn digest_srs(srs: &OrbweaverOpeningSrs) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"sp1-germ/orbweaver-opening-srs/v1");
    h.update((srs.a0.len() as u64).to_le_bytes());
    for ring in &srs.a0 {
        hash_ring64(&mut h, ring);
    }
    hash_ring64(&mut h, &srs.v);
    hash_ring_vector_family(&mut h, srs.u0_positive.as_slice());
    hash_ring_vector_family(&mut h, srs.u0_negative.as_slice());
    h.finalize().into()
}

pub fn write_srs_to_file(path: &Path, srs: &OrbweaverOpeningSrs) -> Result<(), String> {
    std::fs::write(path, encode_srs(srs))
        .map_err(|err| format!("write Orbweaver SRS to {}: {err}", path.display()))
}

pub fn read_srs_from_file(
    path: &Path,
    required_width: usize,
) -> Result<OrbweaverOpeningSrs, String> {
    let bytes = std::fs::read(path)
        .map_err(|err| format!("read Orbweaver SRS from {}: {err}", path.display()))?;
    let srs = decode_srs(&bytes)?;
    validate_srs(&srs, required_width)?;
    Ok(srs)
}

pub fn encode_opening_proof(proof: &OrbweaverOpeningProof) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(proof.pi0.len() as u32).to_le_bytes());
    for ring in &proof.pi0 {
        write_ring64(&mut out, ring);
    }
    out
}

pub fn decode_opening_proof(bytes: &[u8]) -> Result<OrbweaverOpeningProof, String> {
    let mut cursor = 0usize;
    let len = read_u32(bytes, &mut cursor)? as usize;
    let mut pi0 = Vec::with_capacity(len);
    for _ in 0..len {
        pi0.push(read_ring64(bytes, &mut cursor)?);
    }
    if cursor != bytes.len() {
        return Err("Orbweaver opening proof had trailing bytes".to_string());
    }
    Ok(OrbweaverOpeningProof { pi0 })
}

pub fn encode_terminal_value_proof(proof: &OrbweaverTerminalValueProof) -> Vec<u8> {
    let mut out = Vec::new();
    for limb in
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&proof.opened_value)
    {
        write_sp1_field(&mut out, limb);
    }
    for opening in &proof.limb_proofs {
        let encoded = encode_opening_proof(opening);
        out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        out.extend_from_slice(&encoded);
    }
    out
}

pub fn encode_scalar_opening_proof(proof: &OrbweaverScalarOpeningProof) -> Vec<u8> {
    let mut out = Vec::new();
    write_sp1_field(&mut out, &proof.opened_value);
    out.extend_from_slice(&(proof.pi0.len() as u32).to_le_bytes());
    for ring in &proof.pi0 {
        write_ring64(&mut out, ring);
    }
    out
}

pub fn decode_terminal_value_proof(bytes: &[u8]) -> Result<OrbweaverTerminalValueProof, String> {
    let mut cursor = 0usize;
    let mut limbs = [SP1Field::zero(); 4];
    for limb in &mut limbs {
        *limb = read_sp1_field(bytes, &mut cursor)?;
    }
    let mut limb_proofs = Vec::with_capacity(4);
    for _ in 0..4 {
        let proof_len = read_u32(bytes, &mut cursor)? as usize;
        let end = cursor
            .checked_add(proof_len)
            .ok_or_else(|| "Orbweaver terminal proof length overflow".to_string())?;
        let slice = bytes
            .get(cursor..end)
            .ok_or_else(|| "Orbweaver terminal proof truncated".to_string())?;
        limb_proofs.push(decode_opening_proof(slice)?);
        cursor = end;
    }
    if cursor != bytes.len() {
        return Err("Orbweaver terminal proof had trailing bytes".to_string());
    }
    Ok(OrbweaverTerminalValueProof {
        opened_value: SP1ExtensionField::from_base_slice(&limbs),
        limb_proofs: limb_proofs.try_into().expect("4 limb openings"),
    })
}

pub fn decode_scalar_opening_proof(bytes: &[u8]) -> Result<OrbweaverScalarOpeningProof, String> {
    let mut cursor = 0usize;
    let opened_value = read_sp1_field(bytes, &mut cursor)?;
    let len = read_u32(bytes, &mut cursor)? as usize;
    let mut pi0 = Vec::with_capacity(len);
    for _ in 0..len {
        pi0.push(read_ring64(bytes, &mut cursor)?);
    }
    if cursor != bytes.len() {
        return Err("Orbweaver scalar opening proof had trailing bytes".to_string());
    }
    Ok(OrbweaverScalarOpeningProof { opened_value, pi0 })
}

pub fn encode_extension_opening_proof(proof: &OrbweaverExtensionOpeningProof) -> Vec<u8> {
    let mut out = Vec::new();
    write_extension_value(&mut out, &proof.opened_value);
    out.extend_from_slice(&(proof.pi0.len() as u32).to_le_bytes());
    for value in &proof.pi0 {
        write_extension_value(&mut out, value);
    }
    out
}

pub fn decode_extension_opening_proof(
    bytes: &[u8],
) -> Result<OrbweaverExtensionOpeningProof, String> {
    let mut cursor = 0usize;
    let opened_value = read_extension_value(bytes, &mut cursor)?;
    let len = read_u32(bytes, &mut cursor)? as usize;
    let mut pi0 = Vec::with_capacity(len);
    for _ in 0..len {
        pi0.push(read_extension_value(bytes, &mut cursor)?);
    }
    if cursor != bytes.len() {
        return Err("Orbweaver extension opening proof had trailing bytes".to_string());
    }
    Ok(OrbweaverExtensionOpeningProof { opened_value, pi0 })
}

pub fn evaluate_dense(witness: &[SP1Field], coeffs: &[SP1Field]) -> Result<SP1Field, String> {
    if witness.len() != coeffs.len() {
        return Err(format!(
            "Orbweaver dense evaluation length mismatch: witness={} coeffs={}",
            witness.len(),
            coeffs.len()
        ));
    }
    let mut acc = SP1Field::zero();
    for (x, f) in witness.iter().zip(coeffs.iter()) {
        acc += *x * *f;
    }
    Ok(acc)
}

pub fn scalar_ring_element_to_base_field(ring: &KoalaRing64) -> Result<SP1Field, String> {
    let coeffs = ring.coeffs();
    if coeffs.iter().skip(1).any(|coeff| !coeff.is_zero()) {
        return Err("Orbweaver scalarization requires scalar-subring ring elements".to_string());
    }
    Ok(coeffs[0])
}

pub fn commit_extension_stream(
    alpha_powers: &[SP1ExtensionField],
    witness: &[SP1ExtensionField],
) -> Result<SP1ExtensionField, String> {
    if alpha_powers.len() != witness.len() {
        return Err(format!(
            "Orbweaver extension commitment length mismatch: witness={} powers={}",
            witness.len(),
            alpha_powers.len()
        ));
    }
    let mut acc = SP1ExtensionField::zero();
    for (power, value) in alpha_powers.iter().zip(witness.iter()) {
        acc += *power * *value;
    }
    Ok(acc)
}

pub fn preverify_dense_extension(
    alpha_inv_powers: &[SP1ExtensionField],
    coeffs: &[SP1ExtensionField],
) -> Result<SP1ExtensionField, String> {
    if alpha_inv_powers.len() != coeffs.len() {
        return Err(format!(
            "Orbweaver extension preverify length mismatch: coeffs={} powers={}",
            coeffs.len(),
            alpha_inv_powers.len()
        ));
    }
    let mut acc = SP1ExtensionField::zero();
    for (power, coeff) in alpha_inv_powers.iter().zip(coeffs.iter()) {
        acc += *power * *coeff;
    }
    Ok(acc)
}

pub fn preverify_factorized_mle_extension(
    challenges: &[SP1ExtensionField],
    alpha_inv_power_ladder: &[SP1ExtensionField],
) -> Result<SP1ExtensionField, String> {
    if challenges.len() != alpha_inv_power_ladder.len() {
        return Err(format!(
            "Orbweaver extension factorized preverify length mismatch: challenges={} ladder={}",
            challenges.len(),
            alpha_inv_power_ladder.len()
        ));
    }
    let mut acc = SP1ExtensionField::one();
    for (r_k, alpha_pow) in challenges.iter().zip(alpha_inv_power_ladder.iter()) {
        acc *= SP1ExtensionField::one() + (*r_k * (*alpha_pow - SP1ExtensionField::one()));
    }
    Ok(acc)
}

pub fn open_extension_stream(
    srs: &OrbweaverExtensionOpeningSrs,
    witness: &[SP1ExtensionField],
    coeffs: &[SP1ExtensionField],
) -> Result<OrbweaverExtensionOpeningProof, String> {
    if witness.len() != coeffs.len() {
        return Err(format!(
            "Orbweaver extension opening length mismatch: witness={} coeffs={}",
            witness.len(),
            coeffs.len()
        ));
    }
    let width = witness.len();
    if srs.u0_positive.len() <= width.saturating_sub(1)
        || srs.u0_negative.len() <= width.saturating_sub(1)
    {
        return Err(format!(
            "Orbweaver extension SRS too short for width {}: pos={} neg={}",
            width,
            srs.u0_positive.len(),
            srs.u0_negative.len()
        ));
    }
    let proof_len = srs.a0.len();
    let mut pi0 = vec![SP1ExtensionField::zero(); proof_len];
    let mut opened_value = SP1ExtensionField::zero();
    let mut positive_coeffs = vec![SP1ExtensionField::zero(); width];
    let mut negative_coeffs = vec![SP1ExtensionField::zero(); width];
    for (i, x_i) in witness.iter().enumerate() {
        for (j, f_j) in coeffs.iter().enumerate() {
            let term = *x_i * *f_j;
            if i == j {
                opened_value += term;
            } else if i > j {
                positive_coeffs[i - j] += term;
            } else {
                negative_coeffs[j - i] += term;
            }
        }
    }
    for k in 1..width {
        add_scaled_extension_vector(
            pi0.as_mut_slice(),
            positive_coeffs[k],
            srs.u0_positive[k].as_slice(),
        )?;
        add_scaled_extension_vector(
            pi0.as_mut_slice(),
            negative_coeffs[k],
            srs.u0_negative[k].as_slice(),
        )?;
    }
    Ok(OrbweaverExtensionOpeningProof { opened_value, pi0 })
}

pub fn verify_opening_equation_extension(
    a0: &[SP1ExtensionField],
    proof: &OrbweaverExtensionOpeningProof,
    vk_f: &SP1ExtensionField,
    commitment_c: &SP1ExtensionField,
) -> Result<(), String> {
    if a0.len() != proof.pi0.len() {
        return Err(format!(
            "Orbweaver extension opening equation length mismatch: a0={} pi0={}",
            a0.len(),
            proof.pi0.len()
        ));
    }
    let mut lhs = SP1ExtensionField::zero();
    for (a, p) in a0.iter().zip(proof.pi0.iter()) {
        lhs += *a * *p;
    }
    let rhs = (*vk_f * *commitment_c) - proof.opened_value;
    if lhs != rhs {
        return Err("Orbweaver extension opening equation failed".to_string());
    }
    Ok(())
}

#[must_use]
pub fn extension_opening_proof_l_inf_u32(proof: &OrbweaverExtensionOpeningProof) -> u32 {
    proof
        .pi0
        .iter()
        .flat_map(|value| {
            <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value)
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .map(centered_sp1_abs_u32)
        .max()
        .unwrap_or(0)
}

#[must_use]
pub fn extension_terminal_openings_l_inf_u32(
    openings: &Sp1MulTerminalOpeningProofs,
) -> Result<u32, String> {
    let scalar_norm = |bytes: &[u8]| -> Result<u32, String> {
        Ok(decode_scalar_opening_proof(bytes)?
            .pi0
            .iter()
            .map(KoalaRing64::l_inf_norm_u32)
            .max()
            .unwrap_or(0))
    };
    let extension_norm = |bytes: &[u8]| -> Result<u32, String> {
        Ok(extension_opening_proof_l_inf_u32(&decode_extension_opening_proof(bytes)?))
    };
    let ring_norm = |bytes: &[u8]| -> Result<u32, String> {
        Ok(decode_terminal_value_proof(bytes)?
            .limb_proofs
            .iter()
            .map(opening_proof_l_inf_norm_u32)
            .max()
            .unwrap_or(0))
    };
    let limb_max = |bytes: &[u8]| match scalar_norm(bytes) {
        Ok(v) => Ok(v),
        Err(_) => match extension_norm(bytes) {
            Ok(v) => Ok(v),
            Err(_) => ring_norm(bytes),
        },
    };
    Ok([
        limb_max(openings.a.as_slice())?,
        limb_max(openings.b.as_slice())?,
        limb_max(openings.c.as_slice())?,
        limb_max(openings.d.as_slice())?,
    ]
    .into_iter()
    .max()
    .unwrap_or(0))
}

pub fn commit_extension_terminal_stream(
    terms: &[Sp1MulTerm],
    alpha_powers: &[SP1ExtensionField],
    field: OrbweaverTerminalField,
) -> Result<SP1ExtensionField, String> {
    let witness = extract_terminal_extension_stream(terms, field);
    commit_extension_stream(alpha_powers, witness.as_slice())
}

pub fn build_extension_terminal_openings_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
) -> Result<Sp1MulTerminalOpeningProofs, String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver extension terminal opening weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let ext_srs = scalarize_srs_to_extension(srs, terms.len())?;
    let alpha_powers = derive_extension_alpha_positive_powers(&ext_srs, terms.len())?;
    let proof_a = open_extension_stream(
        &ext_srs,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::A).as_slice(),
        weights,
    )?;
    let proof_b = open_extension_stream(
        &ext_srs,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::B).as_slice(),
        weights,
    )?;
    let proof_c = open_extension_stream(
        &ext_srs,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::C).as_slice(),
        weights,
    )?;
    let proof_d = open_extension_stream(
        &ext_srs,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::D).as_slice(),
        weights,
    )?;
    let expected_a = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::A).as_slice(),
    )?;
    let expected_b = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::B).as_slice(),
    )?;
    let expected_c = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::C).as_slice(),
    )?;
    let expected_d = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::D).as_slice(),
    )?;
    if proof_a.opened_value != expected_a {
        return Err("Orbweaver extension A opening value mismatch".to_string());
    }
    if proof_b.opened_value != expected_b {
        return Err("Orbweaver extension B opening value mismatch".to_string());
    }
    if proof_c.opened_value != expected_c {
        return Err("Orbweaver extension C opening value mismatch".to_string());
    }
    if proof_d.opened_value != expected_d {
        return Err("Orbweaver extension D opening value mismatch".to_string());
    }
    let _ = alpha_powers;
    Ok(Sp1MulTerminalOpeningProofs {
        a: encode_extension_opening_proof(&proof_a),
        b: encode_extension_opening_proof(&proof_b),
        c: encode_extension_opening_proof(&proof_c),
        d: encode_extension_opening_proof(&proof_d),
    })
}

pub fn build_terminal_value_openings_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
) -> Result<Sp1MulTerminalOpeningProofs, String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver terminal opening weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let proof_a =
        open_terminal_value_from_mul_terms(srs, terms, weights, OrbweaverTerminalField::A)?;
    let proof_b =
        open_terminal_value_from_mul_terms(srs, terms, weights, OrbweaverTerminalField::B)?;
    let proof_c =
        open_terminal_value_from_mul_terms(srs, terms, weights, OrbweaverTerminalField::C)?;
    let proof_d =
        open_terminal_value_from_mul_terms(srs, terms, weights, OrbweaverTerminalField::D)?;

    let expected_a = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::A).as_slice(),
    )?;
    let expected_b = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::B).as_slice(),
    )?;
    let expected_c = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::C).as_slice(),
    )?;
    let expected_d = evaluate_extension_stream(
        weights,
        extract_terminal_extension_stream(terms, OrbweaverTerminalField::D).as_slice(),
    )?;
    if proof_a.opened_value != expected_a {
        return Err("Orbweaver terminal A opening value mismatch".to_string());
    }
    if proof_b.opened_value != expected_b {
        return Err("Orbweaver terminal B opening value mismatch".to_string());
    }
    if proof_c.opened_value != expected_c {
        return Err("Orbweaver terminal C opening value mismatch".to_string());
    }
    if proof_d.opened_value != expected_d {
        return Err("Orbweaver terminal D opening value mismatch".to_string());
    }
    Ok(Sp1MulTerminalOpeningProofs {
        a: encode_terminal_value_proof(&proof_a),
        b: encode_terminal_value_proof(&proof_b),
        c: encode_terminal_value_proof(&proof_c),
        d: encode_terminal_value_proof(&proof_d),
    })
}

pub fn terminal_scalar_image_forms(weights: &[SP1ExtensionField]) -> [Vec<SP1Field>; 16] {
    let a_forms = terminal_field_base_forms(weights, OrbweaverTerminalField::A);
    let b_forms = terminal_field_base_forms(weights, OrbweaverTerminalField::B);
    let c_forms = terminal_field_base_forms(weights, OrbweaverTerminalField::C);
    let d_forms = terminal_field_base_forms(weights, OrbweaverTerminalField::D);
    [
        a_forms[0].clone(),
        a_forms[1].clone(),
        a_forms[2].clone(),
        a_forms[3].clone(),
        b_forms[0].clone(),
        b_forms[1].clone(),
        b_forms[2].clone(),
        b_forms[3].clone(),
        c_forms[0].clone(),
        c_forms[1].clone(),
        c_forms[2].clone(),
        c_forms[3].clone(),
        d_forms[0].clone(),
        d_forms[1].clone(),
        d_forms[2].clone(),
        d_forms[3].clone(),
    ]
}

pub fn terminal_scalar_image_values(values: &[SP1ExtensionField; 4]) -> [SP1Field; 16] {
    let a =
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&values[0]);
    let b =
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&values[1]);
    let c =
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&values[2]);
    let d =
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&values[3]);
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2],
        d[3],
    ]
}

pub fn aggregate_scalar_image_form(
    forms: &[Vec<SP1Field>; 16],
    aggregation_coeffs: &[SP1Field; 16],
) -> Result<Vec<SP1Field>, String> {
    let len = forms[0].len();
    if forms.iter().any(|form| form.len() != len) {
        return Err("Orbweaver scalar image form length mismatch".to_string());
    }
    let mut out = vec![SP1Field::zero(); len];
    for (coeff, form) in aggregation_coeffs.iter().zip(forms.iter()) {
        for (dst, src) in out.iter_mut().zip(form.iter()) {
            *dst += *coeff * *src;
        }
    }
    Ok(out)
}

pub fn aggregate_scalar_image_value(
    scalar_values: &[SP1Field; 16],
    aggregation_coeffs: &[SP1Field; 16],
) -> SP1Field {
    scalar_values
        .iter()
        .zip(aggregation_coeffs.iter())
        .map(|(value, coeff)| *value * *coeff)
        .sum()
}

pub fn build_aggregated_scalar_image_openings_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
    aggregation_coeffs: &[[SP1Field; 16]; 4],
) -> Result<Sp1MulTerminalOpeningProofs, String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver aggregated scalar opening weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let witness = flatten_mul_terms_to_base_witness(terms);
    let forms = terminal_scalar_image_forms(weights);
    let mut encoded = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (slot, coeffs) in encoded.iter_mut().zip(aggregation_coeffs.iter()) {
        let aggregated_form = aggregate_scalar_image_form(&forms, coeffs)?;
        let (opened_value, proof) = open_dense(srs, witness.as_slice(), aggregated_form.as_slice())?;
        *slot = encode_scalar_opening_proof(&OrbweaverScalarOpeningProof { opened_value, pi0: proof.pi0 });
    }
    Ok(Sp1MulTerminalOpeningProofs {
        a: encoded[0].clone(),
        b: encoded[1].clone(),
        c: encoded[2].clone(),
        d: encoded[3].clone(),
    })
}

pub fn verify_aggregated_scalar_image_openings_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
    aggregation_coeffs: &[[SP1Field; 16]; 4],
    expected_scalar_values: &[SP1Field; 16],
    openings: &Sp1MulTerminalOpeningProofs,
) -> Result<(), String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver aggregated scalar verification weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let witness = flatten_mul_terms_to_base_witness(terms);
    let commitment_c = commit_mul_terms_witness(srs, terms)?;
    let v_inv_powers = derive_negative_powers(srs, witness.len())?;
    let forms = terminal_scalar_image_forms(weights);
    let encoded = [
        openings.a.as_slice(),
        openings.b.as_slice(),
        openings.c.as_slice(),
        openings.d.as_slice(),
    ];
    for (coeffs, bytes) in aggregation_coeffs.iter().zip(encoded.iter()) {
        let aggregated_form = aggregate_scalar_image_form(&forms, coeffs)?;
        let expected_value = aggregate_scalar_image_value(expected_scalar_values, coeffs);
        let proof = decode_scalar_opening_proof(bytes)?;
        if proof.opened_value != expected_value {
            return Err("Orbweaver aggregated scalar opened value mismatch".to_string());
        }
        let vk = preverify_dense(v_inv_powers.as_slice(), aggregated_form.as_slice())?;
        verify_opening_equation(
            &srs.a0,
            &OrbweaverOpeningProof { pi0: proof.pi0.clone() },
            &vk,
            &commitment_c,
            &KoalaRing64::from_scalar(proof.opened_value),
        )?;
    }
    Ok(())
}

pub fn verify_extension_terminal_openings_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
    round_challenges: &[SP1ExtensionField],
    openings: &Sp1MulTerminalOpeningProofs,
) -> Result<(), String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver extension terminal verification weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let ext_srs = scalarize_srs_to_extension(srs, terms.len())?;
    let alpha_powers = derive_extension_alpha_positive_powers(&ext_srs, terms.len())?;
    let alpha_inv_ladder = derive_extension_alpha_inv_power_ladder(
        &ext_srs,
        mul_sumcheck_rounds_from_len(terms.len())?,
    )?;
    verify_one_extension_terminal_opening(
        &ext_srs,
        &alpha_powers,
        &alpha_inv_ladder,
        weights,
        round_challenges,
        terms,
        OrbweaverTerminalField::A,
        openings.a.as_slice(),
    )?;
    verify_one_extension_terminal_opening(
        &ext_srs,
        &alpha_powers,
        &alpha_inv_ladder,
        weights,
        round_challenges,
        terms,
        OrbweaverTerminalField::B,
        openings.b.as_slice(),
    )?;
    verify_one_extension_terminal_opening(
        &ext_srs,
        &alpha_powers,
        &alpha_inv_ladder,
        weights,
        round_challenges,
        terms,
        OrbweaverTerminalField::C,
        openings.c.as_slice(),
    )?;
    verify_one_extension_terminal_opening(
        &ext_srs,
        &alpha_powers,
        &alpha_inv_ladder,
        weights,
        round_challenges,
        terms,
        OrbweaverTerminalField::D,
        openings.d.as_slice(),
    )?;
    Ok(())
}

pub fn verify_terminal_value_openings_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
    openings: &Sp1MulTerminalOpeningProofs,
) -> Result<(), String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver terminal verification weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let commitment_c = commit_mul_terms_witness(srs, terms)?;
    verify_one_terminal_value_opening(
        srs,
        terms,
        weights,
        &commitment_c,
        OrbweaverTerminalField::A,
        openings.a.as_slice(),
    )?;
    verify_one_terminal_value_opening(
        srs,
        terms,
        weights,
        &commitment_c,
        OrbweaverTerminalField::B,
        openings.b.as_slice(),
    )?;
    verify_one_terminal_value_opening(
        srs,
        terms,
        weights,
        &commitment_c,
        OrbweaverTerminalField::C,
        openings.c.as_slice(),
    )?;
    verify_one_terminal_value_opening(
        srs,
        terms,
        weights,
        &commitment_c,
        OrbweaverTerminalField::D,
        openings.d.as_slice(),
    )?;
    Ok(())
}

pub fn derive_positive_powers(
    srs: &OrbweaverOpeningSrs,
    width: usize,
) -> Result<Vec<KoalaRing64>, String> {
    if srs.u0_positive.len() <= width {
        return Err(format!(
            "Orbweaver SRS missing positive powers: need={} have={}",
            width + 1,
            srs.u0_positive.len()
        ));
    }
    (1..=width).map(|idx| KoalaRing64::dot(&srs.a0, srs.u0_positive[idx].as_slice())).collect()
}

pub fn derive_negative_powers(
    srs: &OrbweaverOpeningSrs,
    width: usize,
) -> Result<Vec<KoalaRing64>, String> {
    if srs.u0_negative.len() <= width {
        return Err(format!(
            "Orbweaver SRS missing negative powers: need={} have={}",
            width + 1,
            srs.u0_negative.len()
        ));
    }
    (1..=width).map(|idx| KoalaRing64::dot(&srs.a0, srs.u0_negative[idx].as_slice())).collect()
}

pub fn commit_dense(v_powers: &[KoalaRing64], witness: &[SP1Field]) -> Result<KoalaRing64, String> {
    if v_powers.len() != witness.len() {
        return Err(format!(
            "Orbweaver dense commitment length mismatch: witness={} powers={}",
            witness.len(),
            v_powers.len()
        ));
    }
    let commitment = v_powers.iter().zip(witness.iter()).map(|(power, x)| *power * *x).sum();
    Ok(commitment)
}

pub fn preverify_dense(
    v_inv_powers: &[KoalaRing64],
    coeffs: &[SP1Field],
) -> Result<OrbweaverOpeningVerificationKey, String> {
    if v_inv_powers.len() != coeffs.len() {
        return Err(format!(
            "Orbweaver dense preverify length mismatch: coeffs={} powers={}",
            coeffs.len(),
            v_inv_powers.len()
        ));
    }
    let value = v_inv_powers.iter().zip(coeffs.iter()).map(|(power, coeff)| *power * *coeff).sum();
    Ok(OrbweaverOpeningVerificationKey { value })
}

/// Compute `vk_f = Π_k ((1-r_k) + r_k * v^{-2^k})` for the multilinear-evaluation weights.
pub fn preverify_factorized_mle(
    challenges: &[SP1Field],
    v_inv_power_ladder: &[KoalaRing64],
) -> Result<OrbweaverOpeningVerificationKey, String> {
    if challenges.len() != v_inv_power_ladder.len() {
        return Err(format!(
            "Orbweaver factorized preverify length mismatch: challenges={} ladder={}",
            challenges.len(),
            v_inv_power_ladder.len()
        ));
    }
    let mut acc = KoalaRing64::one();
    for (r_k, v_k) in challenges.iter().zip(v_inv_power_ladder.iter()) {
        let one_minus_r = SP1Field::one() - *r_k;
        let factor = KoalaRing64::from_scalar(one_minus_r) + (*v_k * *r_k);
        acc *= factor;
    }
    Ok(OrbweaverOpeningVerificationKey { value: acc })
}

pub fn open_dense(
    srs: &OrbweaverOpeningSrs,
    witness: &[SP1Field],
    coeffs: &[SP1Field],
) -> Result<(SP1Field, OrbweaverOpeningProof), String> {
    if witness.len() != coeffs.len() {
        return Err(format!(
            "Orbweaver dense opening length mismatch: witness={} coeffs={}",
            witness.len(),
            coeffs.len()
        ));
    }
    let width = witness.len();
    if srs.u0_positive.len() <= width.saturating_sub(1)
        || srs.u0_negative.len() <= width.saturating_sub(1)
    {
        return Err(format!(
            "Orbweaver SRS too short for width {}: pos={} neg={}",
            width,
            srs.u0_positive.len(),
            srs.u0_negative.len()
        ));
    }
    let proof_len = srs.a0.len();
    let mut pi0 = vec![KoalaRing64::zero(); proof_len];
    let mut opened_value = SP1Field::zero();
    let mut positive_coeffs = vec![SP1Field::zero(); width];
    let mut negative_coeffs = vec![SP1Field::zero(); width];
    for (i, x_i) in witness.iter().enumerate() {
        for (j, f_j) in coeffs.iter().enumerate() {
            let term = *x_i * *f_j;
            if i == j {
                opened_value += term;
            } else if i > j {
                positive_coeffs[i - j] += term;
            } else {
                negative_coeffs[j - i] += term;
            }
        }
    }
    if let Some(short_pi0) =
        local_dev_short_opening_from_coeff_buckets(srs, positive_coeffs.as_slice(), negative_coeffs.as_slice())?
    {
        if short_pi0.len() != pi0.len() {
            return Err(format!(
                "Orbweaver local-dev short opening length mismatch: got={} expected={}",
                short_pi0.len(),
                pi0.len()
            ));
        }
        pi0 = short_pi0;
    } else {
        for k in 1..width {
            add_scaled_ring_vector(
                pi0.as_mut_slice(),
                positive_coeffs[k],
                srs.u0_positive[k].as_slice(),
            )?;
            add_scaled_ring_vector(
                pi0.as_mut_slice(),
                negative_coeffs[k],
                srs.u0_negative[k].as_slice(),
            )?;
        }
    }
    Ok((opened_value, OrbweaverOpeningProof { pi0 }))
}

pub fn flatten_mul_terms_to_base_witness(terms: &[Sp1MulTerm]) -> Vec<SP1Field> {
    let mut out = Vec::with_capacity(terms.len() * 16);
    for term in terms {
        out.extend(flatten_extension_limbs(core::slice::from_ref(&term.a)));
        out.extend(flatten_extension_limbs(core::slice::from_ref(&term.b)));
        out.extend(flatten_extension_limbs(core::slice::from_ref(&term.c)));
        out.extend(flatten_extension_limbs(core::slice::from_ref(&term.d)));
    }
    out
}

pub fn terminal_field_base_forms(
    weights: &[SP1ExtensionField],
    field: OrbweaverTerminalField,
) -> [Vec<SP1Field>; 4] {
    let field_forms = linearize_extension_weights_to_base_forms(weights);
    let field_offset = match field {
        OrbweaverTerminalField::A => 0usize,
        OrbweaverTerminalField::B => 1usize,
        OrbweaverTerminalField::C => 2usize,
        OrbweaverTerminalField::D => 3usize,
    };
    let mut out = core::array::from_fn(|_| vec![SP1Field::zero(); weights.len() * 16]);
    for term_idx in 0..weights.len() {
        for row in 0..4 {
            let src = &field_forms[row][term_idx * 4..(term_idx + 1) * 4];
            let dst_offset = term_idx * 16 + field_offset * 4;
            out[row][dst_offset..dst_offset + 4].copy_from_slice(src);
        }
    }
    out
}

pub fn commit_mul_terms_witness(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
) -> Result<KoalaRing64, String> {
    let witness = flatten_mul_terms_to_base_witness(terms);
    let v_powers = derive_positive_powers(srs, witness.len())?;
    commit_dense(&v_powers, &witness)
}

pub fn open_terminal_value_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
    field: OrbweaverTerminalField,
) -> Result<OrbweaverTerminalValueProof, String> {
    if terms.len() != weights.len() {
        return Err(format!(
            "Orbweaver terminal opening weight mismatch: terms={} weights={}",
            terms.len(),
            weights.len()
        ));
    }
    let witness = flatten_mul_terms_to_base_witness(terms);
    let forms = terminal_field_base_forms(weights, field);
    let mut opened_limbs = [SP1Field::zero(); 4];
    let mut limb_proofs = Vec::with_capacity(4);
    for row in 0..4 {
        let (limb, proof) = open_dense(srs, &witness, forms[row].as_slice())?;
        opened_limbs[row] = limb;
        limb_proofs.push(proof);
    }
    Ok(OrbweaverTerminalValueProof {
        opened_value: SP1ExtensionField::from_base_slice(&opened_limbs),
        limb_proofs: limb_proofs.try_into().expect("4 limb proofs"),
    })
}

pub fn verify_terminal_value_from_mul_terms(
    srs: &OrbweaverOpeningSrs,
    terms_len: usize,
    commitment_c: &KoalaRing64,
    weights: &[SP1ExtensionField],
    field: OrbweaverTerminalField,
    proof: &OrbweaverTerminalValueProof,
) -> Result<(), String> {
    if terms_len != weights.len() {
        return Err(format!(
            "Orbweaver terminal verification weight mismatch: terms={} weights={}",
            terms_len,
            weights.len()
        ));
    }
    let witness_width =
        terms_len.checked_mul(16).ok_or_else(|| "Orbweaver witness width overflow".to_string())?;
    let v_inv_powers = derive_negative_powers(srs, witness_width)?;
    let forms = terminal_field_base_forms(weights, field);
    let opened_limbs =
        <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(&proof.opened_value);
    for row in 0..4 {
        let vk = preverify_dense(&v_inv_powers, forms[row].as_slice())?;
        verify_opening_equation(
            &srs.a0,
            &proof.limb_proofs[row],
            &vk,
            commitment_c,
            &KoalaRing64::from_scalar(opened_limbs[row]),
        )?;
    }
    Ok(())
}

pub fn verify_opening_equation(
    a0: &[KoalaRing64],
    proof: &OrbweaverOpeningProof,
    vk_f: &OrbweaverOpeningVerificationKey,
    commitment_c: &KoalaRing64,
    value_y: &KoalaRing64,
) -> Result<(), String> {
    if a0.len() != proof.pi0.len() {
        return Err(format!(
            "Orbweaver opening equation length mismatch: a0={} pi0={}",
            a0.len(),
            proof.pi0.len()
        ));
    }
    let lhs = KoalaRing64::dot(a0, proof.pi0.as_slice())?;
    let rhs = (vk_f.value * *commitment_c) - *value_y;
    if lhs != rhs {
        return Err("Orbweaver opening equation failed".to_string());
    }
    Ok(())
}

#[must_use]
pub fn opening_proof_l_inf_norm_u32(proof: &OrbweaverOpeningProof) -> u32 {
    proof.pi0.iter().map(KoalaRing64::l_inf_norm_u32).max().unwrap_or(0)
}

fn add_scaled_ring_vector(
    acc: &mut [KoalaRing64],
    scale: SP1Field,
    vector: &[KoalaRing64],
) -> Result<(), String> {
    if acc.len() != vector.len() {
        return Err(format!(
            "Orbweaver ring vector length mismatch: acc={} vector={}",
            acc.len(),
            vector.len()
        ));
    }
    for (dst, src) in acc.iter_mut().zip(vector.iter()) {
        *dst += *src * scale;
    }
    Ok(())
}

fn add_scaled_extension_vector(
    acc: &mut [SP1ExtensionField],
    scale: SP1ExtensionField,
    vector: &[SP1ExtensionField],
) -> Result<(), String> {
    if acc.len() != vector.len() {
        return Err(format!(
            "Orbweaver extension vector length mismatch: acc={} vector={}",
            acc.len(),
            vector.len()
        ));
    }
    for (dst, src) in acc.iter_mut().zip(vector.iter()) {
        *dst += *src * scale;
    }
    Ok(())
}

fn scalar_ring_element_to_extension(ring: &KoalaRing64) -> Result<SP1ExtensionField, String> {
    Ok(SP1ExtensionField::from_base_slice(&[
        scalar_ring_element_to_base_field(ring)?,
        SP1Field::zero(),
        SP1Field::zero(),
        SP1Field::zero(),
    ]))
}

fn write_extension_value(out: &mut Vec<u8>, value: &SP1ExtensionField) {
    for limb in <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value) {
        write_sp1_field(out, limb);
    }
}

fn read_extension_value(bytes: &[u8], cursor: &mut usize) -> Result<SP1ExtensionField, String> {
    let limbs = [
        read_sp1_field(bytes, cursor)?,
        read_sp1_field(bytes, cursor)?,
        read_sp1_field(bytes, cursor)?,
        read_sp1_field(bytes, cursor)?,
    ];
    Ok(SP1ExtensionField::from_base_slice(&limbs))
}

fn derive_extension_alpha_positive_powers(
    srs: &OrbweaverExtensionOpeningSrs,
    width: usize,
) -> Result<Vec<SP1ExtensionField>, String> {
    if width > srs.u0_positive.len() {
        return Err(format!(
            "Orbweaver extension positive powers unsupported: requested={} available={}",
            width,
            srs.u0_positive.len().saturating_sub(1)
        ));
    }
    let mut out = Vec::with_capacity(width);
    let mut cur = SP1ExtensionField::one();
    for _ in 0..width {
        out.push(cur);
        cur *= srs.alpha;
    }
    Ok(out)
}

fn derive_extension_alpha_inv_power_ladder(
    srs: &OrbweaverExtensionOpeningSrs,
    rounds: usize,
) -> Result<Vec<SP1ExtensionField>, String> {
    let alpha_inv = srs
        .alpha
        .try_inverse()
        .ok_or_else(|| "Orbweaver extension alpha was not invertible".to_string())?;
    let mut out = Vec::with_capacity(rounds);
    let mut cur = alpha_inv;
    for _ in 0..rounds {
        out.push(cur);
        cur *= cur;
    }
    Ok(out)
}

fn extract_terminal_extension_stream(
    terms: &[Sp1MulTerm],
    field: OrbweaverTerminalField,
) -> Vec<SP1ExtensionField> {
    terms
        .iter()
        .map(|term| match field {
            OrbweaverTerminalField::A => term.a,
            OrbweaverTerminalField::B => term.b,
            OrbweaverTerminalField::C => term.c,
            OrbweaverTerminalField::D => term.d,
        })
        .collect()
}

fn verify_one_extension_terminal_opening(
    srs: &OrbweaverExtensionOpeningSrs,
    alpha_powers: &[SP1ExtensionField],
    alpha_inv_ladder: &[SP1ExtensionField],
    weights: &[SP1ExtensionField],
    round_challenges: &[SP1ExtensionField],
    terms: &[Sp1MulTerm],
    field: OrbweaverTerminalField,
    encoded: &[u8],
) -> Result<(), String> {
    let proof = decode_extension_opening_proof(encoded)?;
    let expected_stream = extract_terminal_extension_stream(terms, field);
    let expected_commitment = commit_extension_stream(alpha_powers, expected_stream.as_slice())?;
    let vk = preverify_factorized_mle_extension(round_challenges, alpha_inv_ladder)?;
    if proof.opened_value != evaluate_extension_stream(weights, expected_stream.as_slice())? {
        return Err("Orbweaver extension opened value mismatch".to_string());
    }
    verify_opening_equation_extension(&srs.a0, &proof, &vk, &expected_commitment)
}

fn verify_one_terminal_value_opening(
    srs: &OrbweaverOpeningSrs,
    terms: &[Sp1MulTerm],
    weights: &[SP1ExtensionField],
    commitment_c: &KoalaRing64,
    field: OrbweaverTerminalField,
    encoded: &[u8],
) -> Result<(), String> {
    let proof = decode_terminal_value_proof(encoded)?;
    verify_terminal_value_from_mul_terms(srs, terms.len(), commitment_c, weights, field, &proof)?;
    let expected_stream = extract_terminal_extension_stream(terms, field);
    if proof.opened_value != evaluate_extension_stream(weights, expected_stream.as_slice())? {
        return Err("Orbweaver terminal opened value mismatch".to_string());
    }
    Ok(())
}

fn evaluate_extension_stream(
    coeffs: &[SP1ExtensionField],
    witness: &[SP1ExtensionField],
) -> Result<SP1ExtensionField, String> {
    if coeffs.len() != witness.len() {
        return Err(format!(
            "Orbweaver extension evaluation length mismatch: coeffs={} witness={}",
            coeffs.len(),
            witness.len()
        ));
    }
    let mut acc = SP1ExtensionField::zero();
    for (c, w) in coeffs.iter().zip(witness.iter()) {
        acc += *c * *w;
    }
    Ok(acc)
}

fn mul_sumcheck_rounds_from_len(len: usize) -> Result<usize, String> {
    if len == 0 {
        return Err("Orbweaver extension path requires non-empty stream".to_string());
    }
    Ok(len.next_power_of_two().trailing_zeros() as usize)
}

fn encode_ring_vector_family(out: &mut Vec<u8>, family: &[Vec<KoalaRing64>]) {
    out.extend_from_slice(&(family.len() as u32).to_le_bytes());
    for vector in family {
        out.extend_from_slice(&(vector.len() as u32).to_le_bytes());
        for ring in vector {
            write_ring64(out, ring);
        }
    }
}

fn decode_ring_vector_family(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<Vec<Vec<KoalaRing64>>, String> {
    let outer_len = read_u32(bytes, cursor)? as usize;
    let mut family = Vec::with_capacity(outer_len);
    for _ in 0..outer_len {
        let inner_len = read_u32(bytes, cursor)? as usize;
        let mut vector = Vec::with_capacity(inner_len);
        for _ in 0..inner_len {
            vector.push(read_ring64(bytes, cursor)?);
        }
        family.push(vector);
    }
    Ok(family)
}

fn hash_ring_vector_family(h: &mut Sha256, family: &[Vec<KoalaRing64>]) {
    h.update((family.len() as u64).to_le_bytes());
    for vector in family {
        h.update((vector.len() as u64).to_le_bytes());
        for ring in vector {
            hash_ring64(h, ring);
        }
    }
}

fn hash_ring64(h: &mut Sha256, ring: &KoalaRing64) {
    for coeff in ring.coeffs() {
        h.update(coeff.as_canonical_u32().to_le_bytes());
    }
}

fn local_dev_gadget_vector() -> Vec<KoalaRing64> {
    let mut out = Vec::with_capacity(LOCAL_DEV_GADGET_DIGITS);
    let mut power = 1u32;
    for _ in 0..LOCAL_DEV_GADGET_DIGITS {
        out.push(KoalaRing64::from_scalar(SP1Field::from_canonical_u32(power)));
        power = power.saturating_mul(LOCAL_DEV_GADGET_BASE_U32);
    }
    out
}

fn local_dev_short_preimage_for_scalar(value: SP1Field) -> Vec<KoalaRing64> {
    let mut n = value.as_canonical_u32();
    let mut out = Vec::with_capacity(LOCAL_DEV_GADGET_DIGITS);
    for _ in 0..LOCAL_DEV_GADGET_DIGITS {
        let digit = n % LOCAL_DEV_GADGET_BASE_U32;
        out.push(KoalaRing64::from_scalar(SP1Field::from_canonical_u32(digit)));
        n /= LOCAL_DEV_GADGET_BASE_U32;
    }
    out
}

fn local_dev_short_opening_from_coeff_buckets(
    srs: &OrbweaverOpeningSrs,
    positive_coeffs: &[SP1Field],
    negative_coeffs: &[SP1Field],
) -> Result<Option<Vec<KoalaRing64>>, String> {
    if srs.a0 != local_dev_gadget_vector() {
        return Ok(None);
    }
    if positive_coeffs.len() != negative_coeffs.len() {
        return Err(format!(
            "Orbweaver coeff bucket mismatch: pos={} neg={}",
            positive_coeffs.len(),
            negative_coeffs.len()
        ));
    }
    let width = positive_coeffs.len();
    if width <= 1 {
        return Ok(Some(local_dev_short_preimage_for_scalar(SP1Field::zero())));
    }
    let positive_powers = derive_positive_powers(srs, width - 1)?;
    let negative_powers = derive_negative_powers(srs, width - 1)?;
    let mut target = KoalaRing64::zero();
    for k in 1..width {
        target += positive_powers[k - 1] * positive_coeffs[k];
        target += negative_powers[k - 1] * negative_coeffs[k];
    }
    let coeffs = target.coeffs();
    if coeffs.iter().skip(1).any(|coeff| !coeff.is_zero()) {
        return Ok(None);
    }
    Ok(Some(local_dev_short_preimage_for_scalar(coeffs[0])))
}

fn centered_sp1_abs_u32(value: SP1Field) -> u32 {
    let canonical = value.as_canonical_u32();
    let modulus = SP1Field::ORDER_U32;
    canonical.min(modulus - canonical)
}

fn write_sp1_field(out: &mut Vec<u8>, value: &SP1Field) {
    out.extend_from_slice(&value.as_canonical_u32().to_le_bytes());
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let end = cursor.checked_add(4).ok_or_else(|| "Orbweaver byte cursor overflow".to_string())?;
    let chunk = bytes.get(*cursor..end).ok_or_else(|| "Orbweaver proof truncated".to_string())?;
    *cursor = end;
    Ok(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_sp1_field(bytes: &[u8], cursor: &mut usize) -> Result<SP1Field, String> {
    Ok(SP1Field::from_wrapped_u32(read_u32(bytes, cursor)?))
}

fn write_ring64(out: &mut Vec<u8>, ring: &KoalaRing64) {
    for coeff in ring.coeffs() {
        write_sp1_field(out, coeff);
    }
}

fn read_ring64(bytes: &[u8], cursor: &mut usize) -> Result<KoalaRing64, String> {
    let mut coeffs = [SP1Field::zero(); 64];
    for coeff in &mut coeffs {
        *coeff = read_sp1_field(bytes, cursor)?;
    }
    Ok(KoalaRing64::from_coeffs(coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(x: u32) -> SP1Field {
        SP1Field::from_canonical_u32(x)
    }

    #[test]
    fn factorized_preverify_matches_dense_preverify_for_two_bits() {
        let r0 = scalar(7);
        let r1 = scalar(11);
        let v_inv_powers =
            [KoalaRing64::from_scalar(scalar(3)), KoalaRing64::from_scalar(scalar(5))];

        let dense_coeffs = [
            (SP1Field::one() - r0) * (SP1Field::one() - r1),
            r0 * (SP1Field::one() - r1),
            (SP1Field::one() - r0) * r1,
            r0 * r1,
        ];
        let dense_powers = [
            KoalaRing64::one(),
            v_inv_powers[0],
            v_inv_powers[1],
            v_inv_powers[0] * v_inv_powers[1],
        ];

        let dense = preverify_dense(&dense_powers, &dense_coeffs).expect("dense preverify");
        let factored =
            preverify_factorized_mle(&[r0, r1], &v_inv_powers).expect("factorized preverify");
        assert_eq!(dense, factored);
    }

    #[test]
    fn verify_opening_equation_accepts_consistent_inputs() {
        let a0 = vec![KoalaRing64::from_scalar(scalar(2)), KoalaRing64::from_scalar(scalar(3))];
        let pi0 = vec![KoalaRing64::from_scalar(scalar(5)), KoalaRing64::from_scalar(scalar(7))];
        let lhs = KoalaRing64::dot(&a0, &pi0).expect("dot");
        let commitment = KoalaRing64::from_scalar(scalar(13));
        let vk = OrbweaverOpeningVerificationKey { value: KoalaRing64::from_scalar(scalar(17)) };
        let value = (vk.value * commitment) - lhs;

        verify_opening_equation(&a0, &OrbweaverOpeningProof { pi0 }, &vk, &commitment, &value)
            .expect("opening equation");
    }

    #[test]
    fn open_dense_matches_verifier_equation() {
        let inv7 = scalar(7).try_inverse().expect("inverse");
        let inv49 = inv7 * inv7;
        let v = KoalaRing64::from_scalar(scalar(7));
        let v_powers = [v, v * v];
        let v_inv_powers = [KoalaRing64::from_scalar(inv7), KoalaRing64::from_scalar(inv49)];
        let srs = generate_local_dev_srs(2);
        let witness = [scalar(5), scalar(9)];
        let coeffs = [scalar(4), scalar(6)];
        let commitment = commit_dense(&v_powers, &witness).expect("commit");
        let (y, proof) = open_dense(&srs, &witness, &coeffs).expect("open");
        let vk = preverify_dense(&v_inv_powers, &coeffs).expect("preverify");
        verify_opening_equation(&srs.a0, &proof, &vk, &commitment, &KoalaRing64::from_scalar(y))
            .expect("verify opening");
        assert_eq!(y, evaluate_dense(&witness, &coeffs).expect("evaluate"));
    }

    fn ext(a0: u32, a1: u32, a2: u32, a3: u32) -> SP1ExtensionField {
        SP1ExtensionField::from_base_slice(&[scalar(a0), scalar(a1), scalar(a2), scalar(a3)])
    }

    #[test]
    fn terminal_value_opening_roundtrip() {
        let srs = generate_local_dev_srs(2);
        let terms = [
            Sp1MulTerm::new(ext(2, 3, 5, 7), ext(11, 13, 17, 19), ext(0, 0, 0, 0), ext(1, 0, 0, 0)),
            Sp1MulTerm::new(
                ext(23, 29, 31, 37),
                ext(41, 43, 47, 53),
                ext(0, 0, 0, 0),
                ext(1, 0, 0, 0),
            ),
        ];
        let weights = [ext(3, 0, 0, 0), ext(5, 0, 0, 0)];
        let proof =
            open_terminal_value_from_mul_terms(&srs, &terms, &weights, OrbweaverTerminalField::A)
                .expect("open terminal value");
        let commitment = commit_mul_terms_witness(&srs, &terms).expect("commit mul witness");
        verify_terminal_value_from_mul_terms(
            &srs,
            terms.len(),
            &commitment,
            &weights,
            OrbweaverTerminalField::A,
            &proof,
        )
        .expect("verify terminal value");
        assert_eq!(proof.opened_value, (weights[0] * terms[0].a) + (weights[1] * terms[1].a));
    }

    #[test]
    fn terminal_value_proof_roundtrip_encoding() {
        let proof = OrbweaverTerminalValueProof {
            opened_value: ext(1, 2, 3, 4),
            limb_proofs: [
                OrbweaverOpeningProof { pi0: vec![KoalaRing64::from_scalar(scalar(5))] },
                OrbweaverOpeningProof { pi0: vec![KoalaRing64::from_scalar(scalar(6))] },
                OrbweaverOpeningProof { pi0: vec![KoalaRing64::from_scalar(scalar(7))] },
                OrbweaverOpeningProof { pi0: vec![KoalaRing64::from_scalar(scalar(8))] },
            ],
        };
        let encoded = encode_terminal_value_proof(&proof);
        let decoded = decode_terminal_value_proof(&encoded).expect("decode");
        assert_eq!(decoded, proof);
    }

    #[test]
    fn srs_roundtrip_encoding_and_validation() {
        let srs = generate_local_dev_srs(4);
        validate_srs(&srs, 4).expect("validate");
        let encoded = encode_srs(&srs);
        let decoded = decode_srs(&encoded).expect("decode");
        assert_eq!(decoded, srs);
        validate_srs(&decoded, 4).expect("validate decoded");
    }
}

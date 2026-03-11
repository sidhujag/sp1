use rand::{rngs::StdRng, SeedableRng};
use sha2::{Digest, Sha256};
use slop_algebra::{AbstractExtensionField, AbstractField};
use sp1_germ::{
    arm_germ_aadp_template, bind_bundle_to_capsule, materialize_transcript_bound_germ_aadp_witness,
    verify_lin, verify_mul, AadpField, GermPublicValues, GermResidualPlan, GermVerifierStage,
    LinearResidualDescriptor, MultiplicativeResidualDescriptor, Sp1AadpField, Sp1GermBundle,
    Sp1LinTerm, Sp1MulTerm, TranscriptBoundSp1GermProofObject,
};
use sp1_primitives::{SP1ExtensionField, SP1Field};

fn ext_from_word(x: u32) -> SP1ExtensionField {
    let limbs: [SP1Field; 4] = core::array::from_fn(|i| SP1Field::from_canonical_u32(x + i as u32));
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

fn main() {
    // Example arm-level binding data. In production this is fixed public arming input `x`.
    let statement_digest: [u8; 32] = {
        let digest = Sha256::digest(b"statement-bytes");
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    };
    let descriptor_digest: [u8; 32] = {
        let digest = Sha256::digest(b"schedule-descriptor-bytes");
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    };
    let verifier_shape_digest: [u8; 32] = {
        let digest = Sha256::digest(b"fixed-verifier-shape-v1");
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    };
    let share_domain_separator: [u8; 32] = {
        let digest = Sha256::digest(b"share-domain-v1");
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    };
    let public_values = GermPublicValues {
        statement_digest,
        descriptor_digest,
        verifier_shape_digest,
        share_index: 0,
        share_domain_separator,
    };
    let residual_plan = GermResidualPlan::new(
        descriptor_digest,
        GermVerifierStage::Compressed,
        1,
        4,
        4,
        vec![LinearResidualDescriptor::Explicit],
        vec![
            MultiplicativeResidualDescriptor::Explicit,
            MultiplicativeResidualDescriptor::Explicit,
        ],
    );
    let capsule = public_values.arm_capsule(&residual_plan);

    // Arm stage: derive key material from arming metadata only (independent of future root/proof).
    let arm_seed: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"sp1-germ/example-arm-seed/v1");
        h.update(public_values.digest());
        let digest = h.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest);
        out
    };

    // Derive one extension-field payload we will later arm to the finalized GERM verifier.
    let mut seed16 = [0u8; 16];
    seed16.copy_from_slice(&arm_seed[..16]);
    let aadp_msg = <Sp1AadpField as AadpField>::from_u128(u128::from_le_bytes(seed16));
    let mut rng = StdRng::from_seed(arm_seed);
    let armed = arm_germ_aadp_template(&capsule, &residual_plan, aadp_msg, &mut rng)
        .expect("arm pre-proof GERM/AADP template");
    assert!(armed.template.stats.multiplication_gates >= 2);

    // Prove stage: build bundle from SP1 output and bind it against the same public arming values.
    let a = ext_from_word(10);
    let b = ext_from_word(20);
    let c = ext_from_word(30);
    let mut bundle = Sp1GermBundle::new(
        b"canonical-shared-object".to_vec(),
        vec![Sp1LinTerm::new(ext_from_word(1), ext_zero())],
        vec![
            Sp1MulTerm::new(a, b, SP1ExtensionField::one(), a * b),
            Sp1MulTerm::new(b, c, SP1ExtensionField::one(), b * c),
        ],
        b"pi-lin-bytes".to_vec(),
        b"pi-mul-bytes".to_vec(),
    );
    let (commitment_root, _challenges) =
        bind_bundle_to_capsule(&mut bundle, &capsule).expect("bundle binding");
    // These checks enforce the global mixed vanishing claims:
    //   V_lin(r_lin) = 0, V_mul(r_mul) = 0.
    let _lin_check =
        verify_lin(&bundle, &public_values, &commitment_root).expect("linear check should verify");
    let _mul_check = verify_mul(&bundle, &public_values, &commitment_root)
        .expect("multiplicative check should verify");
    let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
    let witness = materialize_transcript_bound_germ_aadp_witness(
        &armed.template,
        &capsule,
        &transcript_bound,
    )
    .expect("materialize decrypt witness");
    armed
        .template
        .check_witness(&witness)
        .expect("tiny verifier witness should satisfy AADP constraints");

    // Decap stage uses the compiled witness for the exact armed relation.
    let recovered = armed.decap_checked(&witness).expect("aadp decrypt");
    assert_eq!(recovered, aadp_msg);
}

use rand::{rngs::OsRng, rngs::StdRng, RngCore, SeedableRng};
use sha2::{Digest, Sha256};
use sp1_germ::{
    arm_germ_aadp_template, arm_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings,
    extension_terminal_openings_l_inf_u32, generate_local_dev_srs,
    materialize_transcript_bound_germ_aadp_witness,
    materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings, read_srs_from_file,
    AadpField, GermArmCapsule, GermPublicValues, GermResidualPlan, OrbweaverOpeningSrs,
    Sp1AadpField, TranscriptBoundSp1GermProofObject,
};
use sp1_prover::{
    germ_bridge::{
        build_sp1_germ_bridge_from_recursion_proof,
        build_sp1_germ_proof_object_from_recursion_proof,
        build_sp1_germ_proof_object_from_recursion_proof_with_orbweaver_terminal_openings,
        sp1_germ_residual_plan, sp1_germ_schedule_descriptor_digest,
    },
};
use sp1_sdk::{
    include_elf, utils, Elf, ProveRequest, Prover, ProverClient, ProvingKey, SP1Proof, SP1Stdin,
};
use std::time::Instant;

/// The ELF we want to execute inside the zkVM.
const ELF: Elf = include_elf!("fibonacci-program");

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn current_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kb = line
        .split_whitespace()
        .nth(1)
        .and_then(|word| word.parse::<u64>().ok())?;
    Some(kb / 1024)
}

fn log_stage(stage: &str, started: &Instant) {
    match current_rss_mb() {
        Some(rss_mb) => println!(
            "[germ-arm-bridge] stage={stage} elapsed_ms={} rss_mb={rss_mb}",
            started.elapsed().as_millis()
        ),
        None => println!(
            "[germ-arm-bridge] stage={stage} elapsed_ms={} rss_mb=unknown",
            started.elapsed().as_millis()
        ),
    }
}

#[derive(Debug, Clone)]
struct ArmPackage {
    public_values: GermPublicValues,
    residual_plan: GermResidualPlan,
    capsule: GermArmCapsule,
    key: Sp1AadpField,
}

fn build_arm_context(
    elf: &Elf,
    input_n: u32,
    vk_bytes: &[u8],
) -> ArmPackage {
    let statement_digest = {
        let mut h = Sha256::new();
        h.update(b"sp1-germ/statement/v1");
        h.update(&elf[..]);
        h.update(input_n.to_le_bytes());
        let digest: [u8; 32] = h.finalize().into();
        digest
    };
    let descriptor_digest =
        sp1_germ_schedule_descriptor_digest().expect("derive SP1 GERM schedule descriptor");
    let verifier_shape_digest = {
        let mut h = Sha256::new();
        h.update(b"sp1-germ/verifier-shape/v1");
        h.update(vk_bytes);
        let digest: [u8; 32] = h.finalize().into();
        digest
    };
    let share_domain_separator = sha256_bytes(b"sp1/fibonacci/share-domain/v1");

    let public_values = GermPublicValues {
        statement_digest,
        descriptor_digest,
        verifier_shape_digest,
        share_index: 0,
        share_domain_separator,
    };
    let residual_plan = sp1_germ_residual_plan().expect("derive SP1 GERM residual plan");
    let capsule = public_values.arm_capsule(&residual_plan);
    let mut key_bytes = [0u8; 16];
    OsRng.fill_bytes(&mut key_bytes);
    let key = <Sp1AadpField as AadpField>::from_u128(u128::from_le_bytes(key_bytes));
    ArmPackage { public_values, residual_plan, capsule, key }
}

fn sumcheck_nvars(total_checks: usize) -> usize {
    if total_checks <= 1 {
        0
    } else {
        total_checks.next_power_of_two().trailing_zeros() as usize
    }
}

fn maybe_load_orbweaver_srs(required_width: usize) -> Option<OrbweaverOpeningSrs> {
    if std::env::var_os("SP1_GERM_USE_LOCAL_ORBWEAVER_SRS").is_some() {
        return Some(generate_local_dev_srs(required_width));
    }
    let path = std::env::var_os("SP1_GERM_ORBWEAVER_SRS")?;
    Some(read_srs_from_file(std::path::Path::new(&path), required_width).expect("load Orbweaver opening SRS"))
}

#[tokio::main]
async fn main() {
    utils::setup_logger();
    let started = Instant::now();
    log_stage("start", &started);

    let n = 500u32;
    let vk_probe_client = ProverClient::from_env().await;
    log_stage("client-ready", &started);
    let pk = vk_probe_client.setup(ELF).await.expect("setup");
    log_stage("setup-ready", &started);
    let vk_bytes = bincode::serialize(pk.verifying_key()).expect("serialize vk");
    let required_orbweaver_width =
        (1usize << usize::from(sp1_germ_residual_plan().expect("derive SP1 GERM residual plan").sumcheck_rounds))
            * 16;
    let orbweaver_srs = maybe_load_orbweaver_srs(required_orbweaver_width);

    // Setup happens before proof generation; the exact Orbweaver AADP relation is transcript-bound.
    let mut stdin = SP1Stdin::new();
    stdin.write(&n);
    log_stage("stdin-ready", &started);
    let client = vk_probe_client;
    let arm_pkg = build_arm_context(&ELF, n, &vk_bytes);
    log_stage("arm-context-ready", &started);

    // PROVE phase.
    let proof = client
        .prove(&pk, stdin)
        .compressed()
        .await
        .expect("compressed prove");
    log_stage("compressed-proof-ready", &started);
    client
        .verify(&proof, pk.verifying_key(), None)
        .expect("compressed verify");
    log_stage("compressed-verify-ok", &started);

    // Extract recursion proof and build the in-memory bridge from the real shard proof.
    let recursion_proof = match &proof.proof {
        SP1Proof::Compressed(recursion) => recursion.as_ref(),
        other => panic!("expected compressed proof, got mode {other:?}"),
    };
    let bridge = build_sp1_germ_bridge_from_recursion_proof(recursion_proof).expect("build bridge");
    assert_eq!(
        arm_pkg.public_values.descriptor_digest,
        sp1_germ_schedule_descriptor_digest().expect("derive SP1 GERM schedule descriptor"),
        "arm-time schedule descriptor drifted before bridge binding"
    );
    log_stage("bridge-built", &started);

    let (proof_object, commitment_root) = if let Some(srs) = orbweaver_srs.as_ref() {
        build_sp1_germ_proof_object_from_recursion_proof_with_orbweaver_terminal_openings(
            recursion_proof,
            &arm_pkg.capsule,
            srs,
        )
        .expect("build SP1 GERM proof object with Orbweaver terminal openings")
    } else {
        build_sp1_germ_proof_object_from_recursion_proof(recursion_proof, &arm_pkg.capsule)
            .expect("build SP1 GERM proof object")
    };
    let transcript_bound = TranscriptBoundSp1GermProofObject::new(proof_object, commitment_root);
    log_stage("germ-verified", &started);
    if orbweaver_srs.is_some() {
        let pi_norm =
            extension_terminal_openings_l_inf_u32(&transcript_bound.proof_object.pi_mul_terminal_openings)
                .expect("measure Orbweaver terminal opening norm");
        println!("[germ-arm-bridge] orbweaver_terminal_opening_linf_u32={pi_norm}");
    }

    let mut encryption_seed = [0u8; 32];
    OsRng.fill_bytes(&mut encryption_seed);
    let mut rng = StdRng::from_seed(encryption_seed);
    let armed = if let Some(srs) = orbweaver_srs.as_ref() {
        arm_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings(
            &arm_pkg.capsule,
            &arm_pkg.residual_plan,
            &transcript_bound,
            srs,
            arm_pkg.key,
            &mut rng,
        )
        .expect("arm transcript-bound GERM/AADP template with Orbweaver terminal openings")
    } else {
        arm_germ_aadp_template(&arm_pkg.capsule, &arm_pkg.residual_plan, arm_pkg.key, &mut rng)
            .expect("arm GERM/AADP template")
    };
    log_stage("aadp-armed", &started);

    let decrypt_witness = if let Some(srs) = orbweaver_srs.as_ref() {
        materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings(
            &armed.template,
            &arm_pkg.capsule,
            &transcript_bound,
            srs,
        )
        .expect("materialize post-proof decrypt witness with Orbweaver terminal openings")
    } else {
        materialize_transcript_bound_germ_aadp_witness(
            &armed.template,
            &arm_pkg.capsule,
            &transcript_bound,
        )
        .expect("materialize post-proof decrypt witness")
    };
    armed
        .template
        .check_witness(&decrypt_witness)
        .expect("aadp witness satisfies compiled constraints");
    log_stage("aadp-constraints-armed", &started);

    let recovered = armed.decap_checked(&decrypt_witness).expect("aadp decap");
    assert_eq!(recovered, arm_pkg.key);

    // Negative check: witness tamper must fail checked decap.
    let mut tampered_witness = decrypt_witness.witness.clone();
    tampered_witness[0] += <Sp1AadpField as AadpField>::one();
    let tampered_decap = armed.decap_checked(&sp1_germ::GermAadpWitness { witness: tampered_witness });
    assert!(
        tampered_decap.is_err(),
        "tampered witness unexpectedly passed checked decap"
    );

    // Attack simulation (real model): keep ciphertext fixed, mutate bundle bytes.
    // Without access to arm-time key/ciphertext generation, attacker mutations should not unlock K.
    let mut tampered_proof_object = transcript_bound.proof_object.clone();
    tampered_proof_object.pi_lin[0] ^= 0x01;
    let tampered_transcript = TranscriptBoundSp1GermProofObject::new(
        tampered_proof_object,
        transcript_bound.commitment_root,
    );
    let tampered_compile = if let Some(srs) = orbweaver_srs.as_ref() {
        materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings(
            &armed.template,
            &arm_pkg.capsule,
            &tampered_transcript,
            srs,
        )
    } else {
        materialize_transcript_bound_germ_aadp_witness(
            &armed.template,
            &arm_pkg.capsule,
            &tampered_transcript,
        )
    };
    assert!(
        tampered_compile.is_err(),
        "tampered bundle unexpectedly compiled against fixed armed relation"
    );

    // Stronger attacker model: attacker rebinds a mutated bundle but must use the original ciphertext.
    // This should not recover the original arm-time key.
    let mut rebinding_attack_proof_object = transcript_bound.proof_object.clone();
    rebinding_attack_proof_object.mul_terms[0].d += <Sp1AadpField as AadpField>::one();
    let rebound_transcript = TranscriptBoundSp1GermProofObject::new(
        rebinding_attack_proof_object,
        transcript_bound.commitment_root,
    );
    let rebound_compile = if let Some(srs) = orbweaver_srs.as_ref() {
        materialize_transcript_bound_germ_aadp_witness_with_orbweaver_terminal_openings(
            &armed.template,
            &arm_pkg.capsule,
            &rebound_transcript,
            srs,
        )
    } else {
        materialize_transcript_bound_germ_aadp_witness(
            &armed.template,
            &arm_pkg.capsule,
            &rebound_transcript,
        )
    };
    if let Ok(rebound_witness) = rebound_compile {
            let rebound_try = armed.decap_checked(&rebound_witness);
            assert!(
                !matches!(rebound_try, Ok(k) if k == arm_pkg.key),
                "rebound attack unexpectedly recovered the original armed key"
            );
        }
    log_stage("aadp-decap-ok", &started);

    let nvars = sumcheck_nvars(transcript_bound.proof_object.mul_terms.len());
    println!(
        "ok: orbweaver_enabled={} bridge_tables={} rlin_tables={} rmul_tables={} residual_plan_rounds={} lin_terms={} mul_terms={} sumcheck_nvars={} aadp_vars={} aadp_constraints={} lin_checks={} opening_checks={} mul_gates={}",
        orbweaver_srs.is_some(),
        bridge.tables.len(),
        bridge.manifest.rlin_tables.len(),
        bridge.manifest.rmul_tables.len(),
        arm_pkg.residual_plan.sumcheck_rounds,
        transcript_bound.proof_object.lin_terms.len(),
        transcript_bound.proof_object.mul_terms.len(),
        nvars,
        armed.template.cs.num_variables,
        armed.template.cs.constraints.len(),
        armed.template.stats.linear_round_checks,
        armed.template.stats.opening_checks,
        armed.template.stats.multiplication_gates
    );
}

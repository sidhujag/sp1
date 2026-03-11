use std::borrow::Borrow;

use anyhow::{anyhow, Context, Result};
use itertools::Itertools;
use sha2::{Digest, Sha256};
use slop_air::BaseAir;
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use slop_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use slop_multilinear::{
    full_geq, partial_lagrange_blocking, Mle, MleEval, MultilinearPcsChallenger, Point,
};
use slop_sumcheck::partially_verify_sumcheck_proof;
use sp1_germ::{
    bind_bundle_to_capsule, compute_commitment_root, GermArmCapsule, GermResidualPlan,
    GermVerifierStage, LinearResidualDescriptor, MultiplicativeResidualDescriptor,
    Sp1GermProofObject, Sp1LinTerm, Sp1MulTerm,
};
use sp1_hypercube::{
    air::MachineAir, LogUpEvaluations, LogUpGkrVerifier, MachineRecord, SP1PcsProofInner,
    SP1RecursionProof, ShardProof, PROOF_MAX_NUM_PVS,
};
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext};
use sp1_recursion_executor::RecursionPublicValues;

use crate::{
    components::{RecursionSC, SP1ProverComponents},
    shapes::{SP1RecursionProofShape, DEFAULT_ARITY},
    utils::is_recursion_public_values_valid,
    CpuSP1ProverComponents, SP1_CIRCUIT_VERSION,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeManifest {
    pub schema: String,
    pub schema_version: u32,
    pub public_instance_tables: Vec<String>,
    pub shared_object_tables: Vec<String>,
    pub rlin_tables: Vec<String>,
    pub rmul_tables: Vec<String>,
    pub blob_tables: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeTable {
    pub name: String,
    pub rows: u32,
    pub cols: u32,
    pub values_u32_le: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Sp1GermBridge {
    pub manifest: BridgeManifest,
    pub tables: Vec<BridgeTable>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResidualPlanV1 {
    pub stage: String,
    pub schema: String,
    pub schema_version: u32,
    pub sumcheck_rounds: u16,
    pub linear_opening_rows: u16,
    pub linear_opening_ring_dim: u16,
    pub exported_linear_families: Vec<String>,
    pub exported_multiplicative_families: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeDescriptorV2 {
    pub schema: String,
    pub schema_version: u32,
    pub schedule_descriptor_digest: [u8; 32],
    pub residual_plan: ResidualPlanV1,
    pub linear_descriptors: Vec<LinearResidualDescriptor>,
    pub multiplicative_descriptors: Vec<MultiplicativeResidualDescriptor>,
}

fn sanitize_table_component(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

fn bytes_to_len_prefixed_u32_words(bytes: &[u8]) -> Result<Vec<u32>> {
    let len_u32: u32 = bytes
        .len()
        .try_into()
        .map_err(|_| anyhow!("blob too large to encode as u32 length prefix"))?;
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(4));
    out.push(len_u32);
    for chunk in bytes.chunks(4) {
        let mut padded = [0u8; 4];
        padded[..chunk.len()].copy_from_slice(chunk);
        out.push(u32::from_le_bytes(padded));
    }
    Ok(out)
}

fn push_table(
    tables: &mut Vec<BridgeTable>,
    name: String,
    rows: u32,
    cols: u32,
    values_u32_le: Vec<u32>,
) -> Result<()> {
    let expected_len: usize = (rows as usize)
        .checked_mul(cols as usize)
        .ok_or_else(|| anyhow!("table {name} row*col overflow"))?;
    anyhow::ensure!(
        values_u32_le.len() == expected_len,
        "table {} value count mismatch: got {}, expected {}",
        name,
        values_u32_le.len(),
        expected_len
    );
    tables.push(BridgeTable { name, rows, cols, values_u32_le });
    Ok(())
}

fn push_single_row_table(
    tables: &mut Vec<BridgeTable>,
    name: String,
    values: Vec<u32>,
) -> Result<()> {
    let cols: u32 =
        values.len().try_into().map_err(|_| anyhow!("table {name} too wide for u32 cols"))?;
    push_table(tables, name, 1, cols, values)
}

fn push_blob_table<T: serde::Serialize>(
    tables: &mut Vec<BridgeTable>,
    name: String,
    value: &T,
) -> Result<()> {
    let bytes =
        bincode::serialize(value).with_context(|| format!("serialize blob table {name}"))?;
    let words = bytes_to_len_prefixed_u32_words(&bytes)?;
    push_single_row_table(tables, name, words)
}

fn ext_to_limbs_u32(x: &SP1ExtensionField) -> [u32; 4] {
    let limbs = <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(x);
    [
        limbs[0].as_canonical_u32(),
        limbs[1].as_canonical_u32(),
        limbs[2].as_canonical_u32(),
        limbs[3].as_canonical_u32(),
    ]
}

fn flatten_ext_slice_to_u32(values: &[SP1ExtensionField]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend(ext_to_limbs_u32(value));
    }
    out
}

fn point_ext_to_u32_values(point: &Point<SP1ExtensionField>) -> (u32, u32, Vec<u32>) {
    let vals: Vec<SP1ExtensionField> = point.iter().copied().collect();
    (1, (vals.len() * 4) as u32, flatten_ext_slice_to_u32(&vals))
}

fn mle_eval_ext_to_u32_values(eval: &MleEval<SP1ExtensionField>) -> (u32, u32, Vec<u32>) {
    let values: Vec<SP1ExtensionField> = eval.iter().copied().collect();
    (1, (values.len() * 4) as u32, flatten_ext_slice_to_u32(&values))
}

fn mle_ext_to_u32_values(mle: &Mle<SP1ExtensionField>) -> (u32, u32, Vec<u32>) {
    let rows = 1usize << (mle.num_variables() as usize);
    let cols = mle.num_polynomials() * 4;
    let values = flatten_ext_slice_to_u32(mle.guts().as_slice());
    (rows as u32, cols as u32, values)
}

fn default_mul_sumcheck_rounds() -> u16 {
    let verifier = CpuSP1ProverComponents::compress_verifier();
    let machine = verifier.machine();
    let max_log_row_count = verifier.max_log_row_count();
    let chip_count = machine.chips().len();
    let num_of_interactions =
        machine.chips().iter().map(|c| c.sends().len() + c.receives().len()).sum::<usize>();
    let interaction_vars = num_of_interactions.next_power_of_two().ilog2() as usize;
    let output_layer_checks = 1usize << (interaction_vars + 1);
    let degree_checks_per_chip = (2 * (max_log_row_count + 1)).saturating_sub(1);
    let round_count = max_log_row_count.saturating_sub(1);
    let upper_bound_terms = chip_count * degree_checks_per_chip
        + 2 // public-value / cumulative fallback checks
        + output_layer_checks // denominator inverse checks
        + (2 * round_count) // GKR claimed-sum and final-eval checks
        + max_log_row_count // trace-point equality checks
        + 2; // final numerator/denominator equality checks
    upper_bound_terms.next_power_of_two().ilog2() as u16
}

fn default_linear_descriptors() -> Vec<LinearResidualDescriptor> {
    let verifier = CpuSP1ProverComponents::compress_verifier();
    let machine = verifier.machine();
    vec![
        LinearResidualDescriptor::PublicValuesPadding {
            start_idx: machine.num_pv_elts(),
            count: PROOF_MAX_NUM_PVS.saturating_sub(machine.num_pv_elts()),
        },
        LinearResidualDescriptor::RecursionPublicValuesDigest,
        LinearResidualDescriptor::IsComplete,
        LinearResidualDescriptor::ZerocheckPointEval,
        LinearResidualDescriptor::ZerocheckClaimedSum,
    ]
}

fn default_multiplicative_descriptors() -> Vec<MultiplicativeResidualDescriptor> {
    let verifier = CpuSP1ProverComponents::compress_verifier();
    let machine = verifier.machine();
    let max_log_row_count = verifier.max_log_row_count();
    let mut out = Vec::new();
    for chip in machine.chips() {
        for bit_idx in 0..(max_log_row_count + 1) {
            out.push(MultiplicativeResidualDescriptor::DegreeBitBooleanity {
                chip_name: chip.name().to_string(),
                bit_idx,
            });
        }
        for bit_idx in 1..(max_log_row_count + 1) {
            out.push(MultiplicativeResidualDescriptor::DegreeHeightProduct {
                chip_name: chip.name().to_string(),
                bit_idx,
            });
        }
    }
    out.push(MultiplicativeResidualDescriptor::GkrPowWitness);
    out.push(MultiplicativeResidualDescriptor::GkrCumulativeSum);
    let num_of_interactions =
        machine.chips().iter().map(|c| c.sends().len() + c.receives().len()).sum::<usize>();
    let interaction_vars = num_of_interactions.next_power_of_two().ilog2() as usize;
    let output_layer_checks = 1usize << (interaction_vars + 1);
    for index in 0..output_layer_checks {
        out.push(MultiplicativeResidualDescriptor::GkrDenominatorInverse { index });
    }
    for round_idx in 0..max_log_row_count.saturating_sub(1) {
        out.push(MultiplicativeResidualDescriptor::GkrRoundClaimedSum { round_idx });
        out.push(MultiplicativeResidualDescriptor::GkrRoundFinalEval { round_idx });
    }
    for coord_idx in 0..max_log_row_count {
        out.push(MultiplicativeResidualDescriptor::GkrTracePointCoord { coord_idx });
    }
    out.push(MultiplicativeResidualDescriptor::GkrFinalNumeratorEval);
    out.push(MultiplicativeResidualDescriptor::GkrFinalDenominatorEval);
    out
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

fn base_to_ext(x: SP1Field) -> SP1ExtensionField {
    SP1ExtensionField::from_base_slice(&[x, SP1Field::zero(), SP1Field::zero(), SP1Field::zero()])
}

fn push_linear_residual(linear_terms: &mut Vec<Sp1LinTerm>, residual: SP1ExtensionField) {
    linear_terms.push(Sp1LinTerm::new(ext_one(), residual));
}

fn push_mul_residual(
    mul_terms: &mut Vec<Sp1MulTerm>,
    a: SP1ExtensionField,
    b: SP1ExtensionField,
    c: SP1ExtensionField,
    d: SP1ExtensionField,
) {
    mul_terms.push(Sp1MulTerm::new(a, b, c, d));
}

fn is_zero_ext(value: &SP1ExtensionField) -> bool {
    <SP1ExtensionField as AbstractExtensionField<SP1Field>>::as_base_slice(value)
        .iter()
        .all(|x| x.as_canonical_u32() == 0)
}

fn default_residual_plan_v1(schedule_descriptor_digest: [u8; 32]) -> ResidualPlanV1 {
    ResidualPlanV1 {
        stage: "compressed".to_string(),
        schema: "sp1-hypercube-germ-bridge".to_string(),
        schema_version: 2,
        // The host-side multiplicative proof still reuses the fixed-round sumcheck schedule,
        // while the tiny verifier commits one scalar residual per exported multiplicative descriptor.
        sumcheck_rounds: default_mul_sumcheck_rounds(),
        linear_opening_rows: 4,
        linear_opening_ring_dim: 4,
        exported_linear_families: vec![
            "bridge/proof/public_values".to_string(),
            "bridge/opened/*/{main,pre,degree_bits}".to_string(),
            "bridge/zerocheck/{point,point_eval,claimed_sum}".to_string(),
            "bridge/blob/{main_commitment,zerocheck_proof,evaluation_proof}".to_string(),
        ],
        exported_multiplicative_families: vec![
            "bridge/logup/point".to_string(),
            "bridge/logup/circuit_output/{numerator,denominator}".to_string(),
            "bridge/logup/chip_openings/*/{main_eval,pre_eval}".to_string(),
            "bridge/logup/round/*/{quad,sumcheck_blob}".to_string(),
            "bridge/blob/logup_gkr_proof".to_string(),
            format!("schedule:{:02x?}", schedule_descriptor_digest),
        ],
    }
}

pub fn sp1_germ_schedule_descriptor_digest() -> Result<[u8; 32]> {
    let reduce_shape = SP1RecursionProofShape::compress_proof_shape_from_arity(DEFAULT_ARITY)
        .context("default SP1 recursion arity schedule should exist")?;
    let reduce_shape_bytes =
        bincode::serialize(&reduce_shape).context("serialize SP1 recursion shape")?;

    let mut h = Sha256::new();
    h.update(b"sp1-germ/schedule-descriptor/v1");
    h.update(SP1_CIRCUIT_VERSION.as_bytes());
    h.update((DEFAULT_ARITY as u32).to_le_bytes());
    h.update(b"sp1-hypercube-germ-bridge");
    h.update(1u32.to_le_bytes());
    for family in [
        b"bridge/version".as_slice(),
        b"bridge/proof/public_values",
        b"bridge/opened/*/{main,pre,degree_bits}",
        b"bridge/zerocheck/{point,point_eval,claimed_sum}",
        b"bridge/logup/point",
        b"bridge/logup/circuit_output/{numerator,denominator}",
        b"bridge/logup/chip_openings/*/{main_eval,pre_eval}",
        b"bridge/logup/round/*/{quad,sumcheck_blob}",
        b"bridge/blob/{main_commitment,logup_gkr_proof,zerocheck_proof,evaluation_proof}",
        b"bridge/manifest/json",
    ] {
        h.update((family.len() as u64).to_le_bytes());
        h.update(family);
    }
    h.update((reduce_shape_bytes.len() as u64).to_le_bytes());
    h.update(&reduce_shape_bytes);

    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    Ok(out)
}

pub fn sp1_germ_bridge_descriptor_v2() -> Result<BridgeDescriptorV2> {
    let schedule_descriptor_digest = sp1_germ_schedule_descriptor_digest()?;
    let linear_descriptors = default_linear_descriptors();
    let multiplicative_descriptors = default_multiplicative_descriptors();
    Ok(BridgeDescriptorV2 {
        schema: "sp1-hypercube-germ-bridge".to_string(),
        schema_version: 2,
        schedule_descriptor_digest,
        residual_plan: default_residual_plan_v1(schedule_descriptor_digest),
        linear_descriptors,
        multiplicative_descriptors,
    })
}

pub fn sp1_germ_residual_plan() -> Result<GermResidualPlan> {
    let descriptor = sp1_germ_bridge_descriptor_v2()?;
    Ok(GermResidualPlan::new(
        descriptor.schedule_descriptor_digest,
        GermVerifierStage::Compressed,
        descriptor.residual_plan.sumcheck_rounds,
        descriptor.residual_plan.linear_opening_rows,
        descriptor.residual_plan.linear_opening_ring_dim,
        descriptor.linear_descriptors,
        descriptor.multiplicative_descriptors,
    ))
}

fn export_linear_residual_terms_from_recursion_proof(
    proof: &SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
) -> Result<Vec<Sp1LinTerm>> {
    let descriptors = default_linear_descriptors();
    let verifier = CpuSP1ProverComponents::compress_verifier();
    let shard_proof = &proof.proof;
    let vk = &proof.vk;
    let machine = verifier.machine();
    let max_log_row_count = verifier.max_log_row_count();
    let machine_chips: Vec<_> = machine.chips().iter().cloned().collect();

    let mut linear_terms = Vec::new();

    if shard_proof.public_values.len() != PROOF_MAX_NUM_PVS {
        return Err(anyhow!(
            "invalid recursion public values length: got {} expected {}",
            shard_proof.public_values.len(),
            PROOF_MAX_NUM_PVS
        ));
    }

    let recursion_public_values: &RecursionPublicValues<_> =
        shard_proof.public_values.as_slice().borrow();

    let mut challenger = verifier.challenger();
    vk.observe_into(&mut challenger);

    for &value in &shard_proof.public_values {
        challenger.observe(value);
    }
    challenger.observe(shard_proof.main_commitment);

    let shard_chip_names =
        shard_proof.opened_values.chips.keys().cloned().collect::<std::collections::BTreeSet<_>>();
    challenger.observe(SP1Field::from_canonical_usize(shard_chip_names.len()));

    let mut heights = std::collections::BTreeMap::new();
    let mut degrees = std::collections::BTreeMap::new();
    for (name, chip_values) in &shard_proof.opened_values.chips {
        if chip_values.degree.len() != max_log_row_count + 1 || chip_values.degree.len() >= 30 {
            return Err(anyhow!("invalid degree shape for chip {name}"));
        }
        let acc =
            chip_values.degree.iter().fold(SP1Field::zero(), |acc, &x| x + SP1Field::two() * acc);
        heights.insert(name.clone(), acc);
        degrees.insert(name.clone(), chip_values.degree.clone());
        challenger.observe(acc);
        challenger.observe(SP1Field::from_canonical_usize(name.len()));
        for byte in name.as_bytes() {
            challenger.observe(SP1Field::from_canonical_u8(*byte));
        }
    }

    let machine_chip_names = machine_chips
        .iter()
        .map(|chip| chip.name().to_string())
        .collect::<std::collections::BTreeSet<_>>();
    let preprocessed_chips: Vec<_> =
        machine_chips.iter().filter(|chip| chip.preprocessed_width() != 0).cloned().collect();

    if !shard_chip_names.is_subset(&machine_chip_names)
        || !preprocessed_chips
            .iter()
            .map(|chip| chip.name().to_string())
            .collect::<std::collections::BTreeSet<_>>()
            .is_subset(&shard_chip_names)
    {
        return Err(anyhow!("proof chip set is incompatible with recursion machine"));
    }

    let shard_chip_set = machine
        .chips()
        .iter()
        .filter(|chip| shard_chip_names.contains(chip.name()))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();

    if shard_chip_set.len() != shard_chip_names.len()
        || !machine.shape().chip_clusters.contains(&shard_chip_set)
    {
        return Err(anyhow!("proof shard chip cluster is invalid for recursion machine"));
    }

    if shard_chip_set.len() != shard_proof.opened_values.chips.len()
        || shard_chip_set.len() != shard_proof.logup_gkr_proof.logup_evaluations.chip_openings.len()
    {
        return Err(anyhow!("opened values / GKR chip opening size mismatch"));
    }

    LogUpGkrVerifier::<SP1GlobalContext, RecursionSC>::verify_logup_gkr(
        &shard_chip_set,
        &degrees,
        max_log_row_count,
        &shard_proof.logup_gkr_proof,
        &shard_proof.public_values,
        &mut challenger,
    )
    .map_err(|err| anyhow!("advance challenger through logup verification: {err}"))?;

    let alpha = challenger.sample_ext_element::<SP1ExtensionField>();
    let gkr_batch_open_challenge = challenger.sample_ext_element::<SP1ExtensionField>();
    let lambda = challenger.sample_ext_element::<SP1ExtensionField>();

    if shard_proof.logup_gkr_proof.logup_evaluations.point.dimension() != max_log_row_count
        || shard_proof.zerocheck_proof.point_and_eval.0.dimension() != max_log_row_count
    {
        return Err(anyhow!("zerocheck/GKR point dimension mismatch"));
    }

    let zerocheck_eq_val = Mle::full_lagrange_eval(
        &shard_proof.logup_gkr_proof.logup_evaluations.point,
        &shard_proof.zerocheck_proof.point_and_eval.0,
    );

    let mut rlc_eval = ext_zero();
    for (chip, (chip_name, openings)) in
        shard_chip_set.iter().zip_eq(shard_proof.opened_values.chips.iter())
    {
        if chip.name() != chip_name.as_str() {
            return Err(anyhow!("chip order mismatch in opened values"));
        }
        if openings.preprocessed.local.len() != chip.preprocessed_width()
            || openings.main.local.len() != chip.width()
        {
            return Err(anyhow!("opening width mismatch for chip {chip_name}"));
        }

        let mut point_extended = shard_proof.zerocheck_proof.point_and_eval.0.clone();
        point_extended.add_dimension(SP1ExtensionField::zero());
        let geq_val = full_geq(&openings.degree, &point_extended);
        let padded_row_adjustment =
            sp1_hypercube::ShardVerifier::<SP1GlobalContext, RecursionSC>::compute_padded_row_adjustment(
                chip,
                alpha,
                &shard_proof.public_values,
            );
        let constraint_eval =
            sp1_hypercube::ShardVerifier::<SP1GlobalContext, RecursionSC>::eval_constraints(
                chip,
                openings,
                alpha,
                &shard_proof.public_values,
            ) - padded_row_adjustment * geq_val;

        let openings_batch = openings
            .main
            .local
            .iter()
            .chain(openings.preprocessed.local.iter())
            .copied()
            .zip(gkr_batch_open_challenge.powers().skip(1))
            .map(|(opening, power)| opening * power)
            .sum::<SP1ExtensionField>();

        rlc_eval = rlc_eval * lambda + zerocheck_eq_val * (constraint_eval + openings_batch);
    }

    let zerocheck_sum_modifications_from_gkr = shard_proof
        .logup_gkr_proof
        .logup_evaluations
        .chip_openings
        .values()
        .map(|chip_evaluation| {
            chip_evaluation
                .main_trace_evaluations
                .iter()
                .copied()
                .chain(
                    chip_evaluation
                        .preprocessed_trace_evaluations
                        .iter()
                        .flat_map(|evals| evals.iter().copied()),
                )
                .zip(gkr_batch_open_challenge.powers().skip(1))
                .map(|(opening, power)| opening * power)
                .sum::<SP1ExtensionField>()
        })
        .collect::<Vec<_>>();

    let zerocheck_sum_modification = zerocheck_sum_modifications_from_gkr
        .iter()
        .fold(ext_zero(), |acc, modification| lambda * acc + *modification);
    for descriptor in descriptors {
        match descriptor {
            LinearResidualDescriptor::Explicit => {
                return Err(anyhow!(
                    "explicit linear descriptors are not supported by the SP1 recursion exporter"
                ));
            }
            LinearResidualDescriptor::PublicValuesPadding { start_idx, count } => {
                for value in shard_proof.public_values[start_idx..start_idx + count].iter().copied()
                {
                    push_linear_residual(&mut linear_terms, base_to_ext(value));
                }
            }
            LinearResidualDescriptor::RecursionPublicValuesDigest => {
                if !is_recursion_public_values_valid(shard_proof.public_values.as_slice().borrow())
                {
                    push_linear_residual(&mut linear_terms, ext_one());
                } else {
                    push_linear_residual(&mut linear_terms, ext_zero());
                }
            }
            LinearResidualDescriptor::IsComplete => {
                push_linear_residual(
                    &mut linear_terms,
                    base_to_ext(recursion_public_values.is_complete - SP1Field::one()),
                );
            }
            LinearResidualDescriptor::ZerocheckPointEval => {
                push_linear_residual(
                    &mut linear_terms,
                    shard_proof.zerocheck_proof.point_and_eval.1 - rlc_eval,
                );
            }
            LinearResidualDescriptor::ZerocheckClaimedSum => {
                push_linear_residual(
                    &mut linear_terms,
                    shard_proof.zerocheck_proof.claimed_sum - zerocheck_sum_modification,
                );
            }
        }
    }

    Ok(linear_terms)
}

fn export_multiplicative_residual_terms_from_recursion_proof(
    proof: &SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
) -> Result<Vec<Sp1MulTerm>> {
    let descriptors = default_multiplicative_descriptors();
    let verifier = CpuSP1ProverComponents::compress_verifier();
    let shard_proof = &proof.proof;
    let vk = &proof.vk;
    let machine = verifier.machine();

    let mut challenger = verifier.challenger();
    vk.observe_into(&mut challenger);
    for &value in &shard_proof.public_values {
        challenger.observe(value);
    }
    challenger.observe(shard_proof.main_commitment);

    let shard_chip_names =
        shard_proof.opened_values.chips.keys().cloned().collect::<std::collections::BTreeSet<_>>();
    challenger.observe(SP1Field::from_canonical_usize(shard_chip_names.len()));

    let mut degrees = std::collections::BTreeMap::new();
    for (name, chip_values) in &shard_proof.opened_values.chips {
        let acc =
            chip_values.degree.iter().fold(SP1Field::zero(), |acc, &x| x + SP1Field::two() * acc);
        degrees.insert(name.clone(), chip_values.degree.clone());
        challenger.observe(acc);
        challenger.observe(SP1Field::from_canonical_usize(name.len()));
        for byte in name.as_bytes() {
            challenger.observe(SP1Field::from_canonical_u8(*byte));
        }
    }

    let shard_chips = machine
        .chips()
        .iter()
        .filter(|chip| shard_chip_names.contains(chip.name()))
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();

    let mut mul_terms = Vec::new();

    let logup = &shard_proof.logup_gkr_proof;
    let pow_residual = !challenger.check_witness(sp1_hypercube::GKR_GRINDING_BITS, logup.witness);
    let alpha = challenger.sample_ext_element::<SP1ExtensionField>();
    let beta_seed_dim = {
        let max_interaction_arity = shard_chips
            .iter()
            .flat_map(|c| c.sends().iter().chain(c.receives().iter()))
            .map(|i| i.values.len() + 1)
            .max()
            .unwrap_or(1);
        let max_interaction_kinds_values = sp1_hypercube::prover::Record::<
            SP1GlobalContext,
            RecursionSC,
        >::interactions_in_public_values()
        .iter()
        .map(|kind| kind.num_values() + 1)
        .max()
        .unwrap_or(1);
        std::cmp::max(max_interaction_arity, max_interaction_kinds_values)
            .next_power_of_two()
            .ilog2()
    };
    let beta_seed = (0..beta_seed_dim)
        .map(|_| challenger.sample_ext_element::<SP1ExtensionField>())
        .collect::<Point<_>>();
    let pv_challenge = challenger.sample_ext_element::<SP1ExtensionField>();
    let cumulative_sum =
        match LogUpGkrVerifier::<SP1GlobalContext, RecursionSC>::verify_public_values(
            pv_challenge,
            &alpha,
            &beta_seed,
            &shard_proof.public_values,
        ) {
            Ok(digest) => -digest,
            Err(_) => {
                push_mul_residual(&mut mul_terms, ext_one(), ext_one(), ext_zero(), ext_one());
                ext_zero()
            }
        };

    let numerator = &logup.circuit_output.numerator;
    let denominator = &logup.circuit_output.denominator;
    challenger.observe(SP1Field::from_canonical_usize(numerator.guts().as_slice().len()));
    for &value in numerator.guts().as_slice() {
        challenger.observe_ext_element(value);
    }
    challenger.observe(SP1Field::from_canonical_usize(denominator.guts().as_slice().len()));
    for &value in denominator.guts().as_slice() {
        challenger.observe_ext_element(value);
    }
    let output_cumulative_sum = numerator
        .guts()
        .as_slice()
        .iter()
        .zip_eq(denominator.guts().as_slice().iter())
        .map(|(n, d)| *n / *d)
        .sum::<SP1ExtensionField>();
    let denominator_inverses = denominator
        .guts()
        .as_slice()
        .iter()
        .map(|d| d.try_inverse().unwrap_or_else(ext_zero))
        .collect::<Vec<_>>();

    let num_of_interactions =
        shard_chips.iter().map(|c| c.sends().len() + c.receives().len()).sum::<usize>();
    let number_of_interaction_variables = num_of_interactions.next_power_of_two().ilog2();
    let first_eval_point =
        challenger.sample_point::<SP1ExtensionField>(number_of_interaction_variables + 1);
    let mut numerator_eval = numerator.blocking_eval_at(&first_eval_point)[0];
    let mut denominator_eval = denominator.blocking_eval_at(&first_eval_point)[0];
    let mut eval_point = first_eval_point;

    let mut round_claim_residuals = Vec::with_capacity(logup.round_proofs.len());
    let mut round_final_evals = Vec::with_capacity(logup.round_proofs.len());
    for (round_idx, round_proof) in logup.round_proofs.iter().enumerate() {
        let lambda = challenger.sample_ext_element::<SP1ExtensionField>();
        let expected_claim = numerator_eval * lambda + denominator_eval;
        round_claim_residuals.push(round_proof.sumcheck_proof.claimed_sum - expected_claim);
        partially_verify_sumcheck_proof(
            &round_proof.sumcheck_proof,
            &mut challenger,
            round_idx + number_of_interaction_variables as usize + 1,
            3,
        )
        .map_err(|err| anyhow!("replay logup sumcheck round {}: {err}", round_idx))?;
        let (point, final_eval) = round_proof.sumcheck_proof.point_and_eval.clone();
        let eq_eval = Mle::full_lagrange_eval(&point, &eval_point);
        let numerator_sumcheck_eval = round_proof.numerator_0 * round_proof.denominator_1
            + round_proof.numerator_1 * round_proof.denominator_0;
        let denominator_sumcheck_eval = round_proof.denominator_0 * round_proof.denominator_1;
        let combined = numerator_sumcheck_eval * lambda + denominator_sumcheck_eval;
        round_final_evals.push((eq_eval, combined, final_eval));
        challenger.observe_ext_element(round_proof.numerator_0);
        challenger.observe_ext_element(round_proof.numerator_1);
        challenger.observe_ext_element(round_proof.denominator_0);
        challenger.observe_ext_element(round_proof.denominator_1);
        eval_point = round_proof.sumcheck_proof.point_and_eval.0.clone();
        let last_coordinate = challenger.sample_ext_element::<SP1ExtensionField>();
        eval_point.add_dimension_back(last_coordinate);
        numerator_eval = round_proof.numerator_0
            + (round_proof.numerator_1 - round_proof.numerator_0) * last_coordinate;
        denominator_eval = round_proof.denominator_0
            + (round_proof.denominator_1 - round_proof.denominator_0) * last_coordinate;
    }
    let mut trace_point_residuals = Vec::new();
    let mut final_num_residual = ext_zero();
    let mut final_den_residual = ext_zero();
    if eval_point.dimension() > number_of_interaction_variables as usize {
        let (interaction_point, trace_point): (Point<SP1ExtensionField>, Point<SP1ExtensionField>) =
            eval_point.split_at(number_of_interaction_variables as usize);
        let LogUpEvaluations { point, chip_openings } = &logup.logup_evaluations;
        for (lhs, rhs) in point.iter().zip(trace_point.iter()) {
            trace_point_residuals.push(*lhs - *rhs);
        }

        let betas = partial_lagrange_blocking(&beta_seed);
        let mut numerator_values = Vec::with_capacity(num_of_interactions);
        let mut denominator_values = Vec::with_capacity(num_of_interactions);
        let mut point_extended = point.clone();
        point_extended.add_dimension(SP1ExtensionField::zero());
        challenger.observe(SP1Field::from_canonical_usize(shard_chips.len()));
        for ((chip, openings), threshold) in
            shard_chips.iter().zip_eq(chip_openings.values()).zip_eq(degrees.values())
        {
            if let Some(prep_eval) = openings.preprocessed_trace_evaluations.as_ref() {
                for &value in prep_eval.iter() {
                    challenger.observe_ext_element(value);
                }
            }
            for &value in openings.main_trace_evaluations.iter() {
                challenger.observe_ext_element(value);
            }
            let geq_eval = full_geq(threshold, &point_extended);
            let sp1_hypercube::ChipEvaluation {
                main_trace_evaluations,
                preprocessed_trace_evaluations,
            } = openings;
            for (interaction, is_send) in chip
                .sends()
                .iter()
                .map(|s| (s, true))
                .chain(chip.receives().iter().map(|r| (r, false)))
            {
                let (real_numerator, real_denominator) = interaction.eval(
                    preprocessed_trace_evaluations.as_ref(),
                    main_trace_evaluations,
                    alpha,
                    betas.as_slice(),
                );
                let padding_trace_opening = MleEval::from(vec![
                    SP1ExtensionField::zero();
                    main_trace_evaluations
                        .num_polynomials()
                ]);
                let padding_preprocessed_opening =
                    preprocessed_trace_evaluations.as_ref().map(|eval| {
                        MleEval::from(vec![SP1ExtensionField::zero(); eval.num_polynomials()])
                    });
                let (padding_numerator, padding_denominator) = interaction.eval(
                    padding_preprocessed_opening.as_ref(),
                    &padding_trace_opening,
                    alpha,
                    betas.as_slice(),
                );
                let numerator_eval = real_numerator - padding_numerator * geq_eval;
                let denominator_eval =
                    real_denominator + (SP1ExtensionField::one() - padding_denominator) * geq_eval;
                let numerator_eval = if is_send { numerator_eval } else { -numerator_eval };
                numerator_values.push(numerator_eval);
                denominator_values.push(denominator_eval);
            }
        }
        numerator_values
            .resize(1usize << (interaction_point.dimension() as usize), SP1ExtensionField::zero());
        denominator_values
            .resize(1usize << (interaction_point.dimension() as usize), SP1ExtensionField::one());
        let expected_numerator_eval =
            Mle::from(numerator_values).blocking_eval_at(&interaction_point)[0];
        let expected_denominator_eval =
            Mle::from(denominator_values).blocking_eval_at(&interaction_point)[0];
        final_num_residual = numerator_eval - expected_numerator_eval;
        final_den_residual = denominator_eval - expected_denominator_eval;
    }

    for descriptor in descriptors {
        match descriptor {
            MultiplicativeResidualDescriptor::Explicit => {
                return Err(anyhow!(
                    "explicit multiplicative descriptors are not supported by the SP1 recursion exporter"
                ));
            }
            MultiplicativeResidualDescriptor::DegreeBitBooleanity { chip_name, bit_idx } => {
                let openings = shard_proof
                    .opened_values
                    .chips
                    .get(&chip_name)
                    .ok_or_else(|| anyhow!("missing chip openings for {}", chip_name))?;
                let bit = *openings.degree.get(bit_idx).ok_or_else(|| {
                    anyhow!("degree bit idx {} out of range for {}", bit_idx, chip_name)
                })?;
                let bit_ext = base_to_ext(bit);
                push_mul_residual(
                    &mut mul_terms,
                    bit_ext,
                    bit_ext - ext_one(),
                    ext_zero(),
                    ext_one(),
                );
            }
            MultiplicativeResidualDescriptor::DegreeHeightProduct { chip_name, bit_idx } => {
                let openings = shard_proof
                    .opened_values
                    .chips
                    .get(&chip_name)
                    .ok_or_else(|| anyhow!("missing chip openings for {}", chip_name))?;
                let first = openings.degree.first().copied().unwrap_or(SP1Field::zero());
                let bit = *openings.degree.get(bit_idx).ok_or_else(|| {
                    anyhow!("degree bit idx {} out of range for {}", bit_idx, chip_name)
                })?;
                push_mul_residual(
                    &mut mul_terms,
                    base_to_ext(bit),
                    base_to_ext(first),
                    ext_zero(),
                    ext_one(),
                );
            }
            MultiplicativeResidualDescriptor::GkrPowWitness => {
                if pow_residual {
                    push_mul_residual(&mut mul_terms, ext_one(), ext_one(), ext_zero(), ext_one());
                } else {
                    push_mul_residual(&mut mul_terms, ext_zero(), ext_one(), ext_zero(), ext_one());
                }
            }
            MultiplicativeResidualDescriptor::GkrCumulativeSum => {
                push_mul_residual(
                    &mut mul_terms,
                    output_cumulative_sum - cumulative_sum,
                    ext_one(),
                    ext_zero(),
                    ext_one(),
                );
            }
            MultiplicativeResidualDescriptor::GkrDenominatorInverse { index } => {
                let d = *denominator
                    .guts()
                    .as_slice()
                    .get(index)
                    .ok_or_else(|| anyhow!("denominator inverse idx {} out of range", index))?;
                let inv = *denominator_inverses.get(index).ok_or_else(|| {
                    anyhow!("denominator inverse cache idx {} out of range", index)
                })?;
                push_mul_residual(&mut mul_terms, d, inv, ext_one(), ext_one());
            }
            MultiplicativeResidualDescriptor::GkrRoundClaimedSum { round_idx } => {
                let residual = *round_claim_residuals
                    .get(round_idx)
                    .ok_or_else(|| anyhow!("round claim idx {} out of range", round_idx))?;
                push_mul_residual(&mut mul_terms, residual, ext_one(), ext_zero(), ext_one());
            }
            MultiplicativeResidualDescriptor::GkrRoundFinalEval { round_idx } => {
                let (eq_eval, combined, final_eval) = *round_final_evals
                    .get(round_idx)
                    .ok_or_else(|| anyhow!("round final-eval idx {} out of range", round_idx))?;
                push_mul_residual(&mut mul_terms, eq_eval, combined, final_eval, ext_one());
            }
            MultiplicativeResidualDescriptor::GkrTracePointCoord { coord_idx } => {
                let residual = *trace_point_residuals
                    .get(coord_idx)
                    .ok_or_else(|| anyhow!("trace-point coord idx {} out of range", coord_idx))?;
                push_mul_residual(&mut mul_terms, residual, ext_one(), ext_zero(), ext_one());
            }
            MultiplicativeResidualDescriptor::GkrFinalNumeratorEval => {
                push_mul_residual(
                    &mut mul_terms,
                    final_num_residual,
                    ext_one(),
                    ext_zero(),
                    ext_one(),
                );
            }
            MultiplicativeResidualDescriptor::GkrFinalDenominatorEval => {
                push_mul_residual(
                    &mut mul_terms,
                    final_den_residual,
                    ext_one(),
                    ext_zero(),
                    ext_one(),
                );
            }
        }
    }

    Ok(mul_terms)
}

pub fn build_sp1_germ_bridge(
    proof: &ShardProof<SP1GlobalContext, SP1PcsProofInner>,
) -> Result<Sp1GermBridge> {
    let mut public_instance_tables = Vec::new();
    let mut shared_object_tables = Vec::new();
    let mut rlin_tables = Vec::new();
    let mut rmul_tables = Vec::new();
    let mut blob_tables = Vec::new();
    let mut tables = Vec::new();

    let bridge_version = vec![1u32, 0, 0, 0];
    let version_table = "bridge/version".to_string();
    push_single_row_table(&mut tables, version_table.clone(), bridge_version)?;
    public_instance_tables.push(version_table);

    let proof_public_values =
        proof.public_values.iter().map(|x| x.as_canonical_u32()).collect::<Vec<_>>();
    let proof_pv_table = "bridge/proof/public_values".to_string();
    push_single_row_table(&mut tables, proof_pv_table.clone(), proof_public_values)?;
    public_instance_tables.push(proof_pv_table);

    // Export per-chip opened values used by zerocheck and PCS opening checks.
    for (chip_name, opened) in &proof.opened_values.chips {
        let chip = sanitize_table_component(chip_name);
        let main_values = flatten_ext_slice_to_u32(&opened.main.local);
        let opened_main_table = format!("bridge/opened/{chip}/main");
        push_single_row_table(&mut tables, opened_main_table.clone(), main_values)?;
        shared_object_tables.push(opened_main_table.clone());
        rlin_tables.push(opened_main_table);

        if !opened.preprocessed.local.is_empty() {
            let prep_values = flatten_ext_slice_to_u32(&opened.preprocessed.local);
            let opened_pre_table = format!("bridge/opened/{chip}/pre");
            push_single_row_table(&mut tables, opened_pre_table.clone(), prep_values)?;
            shared_object_tables.push(opened_pre_table.clone());
            rlin_tables.push(opened_pre_table);
        }

        let degree_bits =
            opened.degree.iter().map(|bit| bit.as_canonical_u32()).collect::<Vec<_>>();
        let degree_bits_table = format!("bridge/opened/{chip}/degree_bits");
        push_single_row_table(&mut tables, degree_bits_table.clone(), degree_bits)?;
        shared_object_tables.push(degree_bits_table.clone());
        rlin_tables.push(degree_bits_table);
    }

    let (zc_point_rows, zc_point_cols, zc_point_values) =
        point_ext_to_u32_values(&proof.zerocheck_proof.point_and_eval.0);
    let zc_point_table = "bridge/zerocheck/point".to_string();
    push_table(&mut tables, zc_point_table.clone(), zc_point_rows, zc_point_cols, zc_point_values)?;
    shared_object_tables.push(zc_point_table.clone());
    rlin_tables.push(zc_point_table);
    let zc_point_eval_table = "bridge/zerocheck/point_eval".to_string();
    push_single_row_table(
        &mut tables,
        zc_point_eval_table.clone(),
        ext_to_limbs_u32(&proof.zerocheck_proof.point_and_eval.1).to_vec(),
    )?;
    shared_object_tables.push(zc_point_eval_table.clone());
    rlin_tables.push(zc_point_eval_table);
    let zc_claimed_sum_table = "bridge/zerocheck/claimed_sum".to_string();
    push_single_row_table(
        &mut tables,
        zc_claimed_sum_table.clone(),
        ext_to_limbs_u32(&proof.zerocheck_proof.claimed_sum).to_vec(),
    )?;
    shared_object_tables.push(zc_claimed_sum_table.clone());
    rlin_tables.push(zc_claimed_sum_table);

    // Export all logup/GKR artifacts needed by the interaction verifier.
    let logup = &proof.logup_gkr_proof;
    let (gkr_point_rows, gkr_point_cols, gkr_point_values) =
        point_ext_to_u32_values(&logup.logup_evaluations.point);
    let gkr_point_table = "bridge/logup/point".to_string();
    push_table(
        &mut tables,
        gkr_point_table.clone(),
        gkr_point_rows,
        gkr_point_cols,
        gkr_point_values,
    )?;
    shared_object_tables.push(gkr_point_table.clone());
    rmul_tables.push(gkr_point_table);

    let (num_rows, num_cols, num_values) = mle_ext_to_u32_values(&logup.circuit_output.numerator);
    let gkr_num_table = "bridge/logup/circuit_output/numerator".to_string();
    push_table(&mut tables, gkr_num_table.clone(), num_rows, num_cols, num_values)?;
    shared_object_tables.push(gkr_num_table.clone());
    rmul_tables.push(gkr_num_table);
    let (den_rows, den_cols, den_values) = mle_ext_to_u32_values(&logup.circuit_output.denominator);
    let gkr_den_table = "bridge/logup/circuit_output/denominator".to_string();
    push_table(&mut tables, gkr_den_table.clone(), den_rows, den_cols, den_values)?;
    shared_object_tables.push(gkr_den_table.clone());
    rmul_tables.push(gkr_den_table);

    for (chip_name, evals) in &logup.logup_evaluations.chip_openings {
        let chip = sanitize_table_component(chip_name);
        let (rows, cols, values) = mle_eval_ext_to_u32_values(&evals.main_trace_evaluations);
        let gkr_chip_main_table = format!("bridge/logup/chip_openings/{chip}/main_eval");
        push_table(&mut tables, gkr_chip_main_table.clone(), rows, cols, values)?;
        shared_object_tables.push(gkr_chip_main_table.clone());
        rmul_tables.push(gkr_chip_main_table);

        if let Some(pre_eval) = evals.preprocessed_trace_evaluations.as_ref() {
            let (rows, cols, values) = mle_eval_ext_to_u32_values(pre_eval);
            let gkr_chip_pre_table = format!("bridge/logup/chip_openings/{chip}/pre_eval");
            push_table(&mut tables, gkr_chip_pre_table.clone(), rows, cols, values)?;
            shared_object_tables.push(gkr_chip_pre_table.clone());
            rmul_tables.push(gkr_chip_pre_table);
        }
    }

    for (round_idx, round) in logup.round_proofs.iter().enumerate() {
        let mut quad = Vec::with_capacity(16);
        quad.extend(ext_to_limbs_u32(&round.numerator_0));
        quad.extend(ext_to_limbs_u32(&round.numerator_1));
        quad.extend(ext_to_limbs_u32(&round.denominator_0));
        quad.extend(ext_to_limbs_u32(&round.denominator_1));
        let round_quad_table = format!("bridge/logup/round/{round_idx}/quad");
        push_single_row_table(&mut tables, round_quad_table.clone(), quad)?;
        shared_object_tables.push(round_quad_table.clone());
        rmul_tables.push(round_quad_table);
        let round_sumcheck_table = format!("bridge/logup/round/{round_idx}/sumcheck_blob");
        push_blob_table(&mut tables, round_sumcheck_table.clone(), &round.sumcheck_proof)?;
        shared_object_tables.push(round_sumcheck_table.clone());
        rmul_tables.push(round_sumcheck_table.clone());
        blob_tables.push(round_sumcheck_table);
    }

    // Export raw proof objects so downstream importers can replay exact verifier logic.
    let main_commitment_blob = "bridge/blob/main_commitment".to_string();
    push_blob_table(&mut tables, main_commitment_blob.clone(), &proof.main_commitment)?;
    shared_object_tables.push(main_commitment_blob.clone());
    rlin_tables.push(main_commitment_blob.clone());
    rmul_tables.push(main_commitment_blob.clone());
    blob_tables.push(main_commitment_blob);

    let logup_blob = "bridge/blob/logup_gkr_proof".to_string();
    push_blob_table(&mut tables, logup_blob.clone(), &proof.logup_gkr_proof)?;
    shared_object_tables.push(logup_blob.clone());
    rmul_tables.push(logup_blob.clone());
    blob_tables.push(logup_blob);

    let zerocheck_blob = "bridge/blob/zerocheck_proof".to_string();
    push_blob_table(&mut tables, zerocheck_blob.clone(), &proof.zerocheck_proof)?;
    shared_object_tables.push(zerocheck_blob.clone());
    rlin_tables.push(zerocheck_blob.clone());
    blob_tables.push(zerocheck_blob);

    let eval_blob = "bridge/blob/evaluation_proof".to_string();
    push_blob_table(&mut tables, eval_blob.clone(), &proof.evaluation_proof)?;
    shared_object_tables.push(eval_blob.clone());
    rlin_tables.push(eval_blob.clone());
    blob_tables.push(eval_blob);

    let manifest = BridgeManifest {
        schema: "sp1-hypercube-germ-bridge".to_string(),
        schema_version: 1,
        public_instance_tables,
        shared_object_tables,
        rlin_tables,
        rmul_tables,
        blob_tables,
    };
    let manifest_json =
        serde_json::to_vec(&manifest).context("serialize bridge manifest json table")?;
    let manifest_words = bytes_to_len_prefixed_u32_words(&manifest_json)?;
    push_single_row_table(&mut tables, "bridge/manifest/json".to_string(), manifest_words)?;

    Ok(Sp1GermBridge { manifest, tables })
}

pub fn build_sp1_germ_bridge_from_recursion_proof(
    proof: &SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
) -> Result<Sp1GermBridge> {
    build_sp1_germ_bridge(&proof.proof)
}

pub fn build_sp1_germ_proof_object_from_recursion_proof(
    proof: &SP1RecursionProof<SP1GlobalContext, SP1PcsProofInner>,
    capsule: &GermArmCapsule,
) -> Result<(Sp1GermProofObject, [u8; 32])> {
    let bridge = build_sp1_germ_bridge_from_recursion_proof(proof)?;
    let shared_object = bincode::serialize(&bridge).context("serialize SP1 GERM bridge")?;

    // Keep the extraction centralized in the prover layer so examples consume a canonical
    // proof-object exporter derived from the actual recursion verifier equations.
    let mut lin_terms = export_linear_residual_terms_from_recursion_proof(proof)
        .context("export linear residual terms from recursion proof")?;
    if let Some((idx, term)) =
        lin_terms.iter().enumerate().find(|(_, term)| !is_zero_ext(&term.value))
    {
        anyhow::bail!(
            "first exported linear residual is nonzero at index {}: {:?}",
            idx,
            term.value
        );
    }
    if lin_terms.is_empty() {
        lin_terms.push(Sp1LinTerm::new(ext_one(), ext_zero()));
    }

    let mut mul_terms = export_multiplicative_residual_terms_from_recursion_proof(proof)
        .context("export multiplicative residual terms from recursion proof")?;
    if let Some((idx, _term)) =
        mul_terms.iter().enumerate().find(|(_, term)| (term.a * term.b) != (term.c * term.d))
    {
        anyhow::bail!("first exported multiplicative residual is nonzero at index {}", idx);
    }
    if mul_terms.is_empty() {
        mul_terms.push(Sp1MulTerm::new(ext_zero(), ext_one(), ext_zero(), ext_one()));
    }
    let target_mul_terms = 1usize << usize::from(sp1_germ_residual_plan()?.sumcheck_rounds);
    if mul_terms.len() > target_mul_terms {
        anyhow::bail!(
            "SP1 GERM multiplicative term exporter exceeded fixed schedule: got {} > {}",
            mul_terms.len(),
            target_mul_terms
        );
    }

    let mut proof_object =
        Sp1GermProofObject::new(shared_object, lin_terms, mul_terms, Vec::new(), Vec::new());
    let (commitment_root, _) = bind_bundle_to_capsule(&mut proof_object, capsule)
        .map_err(|err| anyhow!("bind SP1 GERM proof object: {err}"))?;
    let expected_commitment_root = compute_commitment_root(&proof_object.shared_object_commitment);
    anyhow::ensure!(
        expected_commitment_root == commitment_root,
        "commitment root drift while exporting SP1 GERM proof object"
    );
    Ok((proof_object, commitment_root))
}

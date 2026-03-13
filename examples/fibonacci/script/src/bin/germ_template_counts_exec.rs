use std::collections::BTreeMap;

use sp1_germ::{
    bind_bundle_to_capsule_with_orbweaver_terminal_openings, compile_germ_aadp_template,
    compile_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings, generate_local_dev_srs,
    AadpField, GermPublicValues, MultiplicativeResidualDescriptor, Sp1AadpField, Sp1GermBundle,
    Sp1LinTerm, Sp1MulTerm, TranscriptBoundSp1GermProofObject,
};
use sp1_prover::germ_bridge::{sp1_germ_residual_plan, sp1_germ_schedule_descriptor_digest};

fn ext_zero() -> Sp1AadpField {
    <Sp1AadpField as AadpField>::from_u128(0)
}

fn ext_one() -> Sp1AadpField {
    <Sp1AadpField as AadpField>::from_u128(1)
}

fn linear_descriptor_term_count(
    descriptor: &sp1_germ::LinearResidualDescriptor,
) -> usize {
    match descriptor {
        sp1_germ::LinearResidualDescriptor::PublicValuesPadding { count, .. } => *count,
        _ => 1,
    }
}

fn main() {
    let residual_plan = sp1_germ_residual_plan().expect("derive SP1 GERM residual plan");
    let public_values = GermPublicValues {
        statement_digest: [0u8; 32],
        descriptor_digest: sp1_germ_schedule_descriptor_digest()
            .expect("derive SP1 GERM schedule descriptor"),
        verifier_shape_digest: [0u8; 32],
        share_index: 0,
        share_domain_separator: [0u8; 32],
    };
    let capsule = public_values.arm_capsule(&residual_plan);
    let generic_template =
        compile_germ_aadp_template(&capsule, &residual_plan).expect("compile generic AADP template");
    let orbweaver_enabled = std::env::var_os("SP1_GERM_USE_LOCAL_ORBWEAVER_SRS").is_some();
    let template = if orbweaver_enabled {
        let width = (1usize << usize::from(residual_plan.sumcheck_rounds)) * 16;
        let srs = generate_local_dev_srs(width);
        let expected_linear_terms = residual_plan
            .linear_descriptors
            .iter()
            .map(linear_descriptor_term_count)
            .sum::<usize>();
        let expected_mul_terms = 1usize << usize::from(residual_plan.sumcheck_rounds);
        let lin_terms = vec![Sp1LinTerm::new(ext_one(), ext_zero()); expected_linear_terms];
        let mul_terms =
            vec![Sp1MulTerm::new(ext_zero(), ext_one(), ext_zero(), ext_one()); expected_mul_terms];
        let mut bundle =
            Sp1GermBundle::new(b"orbweaver-counts".to_vec(), lin_terms, mul_terms, Vec::new(), Vec::new());
        let (commitment_root, _) = bind_bundle_to_capsule_with_orbweaver_terminal_openings(
            &mut bundle,
            &capsule,
            &srs,
        )
        .expect("bind trivial bundle with Orbweaver openings");
        let transcript_bound = TranscriptBoundSp1GermProofObject::new(bundle, commitment_root);
        compile_transcript_bound_germ_aadp_template_with_orbweaver_terminal_openings(
            &capsule,
            &residual_plan,
            &transcript_bound,
            &srs,
        )
        .expect("compile transcript-bound AADP template with Orbweaver terminal openings")
    } else {
        generic_template.clone()
    };

    let mut multiplicative_variant_counts = BTreeMap::<&'static str, usize>::new();
    for descriptor in &residual_plan.multiplicative_descriptors {
        let label = match descriptor {
            MultiplicativeResidualDescriptor::Explicit => "Explicit",
            MultiplicativeResidualDescriptor::DegreeBitBooleanity { .. } => "DegreeBitBooleanity",
            MultiplicativeResidualDescriptor::DegreeHeightProduct { .. } => "DegreeHeightProduct",
            MultiplicativeResidualDescriptor::GkrPowWitness => "GkrPowWitness",
            MultiplicativeResidualDescriptor::GkrCumulativeSum => "GkrCumulativeSum",
            MultiplicativeResidualDescriptor::GkrDenominatorInverse { .. } => "GkrDenominatorInverse",
            MultiplicativeResidualDescriptor::GkrRoundClaimedSum { .. } => "GkrRoundClaimedSum",
            MultiplicativeResidualDescriptor::GkrRoundFinalEval { .. } => "GkrRoundFinalEval",
            MultiplicativeResidualDescriptor::GkrTracePointCoord { .. } => "GkrTracePointCoord",
            MultiplicativeResidualDescriptor::GkrFinalNumeratorEval => "GkrFinalNumeratorEval",
            MultiplicativeResidualDescriptor::GkrFinalDenominatorEval => "GkrFinalDenominatorEval",
        };
        *multiplicative_variant_counts.entry(label).or_default() += 1;
    }

    println!("aadp_vars={}", template.cs.num_variables);
    println!("aadp_constraints={}", template.cs.constraints.len());
    println!("lin_checks={}", template.stats.linear_round_checks);
    println!("opening_checks={}", template.stats.opening_checks);
    println!("mul_gates={}", template.stats.multiplication_gates);
    println!("generic_aadp_vars={}", generic_template.cs.num_variables);
    println!("generic_aadp_constraints={}", generic_template.cs.constraints.len());
    println!("generic_lin_checks={}", generic_template.stats.linear_round_checks);
    println!("generic_opening_checks={}", generic_template.stats.opening_checks);
    println!("generic_mul_gates={}", generic_template.stats.multiplication_gates);
    println!("orbweaver_enabled={orbweaver_enabled}");
    println!(
        "orbweaver_aadp_var_delta={}",
        template.cs.num_variables as isize - generic_template.cs.num_variables as isize
    );
    println!(
        "orbweaver_aadp_constraint_delta={}",
        template.cs.constraints.len() as isize - generic_template.cs.constraints.len() as isize
    );
    println!(
        "orbweaver_lin_check_delta={}",
        template.stats.linear_round_checks as isize - generic_template.stats.linear_round_checks as isize
    );
    println!(
        "orbweaver_opening_check_delta={}",
        template.stats.opening_checks as isize - generic_template.stats.opening_checks as isize
    );
    println!(
        "orbweaver_mul_gate_delta={}",
        template.stats.multiplication_gates as isize - generic_template.stats.multiplication_gates as isize
    );
    println!("sumcheck_rounds={}", residual_plan.sumcheck_rounds);
    println!("linear_descriptors={}", residual_plan.linear_descriptors.len());
    println!("multiplicative_descriptors={}", residual_plan.multiplicative_descriptors.len());
    for (label, count) in multiplicative_variant_counts {
        println!("mul_variant_{label}={count}");
    }
}

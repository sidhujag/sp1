use std::collections::BTreeMap;

use sp1_germ::{compile_germ_aadp_template, GermPublicValues, MultiplicativeResidualDescriptor};
use sp1_prover::germ_bridge::{sp1_germ_residual_plan, sp1_germ_schedule_descriptor_digest};

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
    let template =
        compile_germ_aadp_template(&capsule, &residual_plan).expect("compile AADP template");

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
    println!("sumcheck_rounds={}", residual_plan.sumcheck_rounds);
    println!("linear_descriptors={}", residual_plan.linear_descriptors.len());
    println!(
        "multiplicative_descriptors={}",
        residual_plan.multiplicative_descriptors.len()
    );
    for (label, count) in multiplicative_variant_counts {
        println!("mul_variant_{label}={count}");
    }
}

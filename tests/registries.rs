//! Integration tests for name-selected implementations and error reporting.

mod common;

use hale::config::ModelConfig;
use hale::generate::{sampler_registry, SamplingParams};
use hale::model::arch::{naming_registry, MlpProj};
use hale::HaleTokenError;

#[test]
fn every_fixture_architecture_resolves_a_tensor_naming() {
    for name in common::FIXTURES {
        let config = ModelConfig::from_dir(&common::fixture(name)).unwrap();
        let naming = naming_registry()
            .build(config.architecture.hf_name(), &())
            .unwrap();
        assert!(naming.expert(0, 0, MlpProj::Gate).contains("experts.0"));
    }
}

#[test]
fn unknown_names_suggest_the_closest_registered_one() {
    let err = match naming_registry().build("MixtralForCausalLm", &()) {
        Ok(_) => panic!("typo must not resolve"),
        Err(e) => e,
    };
    assert!(matches!(err, HaleTokenError::UnknownRegistryKey { .. }));
    assert!(err.to_string().contains("MixtralForCausalLM"), "{err}");
}

#[test]
fn registered_samplers_are_deterministic_for_a_fixed_seed() {
    let params = SamplingParams {
        temperature: 1.0,
        ..SamplingParams::default()
    };
    let logits = [1.0, 2.0, 3.0, 0.5];
    let mut a = sampler_registry().build("top_p", &params).unwrap();
    let mut b = sampler_registry().build("top_p", &params).unwrap();
    for _ in 0..50 {
        assert_eq!(a.sample(&logits), b.sample(&logits));
    }
}

//! End-to-end tests of the `hale` binary on the tiny fixtures.

mod common;

use std::process::Command;

use common::fixture;

fn hale(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_hale"))
        .args(args)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn info_describes_the_model() {
    let dir = fixture("tiny-qwen3-moe");
    let (ok, out, err) = hale(&["info", dir.to_str().unwrap()]);
    assert!(ok, "{err}");
    assert!(out.contains("Qwen3Moe"), "{out}");
    assert!(out.contains("8 per layer, top-3 routing"), "{out}");
    assert!(out.contains("2 MoE, 1 dense"), "{out}");
}

#[test]
fn convert_then_run_generates_text_and_stats() {
    let out_dir = tempfile::tempdir().unwrap();
    let model = out_dir.path().join("model");
    let src = fixture("tiny-mixtral");
    let (ok, _, err) = hale(&[
        "convert",
        src.to_str().unwrap(),
        model.to_str().unwrap(),
        "--expert-dtype",
        "q8_0",
    ]);
    assert!(ok, "{err}");
    for file in [
        "config.json",
        "tokenizer.json",
        "dense.safetensors",
        "experts.hpk",
    ] {
        assert!(model.join(file).exists(), "missing {file}");
    }

    let (ok, out, err) = hale(&[
        "run",
        model.to_str().unwrap(),
        "--prompt",
        "hello world",
        "--temperature",
        "0",
        "-n",
        "6",
        "--expert-ram-gb",
        "0.00001", // 10 KB: room for one ~6.5 KB tiny expert
        "--json-stats",
    ]);
    assert!(ok, "{err}");
    assert!(err.contains("SSD pack"), "{err}");
    let json_line = out.lines().last().unwrap();
    let stats: serde_json::Value = serde_json::from_str(json_line).unwrap();
    assert!(stats["generated_tokens"].as_u64().unwrap() >= 1);
    assert!(stats["cache"]["misses"].as_u64().unwrap() > 0);
    assert!(
        model.join("hale-routing.json").exists(),
        "routing stats saved"
    );
}

#[test]
fn plan_estimates_and_compares() {
    let (ok, out, err) = hale(&[
        "plan",
        "--preset",
        "qwen3-235b",
        "--chip",
        "M4 Max",
        "--ram-gb",
        "64",
    ]);
    assert!(ok, "{err}");
    assert!(out.contains("streamed from SSD"), "{out}");
    assert!(out.contains("tokens/s"), "{out}");

    let (ok, out, _) = hale(&["plan", "--compare"]);
    assert!(ok);
    assert!(out.contains("DeepSeek-V3") && out.contains("M5"), "{out}");
}

#[test]
fn helpful_errors_for_bad_input() {
    let (ok, _, err) = hale(&["info", "/definitely/not/here"]);
    assert!(!ok);
    assert!(err.contains("error:"), "{err}");
    let (ok, _, err) = hale(&["plan", "--preset", "gpt-9"]);
    assert!(!ok);
    assert!(err.contains("unknown preset"), "{err}");
}

#[test]
fn sysinfo_and_kernel_bench_run() {
    let (ok, out, err) = hale(&["sysinfo"]);
    assert!(ok, "{err}");
    assert!(out.contains("cpu"), "{out}");
    let (ok, out, err) = hale(&["bench", "kernels"]);
    assert!(ok, "{err}");
    assert!(out.contains("q4_0"), "{out}");
}

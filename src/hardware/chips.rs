//! Published specifications of every Apple Silicon Mac chip (M1 - M5).
//!
//! Bandwidths are Apple's advertised unified-memory figures in GB/s; core
//! counts are the full (unbinned) configuration. Binned parts are a little
//! slower - `hale bench memory` measures the truth on your machine.

/// Static facts about one chip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChipSpec {
    /// Marketing name, e.g. `"M3 Max"`.
    pub name: &'static str,
    /// Advertised unified-memory bandwidth, GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Performance cores.
    pub performance_cores: u32,
    /// Efficiency cores.
    pub efficiency_cores: u32,
    /// Largest unified-memory configuration sold, GB.
    pub max_memory_gb: u32,
}

const fn chip(name: &'static str, bw: f64, p: u32, e: u32, mem: u32) -> ChipSpec {
    ChipSpec {
        name,
        memory_bandwidth_gbps: bw,
        performance_cores: p,
        efficiency_cores: e,
        max_memory_gb: mem,
    }
}

/// Every M-series chip, ordered by generation then tier.
pub const APPLE_CHIPS: &[ChipSpec] = &[
    chip("M1", 68.25, 4, 4, 16),
    chip("M1 Pro", 200.0, 8, 2, 32),
    chip("M1 Max", 400.0, 8, 2, 64),
    chip("M1 Ultra", 800.0, 16, 4, 128),
    chip("M2", 100.0, 4, 4, 24),
    chip("M2 Pro", 200.0, 8, 4, 32),
    chip("M2 Max", 400.0, 8, 4, 96),
    chip("M2 Ultra", 800.0, 16, 8, 192),
    chip("M3", 100.0, 4, 4, 24),
    chip("M3 Pro", 150.0, 6, 6, 36),
    chip("M3 Max", 400.0, 12, 4, 128),
    chip("M3 Ultra", 819.0, 24, 8, 512),
    chip("M4", 120.0, 4, 6, 32),
    chip("M4 Pro", 273.0, 10, 4, 64),
    chip("M4 Max", 546.0, 12, 4, 128),
    chip("M5", 153.0, 4, 6, 32),
];

/// Finds a chip by name. Accepts `"M3 Max"`, `"m3-max"`, `"Apple M3 Max"`.
///
/// Longer names are tried first so `"M3 Max"` never matches plain `"M3"`.
pub fn lookup_chip(name: &str) -> Option<&'static ChipSpec> {
    let wanted = normalise(name);
    let wanted = wanted.strip_prefix("apple").unwrap_or(&wanted);
    let mut chips: Vec<&ChipSpec> = APPLE_CHIPS.iter().collect();
    chips.sort_by_key(|c| std::cmp::Reverse(c.name.len()));
    chips.into_iter().find(|c| {
        let n = normalise(c.name);
        wanted == n
            || (wanted.starts_with(&n) && !wanted[n.len()..].starts_with(char::is_alphabetic))
    })
}

fn normalise(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_handles_spellings_and_prefers_longest_match() {
        assert_eq!(lookup_chip("Apple M3 Max").unwrap().name, "M3 Max");
        assert_eq!(lookup_chip("m4-pro").unwrap().name, "M4 Pro");
        assert_eq!(lookup_chip("Apple M1").unwrap().name, "M1");
        assert_eq!(lookup_chip("M5").unwrap().memory_bandwidth_gbps, 153.0);
        assert!(lookup_chip("Intel Core i9").is_none());
    }

    #[test]
    fn every_generation_is_covered() {
        for g in 1..=5 {
            assert!(lookup_chip(&format!("M{g}")).is_some(), "M{g} missing");
        }
    }
}

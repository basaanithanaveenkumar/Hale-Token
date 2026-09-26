//! Human-friendly number formatting shared by the commands.

/// `1_500_000_000` -> `"1.50 GB"` (decimal units, like macOS Finder).
pub fn bytes(n: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = n;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// `30_500_000_000` -> `"30.5B"`.
pub fn params(n: f64) -> String {
    if n >= 1e12 {
        format!("{:.2}T", n / 1e12)
    } else if n >= 1e9 {
        format!("{:.1}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.1}M", n / 1e6)
    } else {
        format!("{n:.0}")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn formats() {
        assert_eq!(super::bytes(512.0), "512 B");
        assert_eq!(super::bytes(1.5e9), "1.50 GB");
        assert_eq!(super::params(30.5e9), "30.5B");
        assert_eq!(super::params(1.04e12), "1.04T");
    }
}

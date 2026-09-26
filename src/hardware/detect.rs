//! Runtime detection of the host machine.

use serde::Serialize;

use super::chips::{lookup_chip, ChipSpec};

/// What we learned about the host.
#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    /// CPU brand string, e.g. `"Apple M2 Pro"`.
    pub cpu_name: String,
    /// Matched Apple chip, if any.
    #[serde(skip)]
    pub chip: Option<&'static ChipSpec>,
    /// Installed RAM in bytes (0 if unknown).
    pub total_memory_bytes: u64,
    /// Logical CPUs available to this process.
    pub logical_cpus: usize,
    /// Performance / efficiency core counts when the OS reports them.
    pub performance_cores: Option<u32>,
    pub efficiency_cores: Option<u32>,
    /// True on an `aarch64` Mac.
    pub is_apple_silicon: bool,
}

impl SystemInfo {
    /// Memory bandwidth from the spec table, or a conservative default.
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        self.chip.map(|c| c.memory_bandwidth_gbps).unwrap_or(50.0)
    }

    /// Short summary line.
    pub fn summary(&self) -> String {
        format!(
            "{} | {:.0} GB RAM | {} CPUs{}",
            self.cpu_name,
            self.total_memory_bytes as f64 / 1e9,
            self.logical_cpus,
            match (self.performance_cores, self.efficiency_cores) {
                (Some(p), Some(e)) => format!(" ({p}P + {e}E)"),
                _ => String::new(),
            }
        )
    }
}

/// Detects the host machine.
pub fn detect() -> SystemInfo {
    let raw = platform::probe();
    let chip = lookup_chip(&raw.cpu_name);
    SystemInfo {
        is_apple_silicon: cfg!(all(target_os = "macos", target_arch = "aarch64")),
        chip,
        cpu_name: raw.cpu_name,
        total_memory_bytes: raw.total_memory_bytes,
        logical_cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        performance_cores: raw.performance_cores,
        efficiency_cores: raw.efficiency_cores,
    }
}

struct RawInfo {
    cpu_name: String,
    total_memory_bytes: u64,
    performance_cores: Option<u32>,
    efficiency_cores: Option<u32>,
}

#[cfg(target_os = "macos")]
mod platform {
    use super::RawInfo;
    use std::ffi::CString;

    /// Reads a string sysctl such as `machdep.cpu.brand_string`.
    fn sysctl_string(name: &str) -> Option<String> {
        let cname = CString::new(name).ok()?;
        let mut len: libc::size_t = 0;
        // SAFETY: first call only queries the length; second call writes at
        // most `len` bytes into a buffer of exactly that size.
        unsafe {
            if libc::sysctlbyname(
                cname.as_ptr(),
                std::ptr::null_mut(),
                &mut len,
                std::ptr::null_mut(),
                0,
            ) != 0
            {
                return None;
            }
            let mut buf = vec![0u8; len];
            if libc::sysctlbyname(
                cname.as_ptr(),
                buf.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            ) != 0
            {
                return None;
            }
            buf.truncate(len);
            let s = String::from_utf8_lossy(&buf)
                .trim_end_matches('\0')
                .trim()
                .to_string();
            Some(s)
        }
    }

    /// Reads an integer sysctl (the kernel returns 4 or 8 bytes).
    fn sysctl_u64(name: &str) -> Option<u64> {
        let cname = CString::new(name).ok()?;
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>() as libc::size_t;
        // SAFETY: `value` is 8 writable bytes and `len` says so.
        let rc = unsafe {
            libc::sysctlbyname(
                cname.as_ptr(),
                (&mut value as *mut u64).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(if len == 4 { value & 0xFFFF_FFFF } else { value })
    }

    pub(super) fn probe() -> RawInfo {
        RawInfo {
            cpu_name: sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
            total_memory_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
            performance_cores: sysctl_u64("hw.perflevel0.physicalcpu").map(|v| v as u32),
            efficiency_cores: sysctl_u64("hw.perflevel1.physicalcpu").map(|v| v as u32),
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::RawInfo;

    pub(super) fn probe() -> RawInfo {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpu_name = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| std::env::consts::ARCH.to_string());
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let total_memory_bytes = meminfo
            .lines()
            .find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0);
        RawInfo {
            cpu_name,
            total_memory_bytes,
            performance_cores: None,
            efficiency_cores: None,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn detect_reports_something_sensible() {
        let info = super::detect();
        assert!(info.logical_cpus >= 1);
        assert!(!info.cpu_name.is_empty());
        assert!(info.memory_bandwidth_gbps() > 0.0);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            assert!(info.is_apple_silicon);
            assert!(
                info.chip.is_some(),
                "unrecognised Apple chip: {}",
                info.cpu_name
            );
        }
    }
}

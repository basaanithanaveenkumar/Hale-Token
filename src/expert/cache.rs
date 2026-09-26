//! The tiered expert cache: pinned RAM -> LRU RAM -> SSD.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use super::{Expert, ExpertKey, ExpertProvider, ExpertSource, LruCache, RoutingStats};
use crate::error::Result;

/// Counters describing how well the cache is doing.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CacheStats {
    /// Requests served from the pinned tier.
    pub pinned_hits: u64,
    /// Requests served from the LRU tier.
    pub lru_hits: u64,
    /// Requests that had to be loaded from the source.
    pub misses: u64,
    /// Bytes loaded from the source.
    pub bytes_loaded: u64,
    /// Wall-clock seconds spent loading from the source.
    pub load_seconds: f64,
    pub pinned_experts: usize,
    pub pinned_bytes: usize,
    pub lru_experts: usize,
    pub lru_bytes: usize,
    pub lru_capacity_bytes: usize,
    /// True when the source is memory-mapped and caching is left to the OS.
    pub passthrough: bool,
}

impl CacheStats {
    /// Fraction of requests served from RAM tiers (0 when nothing was requested).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.pinned_hits + self.lru_hits;
        let total = hits + self.misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Observed source read throughput in GB/s.
    pub fn load_gbps(&self) -> f64 {
        if self.load_seconds > 0.0 {
            self.bytes_loaded as f64 / self.load_seconds / 1e9
        } else {
            0.0
        }
    }
}

/// How much RAM each tier may use.
#[derive(Debug, Clone, Default)]
pub struct CachePolicy {
    /// Experts to load at start-up and never evict (tier 0).
    pub pinned: Vec<ExpertKey>,
    /// Byte budget of the LRU tier (tier 1).
    pub lru_bytes: usize,
}

/// Tiered [`ExpertProvider`]. See the [module docs](super) for the big picture.
pub struct ExpertCache {
    source: Arc<dyn ExpertSource>,
    pinned: HashMap<ExpertKey, Arc<Expert>>,
    lru: Mutex<LruCache<ExpertKey, Arc<Expert>>>,
    routing: Mutex<RoutingStats>,
    stats: Mutex<CacheStats>,
    passthrough: bool,
}

impl ExpertCache {
    /// Builds the cache and loads the pinned tier (in parallel).
    ///
    /// When the source is memory-mapped (`loads_into_ram() == false`) the
    /// policy is ignored: experts are served straight from the map and the
    /// OS page cache does the caching.
    pub fn new(
        source: Arc<dyn ExpertSource>,
        routing: RoutingStats,
        policy: CachePolicy,
    ) -> Result<Self> {
        let passthrough = !source.loads_into_ram();
        let (pinned_keys, lru_bytes) = if passthrough {
            (Vec::new(), 0)
        } else {
            (policy.pinned, policy.lru_bytes)
        };

        let pinned: HashMap<ExpertKey, Arc<Expert>> = pinned_keys
            .par_iter()
            .map(|&k| source.load(k).map(|e| (k, Arc::new(e))))
            .collect::<Result<_>>()?;
        let pinned_bytes = pinned.values().map(|e| e.size_bytes()).sum();

        let stats = CacheStats {
            pinned_experts: pinned.len(),
            pinned_bytes,
            lru_capacity_bytes: lru_bytes,
            passthrough,
            ..CacheStats::default()
        };
        Ok(ExpertCache {
            source,
            pinned,
            lru: Mutex::new(LruCache::new(lru_bytes)),
            routing: Mutex::new(routing),
            stats: Mutex::new(stats),
            passthrough,
        })
    }

    /// The underlying cold source.
    pub fn source(&self) -> &Arc<dyn ExpertSource> {
        &self.source
    }
}

impl ExpertProvider for ExpertCache {
    fn fetch(&self, keys: &[ExpertKey]) -> Result<Vec<Arc<Expert>>> {
        {
            let mut routing = self.routing.lock().expect("routing lock");
            keys.iter().for_each(|k| routing.record(*k));
        }

        // Pass 1: serve what we can from RAM, remember the misses.
        let mut found: Vec<Option<Arc<Expert>>> = Vec::with_capacity(keys.len());
        let (mut pinned_hits, mut lru_hits) = (0, 0);
        {
            let mut lru = self.lru.lock().expect("lru lock");
            for key in keys {
                if let Some(e) = self.pinned.get(key) {
                    pinned_hits += 1;
                    found.push(Some(Arc::clone(e)));
                } else if let Some(e) = lru.get(key) {
                    lru_hits += 1;
                    found.push(Some(e));
                } else {
                    found.push(None);
                }
            }
        }

        // Pass 2: load all misses concurrently (keeps the NVMe queue full).
        // The lock is released during I/O, so two threads missing the same
        // expert would both read it; harmless (the second insert replaces the
        // first) and irrelevant for today's single-sequence engine. A server
        // with concurrent requests should add per-key in-flight tracking.
        let missing: Vec<ExpertKey> = keys
            .iter()
            .zip(&found)
            .filter(|(_, f)| f.is_none())
            .map(|(k, _)| *k)
            .collect();
        let started = Instant::now();
        let loaded: Vec<Arc<Expert>> = missing
            .par_iter()
            .map(|&k| self.source.load(k).map(Arc::new))
            .collect::<Result<_>>()?;
        let load_seconds = started.elapsed().as_secs_f64();

        let mut loaded_iter = loaded.into_iter();
        let mut lru = self.lru.lock().expect("lru lock");
        let result: Vec<Arc<Expert>> = found
            .into_iter()
            .zip(keys)
            .map(|(slot, key)| match slot {
                Some(e) => e,
                None => {
                    let e = loaded_iter.next().expect("one load per miss");
                    if !self.passthrough {
                        lru.insert(*key, Arc::clone(&e), e.size_bytes());
                    }
                    e
                }
            })
            .collect();

        let mut stats = self.stats.lock().expect("stats lock");
        stats.pinned_hits += pinned_hits;
        stats.lru_hits += lru_hits;
        stats.misses += missing.len() as u64;
        if !self.passthrough {
            stats.bytes_loaded += (missing.len() * self.source.expert_bytes()) as u64;
            stats.load_seconds += load_seconds;
        }
        stats.lru_experts = lru.len();
        stats.lru_bytes = lru.used_bytes();
        Ok(result)
    }

    fn stats(&self) -> CacheStats {
        self.stats.lock().expect("stats lock").clone()
    }

    fn routing_stats(&self) -> RoutingStats {
        self.routing.lock().expect("routing lock").clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, WeightMatrix};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fake source that counts loads.
    struct CountingSource {
        loads: AtomicUsize,
        in_ram: bool,
    }

    impl ExpertSource for CountingSource {
        fn load(&self, key: ExpertKey) -> Result<Expert> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            let v = vec![key.expert as f32; 32];
            let m = WeightMatrix::from_f32(1, 32, DType::F32, &v).unwrap();
            Ok(Expert {
                gate: m.clone(),
                up: m.clone(),
                down: m,
            })
        }
        fn expert_bytes(&self) -> usize {
            3 * 128
        }
        fn loads_into_ram(&self) -> bool {
            self.in_ram
        }
        fn describe(&self) -> String {
            "counting".into()
        }
    }

    fn source(in_ram: bool) -> Arc<CountingSource> {
        Arc::new(CountingSource {
            loads: AtomicUsize::new(0),
            in_ram,
        })
    }

    #[test]
    fn tiers_are_consulted_in_order() {
        let src = source(true);
        let policy = CachePolicy {
            pinned: vec![ExpertKey::new(0, 0)],
            lru_bytes: 2 * 384, // room for two experts
        };
        let cache = ExpertCache::new(src.clone(), RoutingStats::new(1, 8), policy).unwrap();
        assert_eq!(
            src.loads.load(Ordering::SeqCst),
            1,
            "pinned expert loaded eagerly"
        );

        let k = |e| ExpertKey::new(0, e);
        let got = cache.fetch(&[k(0), k(1), k(2)]).unwrap();
        assert_eq!(got[2].gate.to_f32()[0], 2.0, "results keep request order");
        cache.fetch(&[k(1), k(2)]).unwrap(); // both LRU hits
        cache.fetch(&[k(3)]).unwrap(); // evicts expert 1 (least recently used)
        cache.fetch(&[k(1)]).unwrap(); // miss again

        let s = cache.stats();
        assert_eq!((s.pinned_hits, s.lru_hits, s.misses), (1, 2, 4));
        assert_eq!(src.loads.load(Ordering::SeqCst), 5);
        assert_eq!(s.lru_experts, 2);
        assert_eq!(cache.routing_stats().count(k(1)), 3);
    }

    #[test]
    fn mmap_sources_bypass_the_ram_tiers() {
        let src = source(false);
        let policy = CachePolicy {
            pinned: vec![ExpertKey::new(0, 0)],
            lru_bytes: 1 << 20,
        };
        let cache = ExpertCache::new(src.clone(), RoutingStats::new(1, 4), policy).unwrap();
        cache.fetch(&[ExpertKey::new(0, 0)]).unwrap();
        cache.fetch(&[ExpertKey::new(0, 0)]).unwrap();
        let s = cache.stats();
        assert!(s.passthrough);
        assert_eq!((s.pinned_experts, s.misses), (0, 2));
    }
}

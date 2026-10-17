//! The tiered expert cache: pinned RAM -> LRU RAM -> SSD.

mod cache_policy;
mod cache_stats;

pub use self::cache_policy::CachePolicy;
pub use self::cache_stats::CacheStats;

use super::{Expert, ExpertKey, ExpertProvider, ExpertSource, LruCache, RoutingStats};
use crate::error::{HaleTokenError, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

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
            let mut routing = lock(&self.routing);
            keys.iter().for_each(|k| routing.record(*k));
        }

        // Pass 1: serve what we can from RAM, remember the misses.
        let mut found: Vec<Option<Arc<Expert>>> = Vec::with_capacity(keys.len());
        let (mut pinned_hits, mut lru_hits) = (0, 0);
        {
            let mut lru = lock(&self.lru);
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
        let mut lru = lock(&self.lru);
        let mut result: Vec<Arc<Expert>> = Vec::with_capacity(keys.len());
        for (slot, key) in found.into_iter().zip(keys) {
            let expert = match slot {
                Some(e) => e,
                None => {
                    // Pass 2 loaded exactly one expert per `None` slot, in order.
                    let e = loaded_iter.next().ok_or_else(|| {
                        HaleTokenError::Internal(format!("no loaded expert for cache miss {key:?}"))
                    })?;
                    if !self.passthrough {
                        lru.insert(*key, Arc::clone(&e), e.size_bytes());
                        tracing::trace!(?key, bytes = e.size_bytes(), "expert cached in LRU tier");
                    }
                    e
                }
            };
            result.push(expert);
        }

        let mut stats = lock(&self.stats);
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
        lock(&self.stats).clone()
    }

    fn routing_stats(&self) -> RoutingStats {
        lock(&self.routing).clone()
    }
}

/// Locks `mutex`, recovering the data if another thread panicked while holding it.
///
/// The guarded values (LRU index, counters) stay structurally valid even if an update was
/// interrupted, so one panicking request must not take every later request down with it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
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

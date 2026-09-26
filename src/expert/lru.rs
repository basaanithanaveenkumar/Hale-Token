//! A byte-budgeted Least-Recently-Used cache with O(1) operations.
//!
//! # Data structure
//!
//! The classic LRU is a hash map plus a doubly linked list:
//!
//! ```text
//!   map: key -> slot index            list (most recent first):
//!   ┌──────┬──────┐                   head                         tail
//!   │ k7   │  2   │ ─────────────┐     │                             │
//!   │ k3   │  0   │ ──────┐      │     ▼                             ▼
//!   └──────┴──────┘       │      └──► [slot 2] ⇄ [slot 0] ⇄ ... ⇄ [slot 5]
//! ```
//!
//! * lookup: hash map -> slot, then unlink the slot and relink it at the head.
//! * insert: link a new slot at the head; while over budget, evict the tail.
//!
//! Instead of heap-allocated list nodes (awkward in safe Rust) the nodes live
//! in a `Vec` ("slab") and link to each other by index. Freed slots are
//! recycled through a free list, so steady-state operation never allocates.
//!
//! Entries are *weighted*: each carries its size in bytes and the cache
//! evicts until the total fits `capacity_bytes`. Experts of one model all
//! have the same size, but weighting keeps the cache honest if they don't.

use std::collections::HashMap;
use std::hash::Hash;

/// Sentinel "null pointer" for slab links.
const NIL: usize = usize::MAX;

struct Node<K, V> {
    key: K,
    value: V,
    weight: usize,
    prev: usize,
    next: usize,
}

/// Byte-budgeted LRU cache. See the module docs for the design.
pub struct LruCache<K, V> {
    map: HashMap<K, usize>,
    slab: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    used_bytes: usize,
    capacity_bytes: usize,
}

impl<K: Hash + Eq + Clone, V: Clone> LruCache<K, V> {
    /// Creates an empty cache that holds at most `capacity_bytes` of weight.
    pub fn new(capacity_bytes: usize) -> Self {
        LruCache {
            map: HashMap::new(),
            slab: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            used_bytes: 0,
            capacity_bytes,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Total weight currently stored.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Maximum total weight.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Whether `key` is cached (does not change recency).
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Returns the value for `key` and marks it most-recently used.
    pub fn get(&mut self, key: &K) -> Option<V> {
        let idx = *self.map.get(key)?;
        self.unlink(idx);
        self.push_front(idx);
        Some(self.node(idx).value.clone())
    }

    /// Inserts `key -> value` as most-recently used, evicting least-recently
    /// used entries until the budget is respected.
    ///
    /// Returns the evicted entries. An item heavier than the whole budget is
    /// not stored (it is returned as "evicted" immediately).
    pub fn insert(&mut self, key: K, value: V, weight: usize) -> Vec<(K, V)> {
        if let Some(&idx) = self.map.get(&key) {
            self.remove_slot(idx);
        }
        if weight > self.capacity_bytes {
            return vec![(key, value)];
        }
        let mut evicted = Vec::new();
        while self.used_bytes + weight > self.capacity_bytes {
            let tail = self.tail;
            let node = self.remove_slot(tail);
            evicted.push((node.key, node.value));
        }
        let node = Node {
            key: key.clone(),
            value,
            weight,
            prev: NIL,
            next: NIL,
        };
        let idx = match self.free.pop() {
            Some(i) => {
                self.slab[i] = Some(node);
                i
            }
            None => {
                self.slab.push(Some(node));
                self.slab.len() - 1
            }
        };
        self.push_front(idx);
        self.map.insert(key, idx);
        self.used_bytes += weight;
        evicted
    }

    /// Keys from most- to least-recently used (for tests and debugging).
    pub fn keys_by_recency(&self) -> Vec<K> {
        let mut keys = Vec::with_capacity(self.len());
        let mut cur = self.head;
        while cur != NIL {
            let node = self.node(cur);
            keys.push(node.key.clone());
            cur = node.next;
        }
        keys
    }

    fn node(&self, idx: usize) -> &Node<K, V> {
        self.slab[idx].as_ref().expect("live slot")
    }

    fn node_mut(&mut self, idx: usize) -> &mut Node<K, V> {
        self.slab[idx].as_mut().expect("live slot")
    }

    /// Detaches slot `idx` from the recency list (it stays allocated).
    fn unlink(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.node(idx);
            (n.prev, n.next)
        };
        match prev {
            NIL => self.head = next,
            p => self.node_mut(p).next = next,
        }
        match next {
            NIL => self.tail = prev,
            n => self.node_mut(n).prev = prev,
        }
    }

    /// Links slot `idx` at the head (most-recently used position).
    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        {
            let n = self.node_mut(idx);
            n.prev = NIL;
            n.next = old_head;
        }
        match old_head {
            NIL => self.tail = idx,
            h => self.node_mut(h).prev = idx,
        }
        self.head = idx;
    }

    /// Fully removes slot `idx` and returns its node.
    fn remove_slot(&mut self, idx: usize) -> Node<K, V> {
        self.unlink(idx);
        let node = self.slab[idx].take().expect("live slot");
        self.map.remove(&node.key);
        self.free.push(idx);
        self.used_bytes -= node.weight;
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_least_recently_used_first() {
        let mut lru = LruCache::new(3);
        lru.insert("a", 1, 1);
        lru.insert("b", 2, 1);
        lru.insert("c", 3, 1);
        assert_eq!(lru.get(&"a"), Some(1)); // a is now most recent
        let evicted = lru.insert("d", 4, 1);
        assert_eq!(evicted, vec![("b", 2)]);
        assert_eq!(lru.keys_by_recency(), vec!["d", "a", "c"]);
    }

    #[test]
    fn respects_byte_weights_and_rejects_oversized_items() {
        let mut lru = LruCache::new(10);
        lru.insert(1, 'x', 4);
        lru.insert(2, 'y', 4);
        let evicted = lru.insert(3, 'z', 6); // needs 14 > 10 -> evict key 1
        assert_eq!(evicted, vec![(1, 'x')]);
        assert_eq!(lru.used_bytes(), 10);
        assert_eq!(lru.insert(4, 'w', 11), vec![(4, 'w')]);
        assert!(!lru.contains(&4));
    }

    #[test]
    fn reinserting_a_key_replaces_it_and_slots_are_recycled() {
        let mut lru = LruCache::new(2);
        lru.insert(1, 10, 1);
        lru.insert(1, 11, 1);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.get(&1), Some(11));
        for i in 0..100 {
            lru.insert(i, i, 1);
        }
        assert_eq!(lru.len(), 2);
        assert!(
            lru.slab.len() <= 3,
            "slab should be recycled, got {}",
            lru.slab.len()
        );
    }

    #[test]
    fn zero_capacity_stores_nothing() {
        let mut lru = LruCache::new(0);
        lru.insert(1, 1, 1);
        assert!(lru.is_empty());
        assert_eq!(lru.get(&1), None);
    }
}

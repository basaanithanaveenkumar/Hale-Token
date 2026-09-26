//! Top-k selection with a bounded min-heap.
//!
//! The MoE router scores every expert (e.g. 128 of them) and keeps the best
//! `k` (e.g. 8). Sorting all scores costs `O(n log n)`; keeping a min-heap of
//! size `k` costs `O(n log k)` and never allocates more than `k` entries.
//! The heap's root is the *weakest* of the current winners, so each new
//! score only has to beat the root to get in.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// An index paired with its score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub index: usize,
    pub score: f32,
}

/// Heap entry ordered so the *worst* candidate sits at the top of Rust's
/// max-heap: lower score is "greater", and on ties the higher index is
/// "greater" (so lower indices win ties, which keeps results deterministic).
#[derive(PartialEq)]
struct Worst(Scored);

impl Eq for Worst {}

impl Ord for Worst {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.index.cmp(&other.0.index))
    }
}

impl PartialOrd for Worst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns the `k` highest scores, best first.
pub fn top_k(scores: &[f32], k: usize) -> Vec<Scored> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Worst> = BinaryHeap::with_capacity(k + 1);
    for (index, &score) in scores.iter().enumerate() {
        let candidate = Worst(Scored { index, score });
        if heap.len() < k {
            heap.push(candidate);
        } else if candidate < *heap.peek().expect("heap holds k items") {
            heap.pop();
            heap.push(candidate);
        }
    }
    // `into_sorted_vec` is ascending by `Ord`, i.e. best-first for `Worst`.
    heap.into_sorted_vec().into_iter().map(|w| w.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_full_sort() {
        let scores: Vec<f32> = (0..128)
            .map(|i| ((i * 73) % 128) as f32 * 0.37 - 20.0)
            .collect();
        let mut sorted: Vec<usize> = (0..scores.len()).collect();
        sorted.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let got: Vec<usize> = top_k(&scores, 8).iter().map(|s| s.index).collect();
        assert_eq!(got, sorted[..8]);
    }

    #[test]
    fn ties_prefer_lower_index_and_k_is_clamped() {
        let got: Vec<usize> = top_k(&[1.0, 5.0, 5.0, 5.0], 2)
            .iter()
            .map(|s| s.index)
            .collect();
        assert_eq!(got, vec![1, 2]);
        assert_eq!(top_k(&[1.0, 2.0], 10).len(), 2);
        assert!(top_k(&[1.0], 0).is_empty());
    }
}

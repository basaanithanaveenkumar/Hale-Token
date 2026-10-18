//! One token routing decision: chosen experts and weights.

use crate::expert::ExpertKey;

/// One routing decision: which expert, with what weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    /// The expert chosen.
    pub key: ExpertKey,
    /// Mixing weight of that expert's output.
    pub weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_compare_by_key_and_weight() {
        let a = Route {
            key: ExpertKey::new(0, 1),
            weight: 0.5,
        };
        assert_eq!(a, a);
        assert_ne!(a, Route { weight: 0.25, ..a });
    }
}

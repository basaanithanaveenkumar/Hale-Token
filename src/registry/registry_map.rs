//! [`Registry`]: name -> factory map with duplicate and typo detection.

use std::collections::BTreeMap;

use crate::error::{HaleTokenError, Result};

/// Builds a boxed `T` from borrowed arguments `A`.
type Factory<T, A> = Box<dyn Fn(&A) -> Box<T> + Send + Sync>;

/// A named collection of factories producing `Box<T>` from arguments `A`
/// (Registry + Factory patterns).
///
/// Keys are kept sorted, so listings and error messages are deterministic.
/// Lookups are `O(log n)`; `n` is tiny in practice.
///
/// # Examples
///
/// ```
/// use hale::registry::Registry;
///
/// trait Greeter { fn greet(&self) -> String; }
/// struct English(String);
/// impl Greeter for English {
///     fn greet(&self) -> String { format!("hello {}", self.0) }
/// }
///
/// let mut registry: Registry<dyn Greeter, String> = Registry::new("greeter");
/// registry
///     .register("english", |name| Box::new(English(name.clone())))
///     .unwrap();
/// let greeter = registry.build("english", &"world".to_string()).unwrap();
/// assert_eq!(greeter.greet(), "hello world");
/// assert!(registry.build("englsh", &String::new()).is_err()); // typo is reported
/// ```
pub struct Registry<T: ?Sized, A> {
    kind: &'static str,
    factories: BTreeMap<String, Factory<T, A>>,
}

impl<T: ?Sized, A> Registry<T, A> {
    /// Creates an empty registry; `kind` names it in error messages.
    pub fn new(kind: &'static str) -> Self {
        Registry {
            kind,
            factories: BTreeMap::new(),
        }
    }

    /// Registers `factory` under `key`.
    ///
    /// # Errors
    ///
    /// [`HaleTokenError::DuplicateRegistryKey`] if `key` is already taken; use
    /// [`Registry::register_force`] to replace it deliberately.
    pub fn register<F>(&mut self, key: &str, factory: F) -> Result<()>
    where
        F: Fn(&A) -> Box<T> + Send + Sync + 'static,
    {
        if self.factories.contains_key(key) {
            return Err(HaleTokenError::DuplicateRegistryKey {
                registry: self.kind,
                key: key.to_string(),
            });
        }
        self.register_force(key, factory);
        Ok(())
    }

    /// Registers `factory` under `key`, replacing any existing entry.
    pub fn register_force<F>(&mut self, key: &str, factory: F)
    where
        F: Fn(&A) -> Box<T> + Send + Sync + 'static,
    {
        self.factories.insert(key.to_string(), Box::new(factory));
    }

    /// Whether `key` is registered.
    pub fn contains(&self, key: &str) -> bool {
        self.factories.contains_key(key)
    }

    /// Registered keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }

    /// Builds the implementation registered under `key`.
    ///
    /// # Errors
    ///
    /// [`HaleTokenError::UnknownRegistryKey`], listing the closest registered
    /// names, when `key` is not registered.
    pub fn build(&self, key: &str, args: &A) -> Result<Box<T>> {
        match self.factories.get(key) {
            Some(factory) => Ok(factory(args)),
            None => Err(HaleTokenError::UnknownRegistryKey {
                registry: self.kind,
                key: key.to_string(),
                suggestions: self.suggestions(key),
            }),
        }
    }

    /// Up to three registered keys within edit distance 3 of `key`.
    fn suggestions(&self, key: &str) -> Vec<String> {
        let mut scored: Vec<(usize, &String)> = self
            .factories
            .keys()
            .map(|k| (edit_distance(key, k), k))
            .filter(|(d, _)| *d <= 3)
            .collect();
        scored.sort();
        scored.into_iter().take(3).map(|(_, k)| k.clone()).collect()
    }
}

/// Levenshtein distance in `O(|a| * |b|)` time and `O(|b|)` space.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (diagonal + usize::from(ca != *cb))
                .min(row[j] + 1)
                .min(above + 1);
            diagonal = above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Registry<u32, u32> {
        let mut r = Registry::new("number");
        r.register("double", |x| Box::new(x * 2)).unwrap();
        r.register("square", |x| Box::new(x * x)).unwrap();
        r
    }

    #[test]
    fn builds_by_name() {
        assert_eq!(*registry().build("square", &5).unwrap(), 25);
        assert_eq!(registry().keys().collect::<Vec<_>>(), ["double", "square"]);
    }

    #[test]
    fn duplicate_keys_are_rejected_unless_forced() {
        let mut r = registry();
        assert!(matches!(
            r.register("double", |x| Box::new(*x)),
            Err(HaleTokenError::DuplicateRegistryKey { .. })
        ));
        r.register_force("double", |x| Box::new(*x));
        assert_eq!(*r.build("double", &7).unwrap(), 7);
    }

    #[test]
    fn unknown_keys_list_close_matches() {
        let err = registry().build("dubble", &1).unwrap_err();
        match err {
            HaleTokenError::UnknownRegistryKey { suggestions, .. } => {
                assert_eq!(suggestions, ["double"])
            }
            other => panic!("unexpected error {other}"),
        }
    }

    #[test]
    fn edit_distance_basics() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("same", "same"), 0);
    }
}

//! The `hale` command-line tool. All real work lives in the `hale` library;
//! this binary only parses arguments, configures logging and prints results.
//!
//! Library diagnostics go through [`tracing`]; set `HALE_LOG` (same syntax as
//! `RUST_LOG`, e.g. `HALE_LOG=hale=debug`) to see them. The default shows warnings only, so
//! the command output itself is unchanged.

mod cli;

use tracing_subscriber::EnvFilter;

/// Environment variable holding the log filter directive.
const LOG_ENV: &str = "HALE_LOG";

fn init_logging() {
    let filter = EnvFilter::try_from_env(LOG_ENV).unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .compact()
        .init();
}

fn main() {
    init_logging();
    if let Err(err) = cli::run() {
        // Deliberate CLI output: the one-line error the user acts on.
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

//! The `hale` command-line tool. All real work lives in the `hale` library;
//! this binary only parses arguments and prints results.

mod cli;

fn main() {
    if let Err(err) = cli::run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

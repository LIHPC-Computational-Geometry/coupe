use std::env;
use std::process::Command;

fn main() {
    let git_commit = {
        let hash = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout;
        let mut hash = String::from_utf8(hash).unwrap();
        hash.truncate(7);
        hash
    };

    let git_clean_tree = Command::new("git")
        .args(["status", "-s"])
        .output()
        .unwrap()
        .stdout
        .is_empty();

    let crate_version = env::var("CARGO_PKG_VERSION").unwrap();

    let coupe_version = if git_clean_tree {
        format!("{crate_version}-{git_commit}")
    } else {
        format!("{crate_version}-{git_commit}+dirty")
    };

    println!("cargo:rustc-env=COUPE_VERSION={coupe_version}");
}

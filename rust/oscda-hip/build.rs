use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=OSCDA_LIB_DIR");
    let directory = PathBuf::from(
        env::var_os("OSCDA_LIB_DIR")
            .expect("set OSCDA_LIB_DIR to the directory containing libosc.so"),
    );
    let directory = directory
        .canonicalize()
        .expect("OSCDA_LIB_DIR must be an existing directory");
    assert!(
        directory.join("libosc.so").is_file(),
        "OSCDA_LIB_DIR must contain libosc.so"
    );
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=dylib=osc");
}

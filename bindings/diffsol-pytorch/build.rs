use std::env;
use std::path::Path;
use std::process::Command;

fn has_feature(name: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{name}"))
        .map(|_| true)
        .unwrap_or(false)
}

fn ensure_command(name: &str, args: &[&str]) {
    if env::var("DIFFSOL_SKIP_TOOLCHAIN_CHECK").is_ok() {
        return;
    }
    match Command::new(name).args(args).output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            panic!(
                "Command `{name}` `{args:?}` failed with status {status}. \
                 Install the tool or set DIFFSOL_SKIP_TOOLCHAIN_CHECK=1 to bypass.",
                status = output.status
            );
        }
        Err(err) => panic!(
            "Required build dependency `{name}` not found in PATH ({err}). \
             Install it or export DIFFSOL_SKIP_TOOLCHAIN_CHECK=1 to bypass."
        ),
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=LLVM_SYS_181_PREFIX");
    println!("cargo:rerun-if-env-changed=DIFFSOL_SKIP_TOOLCHAIN_CHECK");

    if !has_feature("LLVM") {
        return;
    }

    ensure_command("cmake", &["--version"]);
    if cfg!(target_os = "windows") {
        ensure_command("ninja", &["--version"]);
    }

    let prefix =
        env::var("LLVM_SYS_181_PREFIX").expect("LLVM_SYS_181_PREFIX must be set for LLVM builds");
    let prefix_path = Path::new(&prefix);
    if !prefix_path.is_dir() {
        panic!("LLVM_SYS_181_PREFIX path {prefix:?} does not exist");
    }
    let llvm_bin = prefix_path.join("bin");
    let llvm_config = llvm_bin.join(if cfg!(target_os = "windows") {
        "llvm-config.exe"
    } else {
        "llvm-config"
    });
    if !llvm_config.exists() {
        panic!(
            "llvm-config not found under {}. Did you point LLVM_SYS_181_PREFIX at a full LLVM install?",
            llvm_bin.display()
        );
    }
    let output = Command::new(&llvm_config)
        .arg("--version")
        .output()
        .expect("failed to invoke llvm-config --version");
    if !output.status.success() {
        panic!("llvm-config --version exited with {}", output.status);
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let major = version
        .split('.')
        .next()
        .unwrap_or_default()
        .parse::<u32>()
        .expect("malformed LLVM version");
    const SUPPORTED: &[u32] = &[18];
    if !SUPPORTED.contains(&major) && env::var("DIFFSOL_ALLOW_UNSUPPORTED_LLVM").is_err() {
        panic!(
            "diffsol-pytorch is only validated with LLVM {SUPPORTED:?}. \
             Detected {version}. Set DIFFSOL_ALLOW_UNSUPPORTED_LLVM=1 to bypass."
        );
    }
}

//! Development tasks for the iletki workspace: `cargo xtask <task>`.

// A command-line tool reports to the terminal.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::{Command, ExitCode};

type Result<T = ()> = std::result::Result<T, String>;

/// The crates published to crates.io; they must also build for the web.
const LIBRARIES: [&str; 7] = [
    "iletki",
    "iletki-core",
    "iletki-wgpu",
    "iletki-2d",
    "iletki-3d",
    "iletki-egui",
    "iletki-slint",
];

const USAGE: &str = "\
usage: cargo xtask <task>

tasks:
  ci       fmt, clippy, test, doc, deny and wasm, in that order
  doc      build the documentation, warnings as errors
  deny     check dependency licenses, advisories, bans and sources
  wasm     build the libraries for wasm32-unknown-unknown";

fn main() -> ExitCode {
    let result = match std::env::args().nth(1).as_deref() {
        Some("ci") => ci(),
        Some("doc") => doc(),
        Some("deny") => deny(),
        Some("wasm") => wasm(),
        Some("-h" | "--help" | "help") | None => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(format!("unknown task `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn ci() -> Result {
    step("fmt", || cargo(&["fmt", "--all", "--check"], &[]))?;
    step("clippy", || {
        cargo(
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
            &[],
        )
    })?;
    step("test", || {
        cargo(&["test", "--workspace", "--all-features", "--locked"], &[])
    })?;
    step("doc", doc)?;
    step("deny", deny)?;
    step("wasm", wasm)?;
    println!("\nci: all checks passed");
    Ok(())
}

fn doc() -> Result {
    let mut args = vec!["doc", "--no-deps", "--all-features", "--locked"];
    for name in LIBRARIES {
        args.extend(["-p", name]);
    }
    cargo(&args, &[("RUSTDOCFLAGS", "-D warnings")])
}

fn deny() -> Result {
    cargo(&["deny", "--locked", "check"], &[]).map_err(|e| {
        format!("{e}\n(install the checker once with `cargo install --locked cargo-deny`)")
    })
}

fn wasm() -> Result {
    let target = "wasm32-unknown-unknown";
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .any(|l| l.trim() == target)
        })
        .unwrap_or(false);
    if !installed {
        return Err(format!(
            "the {target} target is missing; run `rustup target add {target}`"
        ));
    }
    let mut args = vec!["check", "--locked", "--target", target];
    for name in LIBRARIES {
        args.extend(["-p", name]);
    }
    cargo(&args, &[])
}

fn step(name: &str, f: impl FnOnce() -> Result) -> Result {
    println!("\n== {name}");
    f().map_err(|e| format!("{name} failed: {e}"))
}

fn cargo(args: &[&str], env: &[(&str, &str)]) -> Result {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(&cargo)
        .args(args)
        .envs(env.iter().copied())
        .status()
        .map_err(|e| format!("could not run `{cargo}`: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`cargo {}` exited with {status}", args.join(" ")))
    }
}

# iletki — instructions for agents and contributors

iletki is a fast, UI-framework-agnostic `wgpu` viewport library in pure Rust for 2D/3D CAD,
game engine editors, GIS and visualization.

## Workspace

See [docs/architecture.md](docs/architecture.md) for the layers. Today:

- `crates/iletki-core` — pure-Rust math (`f64`), 2D/3D cameras (split `hi/lo` and camera-relative
  Reversed-Z), frustums, clip planes, camera rigs, input/command contracts, draw phases, pick IDs
  and target block sizing. Zero external runtime dependencies.
- `crates/iletki` — top-level facade crate.
- `xtask` — development tasks (`cargo xtask ci`).

## Non-negotiables

- **Framework-agnostic core.** `iletki-core`, `iletki-wgpu`, `iletki-2d` and `iletki-3d` never
  depend on `egui`, `eframe`, `winit`, `iced`, `slint` or any windowing/UI framework. Adapters
  live in thin, separate crates.
- **Never crash.** No `unwrap`/`expect`/`panic!`/`todo!`/`unimplemented!`/`unreachable!`, no
  slice indexing and no `unsafe` outside tests. All inputs (dimensions, coordinates, factors,
  deltas, counts) are treated as hostile: guard against NaN/Inf, zero dimensions and overflows.
  Every crash fix lands with a regression test.
- **Precision first.** World coordinates and camera states are `f64`. 2D uses split `hi/lo` `f32`
  relative subtraction; 3D uses camera-relative transforms and Reversed-Z (`Depth32Float`).
- **Portable.** Core libraries do not touch the file system or spawn threads unless behind a cargo
  feature. Every library crate builds for `wasm32-unknown-unknown`.
- **Clean-room and licenses.** iletki is MIT OR Apache-2.0. Never copy GPL/LGPL/AGPL code.
  Dependencies must pass `cargo deny check`. Ask before adding a new dependency.

## Quality gate

Run `cargo xtask ci` before every commit. CI runs it on Linux, Windows and macOS.

## Conventions

- Rust 2024 edition, `rustfmt` defaults, lints from the workspace `Cargo.toml`.
- Public items are documented (`missing_docs` is on).
- Code, comments, commit messages and documentation are in English.
- Branches and commit messages follow [CONTRIBUTING.md](CONTRIBUTING.md).
- A local `plan/` folder, if present, is private planning and reference material: it is gitignored
  and its contents never go into tracked files, commit messages or public discussions.

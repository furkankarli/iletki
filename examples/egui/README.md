# Minimal egui viewport

This standalone example uses the registry versions of `iletki` and `iletki-egui`. It attaches a
viewport to an `egui_wgpu` paint callback, renders one offscreen frame, and verifies its center
pixel and picking result. It needs a working `wgpu` adapter, and runs without a window.

After the matching crate versions are available on crates.io, run from the repository root:

```sh
cargo run --manifest-path examples/egui/Cargo.toml
```

To run against the repository's packaged archives, use `python3 .github/scripts/package-smoke.py`.

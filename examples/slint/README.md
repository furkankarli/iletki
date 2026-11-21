# iletki in Slint

Run from the repository root on a desktop with a `wgpu` adapter.

```sh
cargo run --locked --manifest-path examples/slint/Cargo.toml
```

Drag with the middle button to pan, scroll to zoom, and click a rectangle to display its 64-bit
picking ID. Resizing the window updates the offscreen viewport in physical pixels.

Slint 1.19 uses its experimental `unstable-wgpu-30` API to provide the device and queue to
`ViewportHost`. The demo imports iletki's resolved GPU texture as a Slint `Image` and clips it to
the active extent. Picking reads back only the selected pixel and completes asynchronously.

## Check the integration

The smoke test reads pixels from the Slint window, sends pointer events through Slint, and
verifies picking, resize, pan, and zoom. Linux CI uses a virtual X11 display at 2× scale to
isolate the test from desktop input. On Linux with Xvfb and a software Vulkan adapter, run:

```sh
SLINT_BACKEND=winit SLINT_WGPU_CPU=1 SLINT_SCALE_FACTOR=2 WGPU_BACKEND=vulkan \
  xvfb-run -a cargo run --locked --manifest-path examples/slint/Cargo.toml -- --smoke
```

```sh
cargo deny --manifest-path examples/slint/Cargo.toml --config examples/slint/deny.toml --locked check
```

## Licensing

This standalone example uses the published iletki crate. The example source is MIT OR Apache-2.0.
Slint is used under its Royalty-free Desktop, Mobile, and Web Applications License, which has
attribution and distribution conditions. The example's dependency policy accepts that license
for the pinned Slint crates and records maintenance-only advisories for their current
transitive dependencies. See the [Slint license terms](https://slint.dev/terms-and-conditions).

[![Made with Slint](https://raw.githubusercontent.com/slint-ui/slint/v1.19.0/logo/MadeWithSlint-logo-whitebg.png)](https://slint.dev/)

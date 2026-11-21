# iletki-slint

Embed an iletki viewport in a Slint application using the shared `wgpu 30` device. The adapter
handles texture import, Slint image cropping, DPI and pixel mapping for GPU picking.

Add the viewport and adapter to a Slint application:

```sh
cargo add iletki iletki-slint
```

Use Slint 1.19 with the `unstable-wgpu-30` feature and a compatible renderer, such as
`renderer-femtovg-wgpu`. Select the backend with `require_wgpu_30`. In the Slint window's
`RenderingSetup` notifier, construct a `SlintViewport` from its device and queue.

```rust,ignore
let mut viewport = iletki_slint::SlintViewport::new(device, queue, config)?;
```

In `BeforeRendering`, update the size from the image element in physical pixels and call the
adapter with Slint's importer.

```rust,ignore
viewport.resize([width_px, height_px], window.scale_factor());
let frame = viewport.render(&world, &mut layers, slint::Image::try_from)?;
if let Some(image) = frame.new_image {
    ui.set_frame(image);
}
let [width, height] = frame.source_size;
ui.set_frame_width(width);
ui.set_frame_height(height);
```

Bind those sizes to `Image.source-clip-width` and `Image.source-clip-height`. Send widget-local
logical points to `begin_pick` and poll the resulting `PendingPick`. Use `host_mut()` to route
camera inputs or edit the camera. Recreate the adapter when the Slint device is recreated.

Slint is provided by the application, which supplies its own image importer and follows the
applicable Slint license. The adapter and its packaged dependencies use permissive licenses.

[Runnable Slint example](https://github.com/furkankarli/iletki/tree/main/examples/slint) · [Slint wgpu API](https://docs.slint.dev/latest/docs/rust/slint/wgpu_30/) · [Support iletki](https://github.com/sponsors/furkankarli)

Licensed under [MIT](https://github.com/furkankarli/iletki/blob/main/LICENSE-MIT) or
[Apache 2.0](https://github.com/furkankarli/iletki/blob/main/LICENSE-APACHE) at your option.

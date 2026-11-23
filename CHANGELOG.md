# Changelog

## 0.1.1

- Apply 4× MSAA to live 3D section boundaries on supported adapters. Use
  `ViewportHost::new_with_adapter` to select the available section path.
- Antialias selection outlines and limit outline work to the affected screen region.
- Batch compatible 3D faces and edges, reduce GPU transform uploads, and group visibility
  tests for large scenes. Merge short gaps in heavily fragmented raster section draws.
- Reduce marquee picking work by restricting geometry to the pick region and reusing GPU
  readback buffers. Preserve ordered 64-bit hit IDs and reconstructed positions.
- Pace interactive redraws and marquee previews in the desktop demo. Report CPU encoding
  and total frame time separately, with the startup frame labeled on its own.
- Show selection, shared camera views, and live sections with real STEP and IFC models
  in the project and crate READMEs.

On Mesa 25.2 lavapipe, the sampled section path can quantize a CAD line's cut endpoint
to a whole pixel. Section clipping still removes the line beyond the cut.

## 0.1.0

Initial crates.io release of the viewport libraries and the egui and Slint adapters.

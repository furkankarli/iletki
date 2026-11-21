//! Embed an iletki viewport in a native Slint window using a shared GPU device.

mod smoke;

use std::cell::RefCell;
use std::error::Error;
use std::rc::Rc;
use std::task::Poll;

use iletki::gpu::wgpu;
use iletki::{
    Cad2DRig, DrawStyle2D, Mesh2D, Modifiers, PendingPick, PickId, PointerButton, Scene2DLayer,
    ScrollUnit, Vec2, ViewportConfig, ViewportHost, ViewportInput, ViewportLayer,
};
use slint::{ComponentHandle, GraphicsAPI, RenderingState};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CENTER_ID: PickId = PickId::new(1, 0x1234_5678, 0x9abc);
const CENTER_COLOR: [u8; 4] = [53, 208, 150, 255];

slint::slint! {
    #[edition = "1.19"]
    export component Demo inherits Window {
        title: "iletki · Slint";
        preferred-width: 640px;
        preferred-height: 420px;
        min-width: 240px;
        min-height: 180px;
        background: #111927;

        in property <image> frame;
        in property <int> frame-width;
        in property <int> frame-height;
        in property <string> status: "Click a shape to pick it";
        callback move-pointer(float, float);
        callback pan-button(float, float, bool);
        callback cancel-pan();
        callback wheel(float, float, float, float);
        callback pick(float, float);

        Image {
            width: parent.width;
            height: parent.height;
            source: root.frame;
            image-fit: fill;
            source-clip-width: root.frame-width;
            source-clip-height: root.frame-height;
            accessible-label: "Interactive 2D viewport";
        }

        TouchArea {
            mouse-cursor: crosshair;
            moved => { root.move-pointer(self.mouse-x / 1px, self.mouse-y / 1px); }
            pointer-event(event) => {
                if event.kind == PointerEventKind.cancel {
                    root.cancel-pan();
                } else if event.button == PointerEventButton.middle {
                    if event.kind == PointerEventKind.down || event.kind == PointerEventKind.up {
                        root.pan-button(self.mouse-x / 1px, self.mouse-y / 1px, event.kind == PointerEventKind.down);
                    }
                }
            }
            scroll-event(event) => {
                root.wheel(self.mouse-x / 1px, self.mouse-y / 1px, event.delta-x / 1px, event.delta-y / 1px);
                return accept;
            }
            clicked => { root.pick(self.mouse-x / 1px, self.mouse-y / 1px); }
        }

        Rectangle {
            y: 0;
            width: parent.width;
            height: 34px;
            background: #162135;
            Text {
                x: 12px;
                width: parent.width - 24px;
                color: #e1e9f6;
                overflow: elide;
                text: "Middle drag to pan · Scroll to zoom";
            }
        }

        Rectangle {
            y: parent.height - self.height;
            width: parent.width;
            height: 34px;
            background: #162135;
            Text {
                x: 12px;
                width: parent.width - 24px;
                color: #e1e9f6;
                overflow: elide;
                text: root.status;
            }
        }
    }
}

struct Viewport {
    host: ViewportHost,
    scene: Scene2DLayer,
    rig: Cad2DRig,
    cursor: Option<Vec2>,
    imported_texture: Option<wgpu::Texture>,
    pick_pos: Option<Vec2>,
    pending_pick: Option<PendingPick>,
}

impl Viewport {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let mut mesh = Mesh2D::default();
        let grid_style = DrawStyle2D::new([41, 57, 79, 255], false, 0.15, PickId::NONE);
        for index in -28..=28 {
            let line = f64::from(index) * 24.0;
            mesh.push_line(Vec2::new(line, -672.0), Vec2::new(line, 672.0), grid_style);
            mesh.push_line(Vec2::new(-672.0, line), Vec2::new(672.0, line), grid_style);
        }
        for (center, color, id) in [
            (
                Vec2::new(-150.0, 42.0),
                [95, 157, 255, 255],
                PickId::new(1, 1, 0),
            ),
            (Vec2::ZERO, CENTER_COLOR, CENTER_ID),
            (
                Vec2::new(150.0, -42.0),
                [247, 162, 81, 255],
                PickId::new(1, 2, 0),
            ),
        ] {
            let half_extent = Vec2::new(44.0, 28.0);
            mesh.push_rect_fill(
                center - half_extent,
                center + half_extent,
                DrawStyle2D::new(color, false, 0.5, id),
            );
        }
        let mut scene = Scene2DLayer::new(1);
        scene.set_static_mesh(&mesh);
        let host = ViewportHost::new(
            device,
            ViewportConfig {
                color_format: wgpu::TextureFormat::Rgba8Unorm,
                clear_color: [0.067, 0.098, 0.153, 1.0],
                ..ViewportConfig::default()
            },
        )?;
        Ok(Self {
            host,
            scene,
            rig: Cad2DRig::default(),
            cursor: None,
            imported_texture: None,
            pick_pos: None,
            pending_pick: None,
        })
    }

    fn render(&mut self, app: &Demo, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        let window = app.window();
        let size = window.size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let scale = window.scale_factor();
        self.host.resize(device, [size.width, size.height], scale);
        let mut encoder = device.create_command_encoder(&Default::default());
        let layers: &mut [&mut dyn ViewportLayer<()>] = &mut [&mut self.scene];
        let frame = self
            .host
            .render_to_texture(device, queue, &mut encoder, &(), layers);
        let [width, height] = frame.active_px;
        let streaming_pending = frame.stats.streaming_pending;
        let texture = frame.texture_view.texture();
        let new_image = if self.imported_texture.as_ref() == Some(texture) {
            None
        } else {
            let image = slint::Image::try_from(texture.clone())?;
            self.imported_texture = Some(texture.clone());
            Some(image)
        };
        queue.submit([encoder.finish()]);

        // Slint samples the active area of iletki's padded allocation.
        let [width_i32, height_i32] = [i32::try_from(width)?, i32::try_from(height)?];
        if app.get_frame_width() != width_i32 {
            app.set_frame_width(width_i32);
        }
        if app.get_frame_height() != height_i32 {
            app.set_frame_height(height_i32);
        }
        if let Some(image) = new_image {
            app.set_frame(image);
        }

        if let Some(pos) = self.pick_pos.take() {
            let x = pos.x * f64::from(scale);
            let y = pos.y * f64::from(scale);
            self.pending_pick =
                if x >= 0.0 && x < f64::from(width) && y >= 0.0 && y < f64::from(height) {
                    Some(self.host.begin_pick_rect(
                        device,
                        queue,
                        &(),
                        layers,
                        [x as u32, y as u32, 1, 1],
                    )?)
                } else {
                    app.set_status("No object selected".into());
                    None
                };
        }
        if let Some(pick) = self.pending_pick.as_mut() {
            match pick.try_read() {
                Poll::Pending => {
                    let weak = app.as_weak();
                    slint::Timer::single_shot(std::time::Duration::from_millis(8), move || {
                        if let Some(app) = weak.upgrade() {
                            app.window().request_redraw();
                        }
                    });
                }
                Poll::Ready(result) => {
                    self.pending_pick = None;
                    let hits = result?;
                    let status = hits.first().map_or_else(
                        || "No object selected".into(),
                        |hit| format!("Picked {}:{}:{}", hit.id.layer, hit.id.object, hit.id.sub),
                    );
                    app.set_status(status.into());
                }
            }
        }
        if streaming_pending {
            window.request_redraw();
        }
        Ok(())
    }

    fn input(&mut self, app: &Demo, input: ViewportInput) {
        if self.host.handle_input(&input, &mut self.rig).camera_changed {
            app.window().request_redraw();
        }
    }
}

type State = Rc<RefCell<Option<Viewport>>>;

fn with_viewport(
    state: &State,
    weak: &slint::Weak<Demo>,
    x: f32,
    y: f32,
    handle: impl FnOnce(&mut Viewport, &Demo, Vec2),
) {
    let pos = Vec2::new(f64::from(x), f64::from(y));
    if pos.is_finite()
        && let (Some(app), Ok(mut state)) = (weak.upgrade(), state.try_borrow_mut())
        && let Some(viewport) = state.as_mut()
    {
        handle(viewport, &app, pos);
    }
}

fn bind_input(app: &Demo, state: &State) {
    let (weak, viewport) = (app.as_weak(), state.clone());
    app.on_move_pointer(move |x, y| {
        with_viewport(&viewport, &weak, x, y, |state, app, pos| {
            let delta = state
                .cursor
                .replace(pos)
                .map_or(Vec2::ZERO, |previous| pos - previous);
            state.input(
                app,
                ViewportInput::PointerMove {
                    pos,
                    delta,
                    modifiers: Modifiers::default(),
                },
            );
        });
    });

    let (weak, viewport) = (app.as_weak(), state.clone());
    app.on_pan_button(move |x, y, pressed| {
        with_viewport(&viewport, &weak, x, y, |state, app, pos| {
            state.cursor = Some(pos);
            let modifiers = Modifiers::default();
            let button = PointerButton::Middle;
            let input = if pressed {
                ViewportInput::PointerDown {
                    pos,
                    button,
                    modifiers,
                }
            } else {
                ViewportInput::PointerUp {
                    pos,
                    button,
                    modifiers,
                }
            };
            state.input(app, input);
        });
    });

    let (weak, viewport) = (app.as_weak(), state.clone());
    app.on_cancel_pan(move || {
        with_viewport(&viewport, &weak, 0.0, 0.0, |state, app, _| {
            state.cursor = None;
            state.input(app, ViewportInput::FocusLost);
        });
    });

    let (weak, viewport) = (app.as_weak(), state.clone());
    app.on_wheel(move |x, y, dx, dy| {
        with_viewport(&viewport, &weak, x, y, |state, app, pos| {
            state.input(
                app,
                ViewportInput::Scroll {
                    pos,
                    delta: Vec2::new(f64::from(dx), f64::from(dy)),
                    unit: ScrollUnit::Pixel,
                    modifiers: Modifiers::default(),
                },
            );
        });
    });

    let (weak, viewport) = (app.as_weak(), state.clone());
    app.on_pick(move |x, y| {
        with_viewport(&viewport, &weak, x, y, |state, app, pos| {
            state.pick_pos = Some(pos);
            app.window().request_redraw();
        });
    });
}

fn main() -> Result<()> {
    slint::BackendSelector::new()
        .require_wgpu_30(slint::wgpu_30::WGPUConfiguration::default())
        .select()?;
    let app = Demo::new()?;
    let state = Rc::new(RefCell::new(None));
    bind_input(&app, &state);

    let weak = app.as_weak();
    app.window().set_rendering_notifier(move |phase, api| {
        let (Some(app), Ok(mut state)) = (weak.upgrade(), state.try_borrow_mut()) else {
            return;
        };
        let GraphicsAPI::WGPU30 { device, queue, .. } = api else {
            app.set_status("Slint did not provide a wgpu 30 device".into());
            return;
        };
        let result = match phase {
            RenderingState::RenderingSetup => {
                Viewport::new(device).map(|viewport| *state = Some(viewport))
            }
            RenderingState::BeforeRendering => state
                .as_mut()
                .map_or(Ok(()), |v| v.render(&app, device, queue)),
            RenderingState::RenderingTeardown => {
                app.set_frame(slint::Image::default());
                *state = None;
                Ok(())
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            app.set_status(format!("Viewport error: {error}").into());
        }
    })?;

    if std::env::args_os()
        .nth(1)
        .is_some_and(|arg| arg == "--smoke")
    {
        smoke::run(&app)
    } else {
        Ok(app.run()?)
    }
}

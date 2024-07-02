use miniquad::EventHandler;
use quad_gl::{
    camera::{Camera, Environment, Projection},
    color::*,
    math::*,
    scene::Scene,
    sprite_batcher::SpriteBatcher,
    QuadGl,
};
use std::sync::{Arc, Mutex};

struct Stage {
    ctx: Arc<Mutex<Box<miniquad::Context>>>,
    camera: Camera,
    scene: Scene,
    canvas: SpriteBatcher,
}

impl Stage {
    pub fn new() -> Stage {
        unsafe { miniquad::gl::glEnable(miniquad::gl::GL_TEXTURE_CUBE_MAP_SEAMLESS) };

        let ctx = miniquad::window::new_rendering_backend();
        let ctx = Arc::new(Mutex::new(ctx));

        let graphics = QuadGl::new(ctx.clone());
        let mut scene = graphics.new_scene();

        let helmet = graphics
            .load_gltf(&include_str!("DamagedHelmet.gltf"))
            .unwrap();
        let _helmet = scene.add_model(&helmet);

        let skybox = quad_gl::cubemap::Cubemap::new(
            ctx.lock().unwrap().as_mut(),
            &[
                include_bytes!("skybox/skybox_px.png"),
                include_bytes!("skybox/skybox_nx.png"),
                include_bytes!("skybox/skybox_py.png"),
                include_bytes!("skybox/skybox_ny.png"),
                include_bytes!("skybox/skybox_pz.png"),
                include_bytes!("skybox/skybox_nz.png"),
            ],
        );

        let camera = Camera {
            environment: Environment::Skybox(skybox),
            depth_enabled: true,
            projection: Projection::Perspective,
            position: vec3(0., 1.5, 4.),
            up: vec3(0., 1., 0.),
            target: vec3(0., 0., 0.),
            z_near: 0.1,
            z_far: 15.0,
            ..Default::default()
        };

        let mut canvas = graphics.new_canvas();
        canvas.draw_text("HELLO 3D WORLD", 10.0, 20.0, 30.0, BLACK);

        Stage {
            ctx,
            camera,
            canvas,
            scene,
        }
    }
}

impl EventHandler for Stage {
    fn update(&mut self) {}

    fn draw(&mut self) {
        self.ctx
            .lock()
            .unwrap()
            .clear(Some((1., 1., 1., 1.)), Some(1.), None);

        self.scene.draw(&mut self.camera);
        self.canvas.draw();
    }
}

fn main() {
    miniquad::start(Default::default(), || Box::new(Stage::new()));
}

use miniquad::{EventHandler, KeyCode, KeyMods};
use quad_gl::{
    camera::{Camera, Environment, Projection},
    color::*,
    math::*,
    math::{vec3, Vec3},
    models,
    scene::{ModelHandle, Scene},
    sprite_batcher::SpriteBatcher,
    QuadGl,
};
use std::sync::{Arc, Mutex};

struct Stage {
    ctx: Arc<Mutex<Box<miniquad::Context>>>,
    camera: Camera,
    scene: Scene,
    bunny: ModelHandle,
    canvas: SpriteBatcher,
    time: f64,
    bunnies: Vec<Vec3>,
    bunnies_dir: Vec<Vec3>,
}

impl Stage {
    pub fn new() -> Stage {
        unsafe { miniquad::gl::glEnable(miniquad::gl::GL_TEXTURE_CUBE_MAP_SEAMLESS) };

        let ctx = miniquad::window::new_rendering_backend();
        let ctx = Arc::new(Mutex::new(ctx));

        let graphics = QuadGl::new(ctx.clone());
        let mut scene = graphics.new_scene();

        let texture = graphics.load_texture(include_bytes!("ferris2.png"));
        // TODO: not quite draw_rectangle
        let bunny = graphics.mesh(models::square(), texture);
        let bunny = scene.add_model(&bunny);

        let camera = Camera {
            environment: Environment::SolidColor(WHITE),
            depth_enabled: false,
            projection: Projection::Orthographic,
            position: vec3(0., 10.0, 0.),
            up: vec3(0., 1., 0.),
            target: vec3(0.0, 0., 1.),
            z_near: 0.1,
            z_far: 15.0,
            ..Default::default()
        };

        let canvas = graphics.new_canvas();

        Stage {
            ctx,
            camera,
            canvas,
            scene,
            bunny,
            time: miniquad::date::now(),
            bunnies: vec![],
            bunnies_dir: vec![],
        }
    }
}

impl EventHandler for Stage {
    fn key_down_event(&mut self, _keycode: KeyCode, _keymods: KeyMods, _repeat: bool) {
        for _ in 0..1000 {
            self.bunnies.push(vec3(
                quad_rand::gen_range(-20.0, 20.0),
                0.0,
                quad_rand::gen_range(-20.0, 20.0),
            ));
            self.bunnies_dir.push(vec3(
                quad_rand::gen_range(-1.0, 1.0),
                0.,
                quad_rand::gen_range(-1.0, 1.0),
            ));
        }
        self.scene
            .update_multi_positions(&self.bunny, &self.bunnies);
    }

    fn update(&mut self) {}
    fn draw(&mut self) {
        let frame_time = miniquad::date::now() - self.time;
        self.time = miniquad::date::now();

        self.ctx
            .lock()
            .unwrap()
            .clear(Some((1., 1., 1., 1.)), Some(1.), None);

        for (bunny, dir) in self.bunnies.iter_mut().zip(self.bunnies_dir.iter_mut()) {
            *bunny += *dir;
            if bunny.x >= 20.0 || bunny.x <= -20.0 {
                dir.x *= -1.0;
            }
            if bunny.z >= 20.0 || bunny.z <= -20.0 {
                dir.z *= -1.0;
            }
        }

        self.scene
            .update_multi_positions(&self.bunny, &self.bunnies);

        self.scene.draw(&mut self.camera);

        self.canvas.clear();
        self.canvas.draw_text(
            &format!("fps: {:0.1}", 1.0 / frame_time),
            0.0,
            16.0,
            16.0,
            BLACK,
        );
        self.canvas.draw_text(
            &format!("bunnies: {}", self.bunnies.len()),
            0.0,
            32.0,
            16.0,
            BLACK,
        );
        self.canvas
            .draw_text(&format!("Press any key"), 0.0, 48.0, 16.0, BLACK);
        self.canvas.draw();
    }
}

fn main() {
    miniquad::start(Default::default(), || Box::new(Stage::new()));
}

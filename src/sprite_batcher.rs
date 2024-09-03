use crate::{
    draw_calls_batcher::{DrawCallsBatcher, Vertex},
    math::{vec2, Mat4, Vec2},
    shapes::{DrawMode, DrawParams},
    text,
};

use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

pub enum ViewportBound {
    Vertical(f32),
    Horizontal(f32),
}

pub struct SpriteBatcher {
    pub(crate) quad_ctx: Arc<Mutex<Box<miniquad::Context>>>,
    pub(crate) fonts_storage: Arc<Mutex<text::FontsStorage>>,
    pub(crate) batcher: DrawCallsBatcher,
    pub(crate) axis: Axis,
    pub viewport_center: Option<Vec2>,
    pub viewport_bound: Option<ViewportBound>,
    pub matrix_override: Option<Mat4>,
}

impl crate::shapes::Mesher for SpriteBatcher {
    fn quad_ctx(&self) -> &Arc<Mutex<Box<miniquad::Context>>> {
        &self.quad_ctx
    }
    fn fonts_storage(&self) -> &Arc<Mutex<crate::text::FontsStorage>> {
        &self.fonts_storage
    }

    fn texture(&mut self, texture: Option<miniquad::TextureId>) {
        self.batcher.texture(texture);
    }
    fn draw_mode(&mut self, mode: DrawMode) {
        self.batcher.draw_mode(mode);
    }
    fn geometry(&mut self, vertices: &[Vertex], indices: &[u16]) {
        self.batcher.geometry(vertices, indices)
    }
}

impl SpriteBatcher {
    pub fn new(
        quad_ctx: Arc<Mutex<Box<miniquad::Context>>>,
        fonts_storage: Arc<Mutex<text::FontsStorage>>,
    ) -> SpriteBatcher {
        let mut ctx = quad_ctx.lock().unwrap();

        let batcher = DrawCallsBatcher::new(&mut **ctx);
        SpriteBatcher {
            quad_ctx: quad_ctx.clone(),
            fonts_storage: fonts_storage.clone(),
            batcher,
            axis: Axis::Z,
            viewport_center: None,
            viewport_bound: None,
            matrix_override: None,
        }
    }

    pub fn clear(&mut self) {
        self.batcher.clear(self.quad_ctx.lock().unwrap().as_mut())
    }

    pub fn set_axis(&mut self, axis: Axis) {
        self.axis = axis;
    }

    pub fn gl(&mut self) -> &mut DrawCallsBatcher {
        &mut self.batcher
    }

    pub fn reset(&mut self) {
        self.batcher.reset()
    }

    pub fn draw(&mut self, shape: impl crate::shapes::Draw, pos: Vec2, p: impl Into<DrawParams>) {
        shape.draw(self, pos, p);
    }

    pub fn blit(&mut self) {
        let mut ctx = self.quad_ctx.lock().unwrap();

        let screen_mat = self.matrix_override.unwrap_or_else(|| {
            let screen_size = miniquad::window::screen_size();
            let aspect = screen_size.0 / screen_size.1;
            let (width, height) = match self.viewport_bound {
                None => screen_size,
                Some(ViewportBound::Vertical(h)) => (h * aspect, h),
                Some(ViewportBound::Horizontal(w)) => (w, w / aspect),
            };
            let Vec2 { x, y } = self
                .viewport_center
                .unwrap_or(vec2(width / 2., height / 2.));
            Mat4::orthographic_rh_gl(
                x - width / 2.,
                x + width / 2.,
                y + height / 2.,
                y - height / 2.,
                -1.,
                1.,
            )
        });
        self.batcher.draw(&mut **ctx, screen_mat, None);
    }

    pub fn set_viewport_center(&mut self, point: Vec2) {
        self.viewport_center = Some(point);
    }

    pub fn set_viewport_bound(&mut self, bound: ViewportBound) {
        self.viewport_bound = Some(bound);
    }

    /// Set screen projection matrix
    /// This override any other set_* functions, canvas.blit() will use
    /// this exact matrix.
    /// Might be useful for custom cameras implementation.
    pub fn set_override_matrix(&mut self, mat: Mat4) {
        self.matrix_override = Some(mat);
    }
}

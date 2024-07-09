use crate::{draw_calls_batcher::DrawCallsBatcher, text};

use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

pub struct SpriteBatcher {
    pub(crate) quad_ctx: Arc<Mutex<Box<miniquad::Context>>>,
    pub(crate) fonts_storage: Arc<Mutex<text::FontsStorage>>,
    pub(crate) batcher: DrawCallsBatcher,
    pub(crate) axis: Axis,
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
        }
    }

    pub fn clear(&mut self) {
        self.batcher
            .clear(self.quad_ctx.lock().unwrap().as_mut())
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

    pub fn wtf(&mut self, mat: crate::math::Mat4) {
        self.batcher.push_model_matrix(mat);
    }

    pub fn draw(&mut self) {
        let mut ctx = self.quad_ctx.lock().unwrap();

        let (width, height) = miniquad::window::screen_size();

        let screen_mat = glam::Mat4::orthographic_rh_gl(0., width, height, 0., -1., 1.);
        self.batcher.draw(&mut **ctx, screen_mat, None);
    }

    pub fn draw2(&mut self, camera: &crate::camera::Camera) {
        let mut ctx = self.quad_ctx.lock().unwrap();

        let (proj, view) = camera.proj_view();
        self.batcher.draw(
            &mut **ctx,
            proj * view,
            camera.render_target.clone().map(|t| t.render_pass),
        );
    }

    // ERIC
    // I needed something like this method to get high dpi to work.
    pub fn draw3(&mut self, mat: crate::math::Mat4)
    {
        let mut ctx = self.quad_ctx.lock().unwrap();
        self.batcher.draw(&mut **ctx, mat, None);
    }

}

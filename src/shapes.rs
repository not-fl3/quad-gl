//! 2D shapes rendering.

use crate::color::Color;

pub use crate::draw_calls_batcher::Vertex;
use crate::{
    math::{vec2, Rect, Vec2, Vec3},
    sprite_batcher::{Axis, SpriteBatcher},
    text::Font,
    texture::Texture2D,
    QuadGl,
};

use std::sync::{Arc, Mutex};

mod circle;
mod line;
mod rectangle;
mod sprite;
mod text;
mod triangle;

pub use circle::Circle;
pub use line::Line;
pub use rectangle::Rectangle;
pub use sprite::Sprite;
pub use text::Text;
pub use triangle::Triangle;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrawMode {
    Triangles,
    Lines,
}

#[derive(Clone, Debug)]
enum DrawStyle {
    Solid,
    Lines { thickness: f32 },
}

#[derive(Clone, Debug)]
pub struct DrawParams {
    pub color: Color,
    pub draw_style: DrawStyle,
    /// Rotation in radians
    pub rotation: f32,
    /// Rotate around this point.
    /// When `None`, rotate around the texture's center.
    /// When `Some`, the coordinates are in screen-space.
    /// E.g. pivot (0,0) rotates around the top left corner of the screen, not of the
    /// texture.
    pub pivot: Option<Vec2>,
    pub axis: Axis,
}
impl Default for DrawParams {
    fn default() -> DrawParams {
        DrawParams {
            color: crate::color::WHITE,
            draw_style: DrawStyle::Solid,
            rotation: 0.0,
            pivot: None,
            axis: Axis::Z,
        }
    }
}
impl From<Color> for DrawParams {
    fn from(color: Color) -> DrawParams {
        DrawParams {
            color,
            draw_style: DrawStyle::Solid,
            rotation: 0.0,
            pivot: None,
            axis: Axis::Z,
        }
    }
}

pub trait Mesher {
    fn quad_ctx(&self) -> &Arc<Mutex<Box<miniquad::Context>>>;
    fn fonts_storage(&self) -> &Arc<Mutex<crate::text::FontsStorage>>;
    fn texture(&mut self, texture: Option<miniquad::TextureId>);
    fn draw_mode(&mut self, mode: DrawMode);
    fn geometry(&mut self, vertices: &[Vertex], indices: &[u16]);
}

pub trait Draw {
    fn draw(self, s: &mut impl Mesher, pos: Vec2, p: impl Into<DrawParams>);
}

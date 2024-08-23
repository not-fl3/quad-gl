//! 2D shapes rendering.

use crate::color::Color;

use crate::{
    draw_calls_batcher::{DrawMode, Vertex},
    math::{vec2, Rect, Vec2, Vec3},
    sprite_batcher::{Axis, SpriteBatcher},
    text::Font,
    texture::Texture2D,
};

use std::sync::Arc;

mod line;
mod rectangle;
mod sprite;
mod text;
mod triangle;

pub use line::Line;
pub use rectangle::Rectangle;
pub use sprite::Sprite;
pub use text::Text;
pub use triangle::Triangle;

pub trait Draw {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>);
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
    pub rotation: f32,
}
impl From<Color> for DrawParams {
    fn from(color: Color) -> DrawParams {
        DrawParams {
            color,
            draw_style: DrawStyle::Solid,
            rotation: 0.0,
        }
    }
}

use crate::{
    color::Color,
    math::{vec2, Rect, Vec2},
    shapes::{Draw, DrawMode, DrawParams, DrawStyle, Vertex, Mesher},
    sprite_batcher::{Axis, SpriteBatcher},
    texture::Texture2D,
};

use std::sync::Arc;

pub struct Sprite<'a> {
    pub texture: &'a Arc<Texture2D>,
    pub size: Option<Vec2>,
    /// Part of texture to draw. If None - draw the whole texture.
    /// Good use example: drawing an image from texture atlas.
    /// Is None by default
    pub source: Option<Rect>,

    /// Mirror on the X axis
    pub flip_x: bool,

    /// Mirror on the Y axis
    pub flip_y: bool,
}
impl<'a> Sprite<'a> {
    pub fn new(texture: &'a Arc<Texture2D>) -> Sprite {
        Sprite {
            texture,
            size: None,
            source: None,
            flip_x: false,
            flip_y: false,
        }
    }
    pub fn size(self, size: Vec2) -> Sprite<'a> {
        Sprite {
            size: Some(size),
            ..self
        }
    }
}
impl<'a> Draw for Sprite<'a> {
    fn draw(self, s: &mut impl Mesher, pos: Vec2, p: impl Into<DrawParams>) {
        let params = p.into();
        let Vec2 { x, y } = pos;
        let (width, height) = {
            let quad_ctx = s.quad_ctx().lock().unwrap();
            quad_ctx.texture_size(self.texture.raw_miniquad_id())
        };
        let (width, height) = (width as f32, height as f32);
        let Rect {
            x: mut sx,
            y: mut sy,
            w: mut sw,
            h: mut sh,
        } = self.source.unwrap_or_else(|| Rect {
            x: 0.,
            y: 0.,
            w: width,
            h: height,
        });

        // let texture = context
        //     .texture_batcher
        //     .get(texture)
        //     .map(|(batched_texture, uv)| {
        //         sx = ((sx / texture.width()) * uv.w + uv.x) * batched_texture.width();
        //         sy = ((sy / texture.height()) * uv.h + uv.y) * batched_texture.height();
        //         sw = (sw / texture.width()) * uv.w * batched_texture.width();
        //         sh = (sh / texture.height()) * uv.h * batched_texture.height();

        //         batched_texture
        //     })
        //     .unwrap_or(texture.clone());

        let (mut w, mut h) = match self.size {
            Some(dst) => (dst.x, dst.y),
            _ => (sw, sh),
        };
        let mut x = x;
        let mut y = y;
        if self.flip_x {
            x = x + w;
            w = -w;
        }
        if self.flip_y {
            y = y + h;
            h = -h;
        }

        let pivot = params.pivot.unwrap_or(vec2(x + w / 2., y + h / 2.));
        let m = pivot;
        let p = [
            vec2(x, y) - pivot,
            vec2(x + w, y) - pivot,
            vec2(x + w, y + h) - pivot,
            vec2(x, y + h) - pivot,
        ];
        let r = params.rotation;
        let p = [
            vec2(
                p[0].x * r.cos() - p[0].y * r.sin(),
                p[0].x * r.sin() + p[0].y * r.cos(),
            ) + m,
            vec2(
                p[1].x * r.cos() - p[1].y * r.sin(),
                p[1].x * r.sin() + p[1].y * r.cos(),
            ) + m,
            vec2(
                p[2].x * r.cos() - p[2].y * r.sin(),
                p[2].x * r.sin() + p[2].y * r.cos(),
            ) + m,
            vec2(
                p[3].x * r.cos() - p[3].y * r.sin(),
                p[3].x * r.sin() + p[3].y * r.cos(),
            ) + m,
        ];
        let color = params.color;
        #[rustfmt::skip]
        let vertices = [
            Vertex::new(p[0].x, p[0].y, 0.,  sx      /width,  sy      /height, color),
            Vertex::new(p[1].x, p[1].y, 0., (sx + sw)/width,  sy      /height, color),
            Vertex::new(p[2].x, p[2].y, 0., (sx + sw)/width, (sy + sh)/height, color),
            Vertex::new(p[3].x, p[3].y, 0.,  sx      /width, (sy + sh)/height, color),
        ];
        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

        s.texture(Some(self.texture.raw_miniquad_id()));
        s.draw_mode(DrawMode::Triangles);
        s.geometry(&vertices, &indices);
    }
}

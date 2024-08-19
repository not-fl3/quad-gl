//! 2D shapes rendering.

use crate::color::Color;

use crate::{
    draw_calls_batcher::{DrawMode, Vertex},
    math::{vec2, Rect, Vec2, Vec3},
    sprite_batcher::{Axis, SpriteBatcher},
    text::{self, Font},
    texture::Texture2D,
};

use std::sync::Arc;

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
pub struct Rectangle {
    pub width: f32,
    pub height: f32,
}
impl Rectangle {
    pub fn new(width: f32, height: f32) -> Rectangle {
        Rectangle { width, height }
    }
}
impl Draw for Rectangle {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>) {
        let p = p.into();

        let Vec2 { x, y } = pos;
        let (w, h) = (self.width, self.height);
        match p.draw_style {
            DrawStyle::Solid => {
                #[rustfmt::skip]
                let vertices = [
                    Vertex::new(x    , y    , 0., 0.0, 1.0, p.color),
                    Vertex::new(x + w, y    , 0., 1.0, 0.0, p.color),
                    Vertex::new(x + w, y + h, 0., 1.0, 1.0, p.color),
                    Vertex::new(x    , y + h, 0., 0.0, 0.0, p.color),
                ];
                let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
                s.gl().texture(None);
                s.gl().draw_mode(DrawMode::Triangles);
                s.gl().geometry(&vertices, &indices);
            }
            DrawStyle::Lines { thickness } => {
                let t = thickness / 2.;

                #[rustfmt::skip]
                let vertices = [
                    Vertex::new(x    , y    , 0., 0.0, 1.0, p.color),
                    Vertex::new(x + w, y    , 0., 1.0, 0.0, p.color),
                    Vertex::new(x + w, y + h, 0., 1.0, 1.0, p.color),
                    Vertex::new(x    , y + h, 0., 0.0, 0.0, p.color),
                    //inner rectangle
                    Vertex::new(x + t    , y + t    , 0., 0.0, 0.0, p.color),
                    Vertex::new(x + w - t, y + t    , 0., 0.0, 0.0, p.color),
                    Vertex::new(x + w - t, y + h - t, 0., 0.0, 0.0, p.color),
                    Vertex::new(x + t    , y + h - t, 0., 0.0, 0.0, p.color),
                ];
                let indices: [u16; 24] = [
                    0, 1, 4, 1, 4, 5, 1, 5, 6, 1, 2, 6, 3, 7, 2, 2, 7, 6, 0, 4, 3, 3, 4, 7,
                ];

                s.gl().texture(None);
                s.gl().draw_mode(DrawMode::Triangles);
                s.gl().geometry(&vertices, &indices);
            }
        }
    }
}
pub struct Sprite<'a> {
    pub texture: &'a Arc<Texture2D>,
    pub dest_size: Option<Vec2>,
    /// Part of texture to draw. If None - draw the whole texture.
    /// Good use example: drawing an image from texture atlas.
    /// Is None by default
    pub source: Option<Rect>,

    /// Rotation in radians
    pub rotation: f32,

    /// Mirror on the X axis
    pub flip_x: bool,

    /// Mirror on the Y axis
    pub flip_y: bool,

    /// Rotate around this point.
    /// When `None`, rotate around the texture's center.
    /// When `Some`, the coordinates are in screen-space.
    /// E.g. pivot (0,0) rotates around the top left corner of the screen, not of the
    /// texture.
    pub pivot: Option<Vec2>,
}
impl<'a> Sprite<'a> {
    pub fn new(texture: &'a Arc<Texture2D>) -> Sprite {
        Sprite {
            texture,
            dest_size: None,
            source: None,
            rotation: 0.,
            pivot: None,
            flip_x: false,
            flip_y: false,
        }
    }
}
impl<'a> Draw for Sprite<'a> {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>) {
        let params = p.into();
        let Vec2 { x, y } = pos;
        let (width, height) = {
            let quad_ctx = s.quad_ctx.lock().unwrap();
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

        let (mut w, mut h) = match self.dest_size {
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

        let pivot = self.pivot.unwrap_or(vec2(x + w / 2., y + h / 2.));
        let m = pivot;
        let p = [
            vec2(x, y) - pivot,
            vec2(x + w, y) - pivot,
            vec2(x + w, y + h) - pivot,
            vec2(x, y + h) - pivot,
        ];
        let r = self.rotation;
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

        s.gl().texture(Some(self.texture.raw_miniquad_id()));
        s.gl().draw_mode(DrawMode::Triangles);
        s.gl().geometry(&vertices, &indices);
    }
}
pub struct Triangle {
    pub v1: Vec2,
    pub v2: Vec2,
    pub v3: Vec2,
}
impl Triangle {
    pub fn new(v1: Vec2, v2: Vec2, v3: Vec2) -> Triangle {
        Triangle { v1, v2, v3 }
    }
}
impl Draw for Triangle {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>) {
        let p = p.into();
        let mut vertices = Vec::<Vertex>::with_capacity(3);

        let v1 = self.v1 + pos;
        let v2 = self.v2 + pos;
        let v3 = self.v3 + pos;
        vertices.push(Vertex::new(v1.x, v1.y, 0., 0., 0., p.color));
        vertices.push(Vertex::new(v2.x, v2.y, 0., 0., 0., p.color));
        vertices.push(Vertex::new(v3.x, v3.y, 0., 0., 0., p.color));
        let indices: [u16; 3] = [0, 1, 2];

        s.gl().texture(None);
        s.gl().draw_mode(DrawMode::Triangles);
        s.gl().geometry(&vertices, &indices);
    }
}

pub struct Circle {
    pub radius: f32,
    pub sides: u32,
}

impl Circle {
    pub fn new(radius: f32) -> Circle {
        Circle { radius, sides: 20 }
    }
}

impl Draw for Circle {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>) {
        let Vec2 { x, y } = pos;
        let p = p.into();
        match p.draw_style {
            DrawStyle::Solid => {
                let mut vertices = Vec::<Vertex>::with_capacity(self.sides as usize + 2);
                let mut indices = Vec::<u16>::with_capacity(self.sides as usize * 3);

                let rot = p.rotation;
                vertices.push(Vertex::new(x, y, 0., 0., 0., p.color));
                for i in 0..self.sides + 1 {
                    let rx = (i as f32 / self.sides as f32 * std::f32::consts::PI * 2. + rot).cos();
                    let ry = (i as f32 / self.sides as f32 * std::f32::consts::PI * 2. + rot).sin();

                    let vertex = Vertex::new(
                        x + self.radius * rx,
                        y + self.radius * ry,
                        0.,
                        rx,
                        ry,
                        p.color,
                    );

                    vertices.push(vertex);

                    if i != self.sides {
                        indices.extend_from_slice(&[0, i as u16 + 1, i as u16 + 2]);
                    }
                }

                s.gl().texture(None);
                s.gl().draw_mode(DrawMode::Triangles);
                s.gl().geometry(&vertices, &indices);
            }
            DrawStyle::Lines { thickness } => {
                let rot = p.rotation.to_radians();

                for i in 0..self.sides {
                    let rx = (i as f32 / self.sides as f32 * std::f32::consts::PI * 2. + rot).cos();
                    let ry = (i as f32 / self.sides as f32 * std::f32::consts::PI * 2. + rot).sin();

                    let p0 = vec2(x + self.radius * rx, y + self.radius * ry);

                    let rx = ((i + 1) as f32 / self.sides as f32 * std::f32::consts::PI * 2. + rot)
                        .cos();
                    let ry = ((i + 1) as f32 / self.sides as f32 * std::f32::consts::PI * 2. + rot)
                        .sin();

                    let p1 = vec2(x + self.radius * rx, y + self.radius * ry);

                    Line::new(p0, p1, thickness).draw(s, Vec2::ZERO, p.clone());
                }
            }
        }
    }
}

struct Line {
    pub p0: Vec2,
    pub p1: Vec2,
    pub thickness: f32,
}

impl Line {
    pub fn new(p0: Vec2, p1: Vec2, thickness: f32) -> Line {
        Line { p0, p1, thickness }
    }
}
impl Draw for Line {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>) {
        let p = p.into();
        let Vec2 { x: x1, y: y1 } = self.p0 + pos;
        let Vec2 { x: x2, y: y2 } = self.p1 + pos;
        let dx = x2 - x1;
        let dy = y2 - y1;

        // https://stackoverflow.com/questions/1243614/how-do-i-calculate-the-normal-vector-of-a-line-segment

        let nx = -dy;
        let ny = dx;

        let tlen = (nx * nx + ny * ny).sqrt() / (self.thickness * 0.5);
        if tlen < std::f32::EPSILON {
            return;
        }
        let tx = nx / tlen;
        let ty = ny / tlen;

        let axis = s.axis;
        s.gl().texture(None);
        s.gl().draw_mode(DrawMode::Triangles);
        s.gl().geometry(
            &[
                vertex(x1 + tx, y1 + ty, 0., 0., p.color, axis),
                vertex(x1 - tx, y1 - ty, 0., 0., p.color, axis),
                vertex(x2 + tx, y2 + ty, 0., 0., p.color, axis),
                vertex(x2 - tx, y2 - ty, 0., 0., p.color, axis),
            ],
            &[0, 1, 2, 2, 1, 3],
        );
    }
}
#[derive(Debug, Clone)]
pub struct Text<'a, 'b> {
    pub text: &'a str,
    pub font: Option<&'b Font>,
    /// Base size for character height. The size in pixel used during font rasterizing.
    pub font_size: u16,
    /// The glyphs sizes actually drawn on the screen will be font_size * font_scale
    /// However with font_scale too different from 1.0 letters may be blurry
    pub font_scale: f32,
    /// Font X axis would be scaled by font_scale * font_scale_aspect
    /// and Y axis would be scaled by font_scale
    /// Default is 1.0
    pub font_scale_aspect: f32,
}

impl<'a, 'b> Text<'a, 'b> {
    pub fn new(text: &'a str, font_size: u16) -> Text<'a, 'b> {
        Text {
            text,
            font: None,
            font_size,
            font_scale: 1.0,
            font_scale_aspect: 1.0,
        }
    }
}

impl<'a, 'b> Draw for Text<'a, 'b> {
    fn draw(self, s: &mut SpriteBatcher, pos: Vec2, p: impl Into<DrawParams>) {
        let Vec2 { x, y } = pos;
        let p = p.into();
        let font = {
            let fonts = s.fonts_storage.lock().unwrap();
            self.font.unwrap_or_else(|| &fonts.default_font).clone()
        };

        let font_scale_x = self.font_scale * self.font_scale_aspect;
        let font_scale_y = self.font_scale;
        let dpi_scaling = miniquad::window::dpi_scale();

        let font_size = (self.font_size as f32 * dpi_scaling).ceil() as u16;

        let mut total_width = 0.;
        for character in self.text.chars() {
            if !font
                .characters
                .lock()
                .unwrap()
                .contains_key(&(character, font_size))
            {
                font.cache_glyph(character, font_size);
            }
            let mut atlas = font.atlas.lock().unwrap();
            let font_data = &font.characters.lock().unwrap()[&(character, font_size)];
            let glyph = atlas.get(font_data.sprite).unwrap().rect;
            let angle_rad = p.rotation;
            let angle_rad = 0.0f32;
            let left_coord = (font_data.offset_x as f32 * font_scale_x + total_width)
                * angle_rad.cos()
                + (glyph.h as f32 * font_scale_y + font_data.offset_y as f32 * font_scale_y)
                    * angle_rad.sin();
            let top_coord = (font_data.offset_x as f32 * font_scale_x + total_width)
                * angle_rad.sin()
                + (0.0 - glyph.h as f32 * font_scale_y - font_data.offset_y as f32 * font_scale_y)
                    * angle_rad.cos();

            total_width += font_data.advance * font_scale_x;

            let dest = Rect::new(
                left_coord / dpi_scaling as f32 + x,
                top_coord / dpi_scaling as f32 + y,
                glyph.w as f32 / dpi_scaling as f32 * font_scale_x,
                glyph.h as f32 / dpi_scaling as f32 * font_scale_y,
            );

            let source = Rect::new(
                glyph.x as f32,
                glyph.y as f32,
                glyph.w as f32,
                glyph.h as f32,
            );

            // let texture = {
            //     let mut ctx = self.quad_ctx.lock().unwrap();
            //     atlas.texture(&mut **ctx)
            // };
            // self.draw_texture_ex(
            //     &crate::texture::Texture2D {
            //         texture: TextureHandle::Unmanaged(texture),
            //     },
            //     dest.x,
            //     dest.y,
            //     self.color,
            //     crate::texture::DrawTextureParams {
            //         dest_size: Some(vec2(dest.w, dest.h)),
            //         source: Some(source),
            //         rotation: angle_rad,
            //         pivot: Option::Some(vec2(dest.x, dest.y)),
            //         ..Default::default()
            //     },
            // );
        }
    }
}

impl SpriteBatcher {
    // pub fn draw_hexagon(
    //     x: f32,
    //     y: f32,
    //     size: f32,
    //     border: f32,
    //     vertical: bool,
    //     border_color: Color,
    //     fill_color: Color,
    // ) {
    //     let rotation = if vertical { 90. } else { 0. };
    //     draw_poly(x, y, 6, size, rotation, fill_color);
    //     if border > 0. {
    //         draw_poly_lines(x, y, 6, size, rotation, border, border_color);
    //     }
    // }

    // pub fn draw_poly_lines(
    //     x: f32,
    //     y: f32,
    //     sides: u8,
    //     radius: f32,
    //     rotation: f32,
    //     thickness: f32,
    //     color: Color,
    // ) {
    //     let rot = rotation.to_radians();

    //     for i in 0..sides {
    //         let rx = (i as f32 / sides as f32 * std::f32::consts::PI * 2. + rot).cos();
    //         let ry = (i as f32 / sides as f32 * std::f32::consts::PI * 2. + rot).sin();

    //         let p0 = vec2(x + radius * rx, y + radius * ry);

    //         let rx = ((i + 1) as f32 / sides as f32 * std::f32::consts::PI * 2. + rot).cos();
    //         let ry = ((i + 1) as f32 / sides as f32 * std::f32::consts::PI * 2. + rot).sin();

    //         let p1 = vec2(x + radius * rx, y + radius * ry);

    //         draw_line(p0.x, p0.y, p1.x, p1.y, thickness, color);
    //     }
    // }

    pub fn draw_line_3d(&mut self, start: Vec3, end: Vec3, _: f32, color: Color) {
        let uv = [0., 0.];
        let color: [f32; 4] = color.into();
        let indices = [0, 1];

        let line = [
            ([start.x, start.y, start.z], uv, color),
            ([end.x, end.y, end.z], uv, color),
        ];
        self.gl().texture(None);
        self.gl().draw_mode(DrawMode::Lines);
        self.gl().geometry(&line[..], &indices);
    }
}
fn vertex(x: f32, y: f32, uv_x: f32, uv_y: f32, color: Color, axis: Axis) -> Vertex {
    match axis {
        Axis::X => Vertex::new(0., x, y, uv_x, uv_y, color),
        Axis::Y => Vertex::new(x, 0., y, uv_x, uv_y, color),
        Axis::Z => Vertex::new(x, y, 0., uv_x, uv_y, color),
    }
}

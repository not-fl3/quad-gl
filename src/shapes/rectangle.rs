use crate::{
    color::Color,
    math::Vec2,
    shapes::{Draw, DrawMode, DrawParams, DrawStyle, Mesher, Vertex},
    sprite_batcher::{Axis, SpriteBatcher},
};

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
    fn draw(self, s: &mut impl Mesher, pos: Vec2, p: impl Into<DrawParams>) {
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
                s.texture(None);
                s.draw_mode(DrawMode::Triangles);
                s.geometry(&vertices, &indices);
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

                s.texture(None);
                s.draw_mode(DrawMode::Triangles);
                s.geometry(&vertices, &indices);
            }
        }
    }
}

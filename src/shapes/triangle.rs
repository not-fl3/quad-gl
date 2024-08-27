use crate::{
    color::Color,
    math::Vec2,
    shapes::{Draw, DrawMode, DrawParams, Vertex, DrawStyle, Mesher},
    sprite_batcher::{Axis, SpriteBatcher},
};

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
    fn draw(self, s: &mut impl Mesher, pos: Vec2, p: impl Into<DrawParams>) {
        let p = p.into();
        let mut vertices = Vec::<Vertex>::with_capacity(3);

        let v1 = self.v1 + pos;
        let v2 = self.v2 + pos;
        let v3 = self.v3 + pos;
        vertices.push(Vertex::new(v1.x, v1.y, 0., 0., 0., p.color));
        vertices.push(Vertex::new(v2.x, v2.y, 0., 0., 0., p.color));
        vertices.push(Vertex::new(v3.x, v3.y, 0., 0., 0., p.color));
        let indices: [u16; 3] = [0, 1, 2];

        s.texture(None);
        s.draw_mode(DrawMode::Triangles);
        s.geometry(&vertices, &indices);
    }
}

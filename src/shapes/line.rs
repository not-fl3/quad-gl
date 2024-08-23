use crate::{
    color::Color,
    math::Vec2,
    shapes::{Draw, DrawMode, DrawParams, Vertex},
    sprite_batcher::{Axis, SpriteBatcher},
};

pub struct Line {
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

fn vertex(x: f32, y: f32, uv_x: f32, uv_y: f32, color: Color, axis: Axis) -> Vertex {
    match axis {
        Axis::X => Vertex::new(0., x, y, uv_x, uv_y, color),
        Axis::Y => Vertex::new(x, 0., y, uv_x, uv_y, color),
        Axis::Z => Vertex::new(x, y, 0., uv_x, uv_y, color),
    }
}

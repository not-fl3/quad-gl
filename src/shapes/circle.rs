use crate::{
    color::Color,
    math::{Vec2, vec2},
    shapes::{Draw, DrawMode, DrawParams, DrawStyle, Mesher, Vertex, Line},
    sprite_batcher::{Axis, SpriteBatcher},
};

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
    fn draw(self, s: &mut impl Mesher, pos: Vec2, p: impl Into<DrawParams>) {
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

                s.texture(None);
                s.draw_mode(DrawMode::Triangles);
                s.geometry(&vertices, &indices);
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

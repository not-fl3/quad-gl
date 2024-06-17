
use crate::color::Color;

use crate::{
    math::{vec2, Rect, Vec2, Vec3},
    draw_calls_batcher::{DrawMode, Vertex},
    sprite_batcher::{Axis, SpriteBatcher},
};

#[derive(Debug, Clone)]
 pub struct DrawRectangleParams2 {
     /// Rotation in radians
     pub rotation: f32,
     /// Rotate around this point.
     /// When `None`, rotate around the rectangle's center.
     /// When `Some`, the coordinates are in world-space.
     /// E.g. pivot (0,0) rotates around the top left corner of the world, not of the
     /// rectangle.
     pub pivot: Option<Vec2>,
     /// Rectangle will be filled with gradient.
     /// Corner colors are specified in order: `[top_left, top_right, bottom_right, bottom_left]`
     /// Overriders `color`.
     pub gradient: Option<[Color; 4]>,
     /// Color of the rectangle. Used if `gradient` is `None`.
     pub color: Color,
     /// If greater than 0.0, draws a rectangle outline with given `line_thickness`
     pub line_thickness: f32,
     /// Horizontal and vertical skew proportions
     pub skew: Vec2,
     /// Radius of rectangle's corners
     pub border_radius: f32,
     /// Number of segments used for drawing each corner
     /// Ignored if `border_radius` is 0.0
     pub border_radius_segments: u8,
 }

 impl Default for DrawRectangleParams2 {
     fn default() -> DrawRectangleParams2 {
         DrawRectangleParams2 {
             gradient: None,
             rotation: 0.,
             color: Color::new(1.0, 1.0, 1.0, 1.0),
             line_thickness: 0.,
             pivot: None,
             skew: Vec2::ZERO,
             border_radius: 0.0,
             border_radius_segments: 5,
         }
     }
 }

impl SpriteBatcher {

     fn mix_colors(first: &Color, second: &Color, amount: f32) -> Color {
         let amount_s = 1.0 - amount;
         Color::new(
             first.r * amount + second.r * amount_s,
             first.g * amount + second.g * amount_s,
             first.b * amount + second.b * amount_s,
             first.a * amount + second.a * amount_s,
         )
     }

     /// Note: last `Vertex` in returned `Vec` is center
     fn rounded_rect(
         quart_vertices: u8,
         rect: Rect,
         border_radius: f32,
         gradient: Option<&[Color; 4]>,
         center_color: Color,
         generate_indices: bool,
     ) -> (Vec<Vertex>, Vec<u16>) {
         use std::f32::consts::PI;
         let Rect { x, y, w, h } = rect;
         let mut indices: Vec<u16> = vec![];

         let rc = rect.center();
         let c0 = vec2(x + w - border_radius, y + border_radius);
         let c1 = vec2(x + border_radius, y + border_radius);
         let c2 = vec2(x + border_radius, y + h - border_radius);
         let c3 = vec2(x + w - border_radius, y + h - border_radius);

         let mut vertices: Vec<Vertex> = vec![];

         let v_num = quart_vertices * 4;

         vertices.extend((0..v_num).map(|i| {
             if generate_indices {
                 if i < v_num - 1 {
                     indices.extend([v_num as u16, (i) as u16, (i + 1) as u16]);
                 } else {
                     indices.extend([v_num as u16, (i) as u16, 0]);
                 }
             }
             let (r, angle_cs) = match i {
                 i if i >= quart_vertices * 3 => {
                     // Top right quarter circle
                     let angle = ((i - quart_vertices * 3) as f32 / (quart_vertices - 1) as f32) * PI
                         / 2.
                         + (3.) * PI / 2.;
                     let angle_cs = vec2(angle.cos(), angle.sin());
                     let r = c0 + (angle_cs * border_radius);
                     (r, angle_cs)
                 }
                 i if i >= quart_vertices * 2 => {
                     // Top left quarter circle
                     let angle =
                         (i - quart_vertices * 2) as f32 / (quart_vertices - 1) as f32 * (PI / 2.) + PI;
                     let angle_cs = vec2(angle.cos(), angle.sin());
                     let r = c1 + (angle_cs * border_radius);
                     (r, angle_cs)
                 }
                 i if i >= quart_vertices => {
                     // Bottom right quarter circle
                     let angle =
                         (i - quart_vertices) as f32 / (quart_vertices - 1) as f32 * PI / 2. + PI / 2.;
                     let angle_cs = vec2(angle.cos(), angle.sin());
                     let r = c2 + (angle_cs * border_radius);
                     (r, angle_cs)
                 }
                 i => {
                     // Bottom left quarter circle
                     let angle = i as f32 / (quart_vertices - 1) as f32 * PI / 2.;
                     let angle_cs = vec2(angle.cos(), angle.sin());
                     let r = c3 + (angle_cs * border_radius);
                     (r, angle_cs)
                 }
             };

             let color = if let Some(gradient) = gradient {
                 let h_rel = ((x + w) - r.x) / w;
                 let v_rel = ((y + h) - r.y) / h;

                 // Seems to work:
                 // mix top left and top right colors based on horizontal distance
                 let color_top = Self::mix_colors(&gradient[0], &gradient[1], h_rel);
                 // mix bot left and bot right colors based on horizontal distance
                 let color_bot = Self::mix_colors(&gradient[3], &gradient[2], h_rel);
                 // mix results based on vertical distance
                 Self::mix_colors(&color_top, &color_bot, v_rel)
             } else {
                 center_color
             };

             Vertex::new(r.x, r.y, 0., angle_cs.x, angle_cs.y, color)
         }));

         vertices.push(Vertex::new(rc.x, rc.y, 0., 0., 0., center_color));

         (vertices, indices)
     }
     fn skew_vertices(vertices: &mut [Vertex], skew: Vec2, pivot: Vec2) {
         vertices.iter_mut().for_each(|v| {
             let p = vec2(v.pos[0] - pivot.x, v.pos[1] - pivot.y);

             v.pos[0] = p.x + (skew.x * p.y) + pivot.x;
             v.pos[1] = p.y + (skew.y * p.x) + pivot.y;
         });
     }
     fn rotate_vertices(vertices: &mut [Vertex], rot: f32, pivot: Vec2) {
         let sin = rot.sin();
         let cos = rot.cos();
         vertices.iter_mut().for_each(|v| {
             let p = vec2(v.pos[0] - pivot.x, v.pos[1] - pivot.y);

             v.pos[0] = p.x * cos - p.y * sin + pivot.x;
             v.pos[1] = p.x * sin + p.y * cos + pivot.y;
         });
     }

     /// Draws a rectangle with its top-left corner at `[x, y]` with size `[w, h]` (width going to
     /// the right, height going down), with a given `params`.
     pub fn draw_rectangle_ex2(&mut self, x: f32, y: f32, w: f32, h: f32, param: &DrawRectangleParams2) {
         let center = vec2(x + w / 2., y + h / 2.);
         let p = [
             vec2(x, y),
             vec2(x + w, y),
             vec2(x + w, y + h),
             vec2(x, y + h),
         ];

         let g = &param.gradient;
         let c = param.color;
         let t = param.line_thickness;

         let center_color = g.map_or(c, |g| {
             Color::new(
                 g.iter().fold(0.0, |a, c| a + c.r) / 4.0,
                 g.iter().fold(0.0, |a, c| a + c.g) / 4.0,
                 g.iter().fold(0.0, |a, c| a + c.b) / 4.0,
                 g.iter().fold(0.0, |a, c| a + c.a) / 4.0,
             )
         });

         let (mut outer_vertices, outer_indices): (Vec<Vertex>, Vec<u16>) = if param.border_radius > 0.0
         {
             // Rectangle with rounded corners
             Self::rounded_rect(
                 param.border_radius_segments * 2,
                 Rect::new(x, y, w, h),
                 param.border_radius,
                 g.as_ref(),
                 center_color,
                 true,
             )
         } else {
             // Regular rectangle
             (
                 vec![
                     Vertex::new(p[0].x, p[0].y, 0., 0., 0., g.map_or(c, |g| g[0])),
                     Vertex::new(p[1].x, p[1].y, 0., 1., 0., g.map_or(c, |g| g[1])),
                     Vertex::new(p[2].x, p[2].y, 0., 1., 1., g.map_or(c, |g| g[2])),
                     Vertex::new(p[3].x, p[3].y, 0., 0., 1., g.map_or(c, |g| g[3])),
                 ],
                 vec![0, 1, 2, 0, 2, 3],
             )
         };

         if param.skew != Vec2::ZERO {
             Self::skew_vertices(&mut outer_vertices, param.skew, center);
         }

         let pivot = param.pivot.unwrap_or(center);

         if param.rotation != 0. {
             Self::rotate_vertices(&mut outer_vertices, param.rotation, pivot);
         };

         let mut indices: Vec<u16>;
         if t > 0. {
             // Draw rectangle outline
             let mut inner_vertices: Vec<Vertex> = if param.border_radius > 0.0 {
                 // Rectangle with rounded corners
                 let mut inner_vert = Self::rounded_rect(
                     param.border_radius_segments * 2,
                     Rect::new(x + t, y + t, w - 2. * t, h - 2. * t),
                     param.border_radius * (w - 2. * t) / w,
                     g.as_ref(),
                     center_color,
                     false,
                 )
                 .0;
                 // We don't need center vertices when drawing outline
                 outer_vertices.pop();
                 inner_vert.pop();
                 inner_vert
             } else {
                 // Regular rectangle
                 vec![
                     Vertex::new(p[0].x + t, p[0].y + t, 0., 0., 0., g.map_or(c, |g| g[0])),
                     Vertex::new(p[1].x - t, p[1].y + t, 0., 1., 0., g.map_or(c, |g| g[1])),
                     Vertex::new(p[2].x - t, p[2].y - t, 0., 1., 1., g.map_or(c, |g| g[2])),
                     Vertex::new(p[3].x + t, p[3].y - t, 0., 0., 1., g.map_or(c, |g| g[3])),
                 ]
             };

             if param.skew != Vec2::ZERO {
                 Self::skew_vertices(&mut inner_vertices, param.skew, center);
             }
             if param.rotation != 0. {
                 Self::rotate_vertices(&mut inner_vertices, param.rotation, pivot);
             };

             let v_len = outer_vertices.len() as u16;

             // Merge outer and innver vertices
             outer_vertices.extend(&inner_vertices);

             // Generate indices
             indices = vec![];
             for i in 0..v_len {
                 indices.extend([i, ((i + 1) % v_len as u16), v_len + (i as u16)]);
                 indices.extend([
                     i + v_len as u16,
                     (i + 1) % v_len as u16,
                     v_len + ((i + 1) % v_len) as u16,
                 ]);
             }
         } else {
             indices = outer_indices;
         };

         self.gl().texture(None);
         self.gl().draw_mode(DrawMode::Triangles);
         self.gl().geometry(&outer_vertices, &indices);
     }
}



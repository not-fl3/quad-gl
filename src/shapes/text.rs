use crate::{
    color::Color,
    math::{vec2, Vec2},
    shapes::{Draw, DrawMode, DrawParams, DrawStyle, Mesher, Rect, Sprite, Vertex},
    sprite_batcher::{Axis, SpriteBatcher},
    text::Font,
};

use std::sync::Arc;

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
    fn draw(self, s: &mut impl Mesher, pos: Vec2, p: impl Into<DrawParams>) {
        let Vec2 { x, y } = pos;
        let p = p.into();
        let font = {
            let fonts = s.fonts_storage().lock().unwrap();
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

            let (texture, w, h) = {
                let mut ctx = s.quad_ctx().lock().unwrap();
                atlas.texture(&mut **ctx)
            };
            Sprite {
                dest_size: Some(vec2(dest.w, dest.h)),
                source: Some(source),
                rotation: angle_rad,
                pivot: Option::Some(vec2(dest.x, dest.y)),
                ..Sprite::new(&Arc::new(crate::texture::Texture2D::from_miniquad_id(
                    texture, w as _, h as _,
                )))
            }
            .draw(s, vec2(dest.x, dest.y), p.color);
        }
    }
}

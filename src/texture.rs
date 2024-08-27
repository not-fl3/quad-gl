//! Loading and rendering textures. Also render textures, per-pixel image manipulations.

use crate::{color::Color, image, math::Rect, text::atlas::SpriteKey, Error};

use glam::{vec2, Vec2};

pub use miniquad::FilterMode;

use std::sync::{Arc, Mutex};

use crate::sprite_batcher::SpriteBatcher;

/// Image, data stored in CPU memory
#[derive(Clone)]
pub struct Image {
    pub bytes: Vec<u8>,
    pub width: u16,
    pub height: u16,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes.len()", &self.bytes.len())
            .finish()
    }
}

impl Image {
    /// Creates an empty Image.
    ///
    /// ```
    /// # use macroquad::prelude::*;
    /// let image = Image::empty();
    /// ```
    pub fn empty() -> Image {
        Image {
            width: 0,
            height: 0,
            bytes: vec![],
        }
    }

    /// Creates an Image filled with the provided [Color].
    pub fn gen_image_color(width: u16, height: u16, color: Color) -> Image {
        let mut bytes = vec![0; width as usize * height as usize * 4];
        for i in 0..width as usize * height as usize {
            bytes[i * 4 + 0] = (color.r * 255.) as u8;
            bytes[i * 4 + 1] = (color.g * 255.) as u8;
            bytes[i * 4 + 2] = (color.b * 255.) as u8;
            bytes[i * 4 + 3] = (color.a * 255.) as u8;
        }
        Image {
            width,
            height,
            bytes,
        }
    }

    /// Updates this image from a slice of [Color]s.
    pub fn update(&mut self, colors: &[Color]) {
        assert!(self.width as usize * self.height as usize == colors.len());

        for i in 0..colors.len() {
            self.bytes[i * 4] = (colors[i].r * 255.) as u8;
            self.bytes[i * 4 + 1] = (colors[i].g * 255.) as u8;
            self.bytes[i * 4 + 2] = (colors[i].b * 255.) as u8;
            self.bytes[i * 4 + 3] = (colors[i].a * 255.) as u8;
        }
    }

    /// Returns the width of this image.
    pub fn width(&self) -> usize {
        self.width as usize
    }

    /// Returns the height of this image.
    pub fn height(&self) -> usize {
        self.height as usize
    }

    /// Returns this image's data as a slice of 4-byte arrays.
    pub fn get_image_data(&self) -> &[[u8; 4]] {
        use std::slice;

        unsafe {
            slice::from_raw_parts(
                self.bytes.as_ptr() as *const [u8; 4],
                self.width as usize * self.height as usize,
            )
        }
    }

    /// Returns this image's data as a mutable slice of 4-byte arrays.
    pub fn get_image_data_mut(&mut self) -> &mut [[u8; 4]] {
        use std::slice;

        unsafe {
            slice::from_raw_parts_mut(
                self.bytes.as_mut_ptr() as *mut [u8; 4],
                self.width as usize * self.height as usize,
            )
        }
    }

    /// Modifies a pixel [Color] in this image.
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Color) {
        let width = self.width;

        self.get_image_data_mut()[(y * width as u32 + x) as usize] = color.into();
    }

    /// Returns a pixel [Color] from this image.
    pub fn get_pixel(&self, x: u32, y: u32) -> Color {
        self.get_image_data()[(y * self.width as u32 + x) as usize].into()
    }

    /// Returns an Image from a rect inside this image.
    pub fn sub_image(&self, rect: Rect) -> Image {
        let width = rect.w as usize;
        let height = rect.h as usize;
        let mut bytes = vec![0; width * height * 4];

        let x = rect.x as usize;
        let y = rect.y as usize;
        let mut n = 0;
        for y in y..y + height {
            for x in x..x + width {
                bytes[n] = self.bytes[y * self.width as usize * 4 + x * 4 + 0];
                bytes[n + 1] = self.bytes[y * self.width as usize * 4 + x * 4 + 1];
                bytes[n + 2] = self.bytes[y * self.width as usize * 4 + x * 4 + 2];
                bytes[n + 3] = self.bytes[y * self.width as usize * 4 + x * 4 + 3];
                n += 4;
            }
        }
        Image {
            width: width as u16,
            height: height as u16,
            bytes,
        }
    }

    /// Saves this image as a PNG file.
    pub fn export_png(&self, path: &str) {
        let mut bytes = vec![0; self.width as usize * self.height as usize * 4];

        // flip the image before saving
        for y in 0..self.height as usize {
            for x in 0..self.width as usize * 4 {
                bytes[y * self.width as usize * 4 + x] =
                    self.bytes[(self.height as usize - y - 1) * self.width as usize * 4 + x];
            }
        }

        // image::save_buffer(
        //     path,
        //     &bytes[..],
        //     self.width as _,
        //     self.height as _,
        //     image::ColorType::Rgba8,
        // )
        // .unwrap();
        unimplemented!()
    }
}

#[derive(Clone, Debug)]
pub struct RenderTarget {
    pub texture: Arc<Texture2D>,
    pub render_pass: miniquad::RenderPass,
}

impl RenderTarget {
    pub fn delete(&self) {
        // let context = get_quad_ctx();
        // context.delete_render_pass(self.render_pass);
    }
}

/// Get pixel data from screen buffer and return an Image (screenshot)
// pub fn get_screen_data() -> Image {
//     unsafe {
//         crate::window::get_internal_gl().flush();
//     }

//     let context = get_context();

//     let texture = Texture2D::from_miniquad_texture(get_quad_ctx().new_render_texture(
//         miniquad::TextureParams {
//             width: context.screen_width as _,
//             height: context.screen_height as _,
//             ..Default::default()
//         },
//     ));

//     texture.grab_screen();

//     texture.get_texture_data()
// }

/// Texture, data stored in GPU memory
#[derive(Debug, PartialEq)]
pub struct Texture2D {
    pub(crate) texture: miniquad::TextureId,
    pub(crate) width: u16,
    pub(crate) height: u16,
}

impl Texture2D {
    pub fn from_miniquad_id(texture: miniquad::TextureId, width: u16, height: u16) -> Texture2D {
        Texture2D {
            texture,
            width,
            height,
        }
    }
    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }
}

impl crate::QuadGl {
    /// Creates a Texture2D from a slice of bytes that contains an encoded image.
    ///
    /// If `format` is None, it will make an educated guess on the
    /// [ImageFormat][image::ImageFormat].
    ///
    /// # Example
    /// ```
    /// # use macroquad::prelude::*;
    /// # #[macroquad::main("test")]
    /// # async fn main() {
    /// # let texture = Texture2D::from_file(include_bytes!("../examples/rust.png"));
    /// # }
    /// ```
    pub fn load_texture(&self, bytes: &[u8]) -> Arc<Texture2D> {
        let img = image::decode(bytes).unwrap_or_else(|_| panic!());

        self.from_rgba8(img.width as _, img.height as _, &img.data)
    }

    pub fn render_target(&self, width: u16, height: u16) -> RenderTarget {
        let mut quad_ctx = self.quad_ctx.lock().unwrap();

        let texture = quad_ctx.new_render_texture(miniquad::TextureParams {
            width: width as _,
            height: height as _,
            ..Default::default()
        });
        let depth_img = quad_ctx.new_render_texture(miniquad::TextureParams {
            width: width as _,
            height: height as _,
            format: miniquad::TextureFormat::Depth,
            ..Default::default()
        });

        let render_pass = quad_ctx.new_render_pass(texture, Some(depth_img));
        let texture = Arc::new(Texture2D {
            texture,
            width,
            height,
        });

        RenderTarget {
            texture,
            render_pass,
        }
    }

    /// Creates a Texture2D from an [Image].
    pub fn from_image(&self, image: &Image) -> Arc<Texture2D> {
        self.from_rgba8(image.width, image.height, &image.bytes)
    }

    /// Creates a Texture2D from a slice of bytes in an R,G,B,A sequence,
    /// with the given width and height.
    ///
    /// # Example
    ///
    /// ```
    /// # use macroquad::prelude::*;
    /// # #[macroquad::main("test")]
    /// # async fn main() {
    /// // Create a 2x2 texture from a byte slice with 4 rgba pixels
    /// let bytes: Vec<u8> = vec![255, 0, 0, 192, 0, 255, 0, 192, 0, 0, 255, 192, 255, 255, 255, 192];
    /// let texture = Texture2D::from_rgba8(2, 2, &bytes);
    /// # }
    /// ```
    pub fn from_rgba8(&self, width: u16, height: u16, bytes: &[u8]) -> Arc<Texture2D> {
        let mut quad_ctx = self.quad_ctx.lock().unwrap();
        let texture = quad_ctx.new_texture_from_rgba8(width, height, bytes);

        let texture = Arc::new(Texture2D {
            texture,
            width,
            height,
        });

        //ctx.texture_batcher.add_unbatched(&texture);

        texture
    }
}

// impl Texture2D {
//     /// Uploads [Image] data to this texture.
//     pub fn update(&self, image: &Image) {
//         let ctx = get_quad_ctx();
//         let (width, height) = ctx.texture_size(self.raw_miniquad_id());

//         assert_eq!(width, image.width as u32);
//         assert_eq!(height, image.height as u32);

//         ctx.texture_update(self.raw_miniquad_id(), &image.bytes);
//     }

//     /// Uploads [Image] data to part of this texture.
//     pub fn update_part(
//         &self,
//         image: &Image,
//         x_offset: i32,
//         y_offset: i32,
//         width: i32,
//         height: i32,
//     ) {
//         let ctx = get_quad_ctx();

//         ctx.texture_update_part(
//             self.raw_miniquad_id(),
//             x_offset,
//             y_offset,
//             width,
//             height,
//             &image.bytes,
//         )
//     }

//     // /// Returns the width of this texture.
//     // pub fn width(&self) -> f32 {
//     //     let ctx = get_quad_ctx();
//     //     let (width, _) = ctx.texture_size(self.raw_miniquad_id());
//     //     width as f32
//     // }

//     // /// Returns the height of this texture.
//     // pub fn height(&self) -> f32 {
//     //     let ctx = get_quad_ctx();
//     //     let (_, height) = ctx.texture_size(self.raw_miniquad_id());
//     //     height as f32
//     // }

//     /// Sets the [FilterMode] of this texture.
//     ///
//     /// Use Nearest if you need integer-ratio scaling for pixel art, for example.
//     ///
//     /// # Example
//     /// ```
//     /// # use macroquad::prelude::*;
//     /// # #[macroquad::main("test")]
//     /// # async fn main() {
//     /// let texture = Texture2D::empty();
//     /// texture.set_filter(FilterMode::Linear);
//     /// # }
//     /// ```
//     pub fn set_filter(&self, filter_mode: FilterMode) {
//         let ctx = get_quad_ctx();

//         ctx.texture_set_filter(self.raw_miniquad_id(), filter_mode);
//     }

impl Texture2D {
    // /// Creates a Texture2D from a miniquad
    // /// [Texture](https://docs.rs/miniquad/0.3.0-alpha/miniquad/graphics/struct.Texture.html)
    // pub fn from_miniquad_texture(texture: miniquad::TextureId) -> Texture2D {
    //     Texture2D {
    //         texture,
    //     }
    // }

    /// Returns the handle for this texture.
    pub fn raw_miniquad_id(&self) -> miniquad::TextureId {
        self.texture
    }
}

//     /// Updates this texture from the screen.
//     pub fn grab_screen(&self) {
//         use miniquad::*;
//         let texture = self.raw_miniquad_id();
//         let ctx = get_quad_ctx();
//         let params = ctx.texture_params(texture);
//         let raw_id = match unsafe { ctx.texture_raw_id(texture) } {
//             miniquad::RawId::OpenGl(id) => id,
//             _ => unimplemented!(),
//         };
//         let internal_format = match params.format {
//             TextureFormat::RGB8 => miniquad::gl::GL_RGB,
//             TextureFormat::RGBA8 => miniquad::gl::GL_RGBA,
//             TextureFormat::Depth => miniquad::gl::GL_DEPTH_COMPONENT,
//             #[cfg(target_arch = "wasm32")]
//             TextureFormat::Alpha => miniquad::gl::GL_ALPHA,
//             #[cfg(not(target_arch = "wasm32"))]
//             TextureFormat::Alpha => miniquad::gl::GL_R8,
//         };
//         unsafe {
//             gl::glBindTexture(gl::GL_TEXTURE_2D, raw_id);
//             gl::glCopyTexImage2D(
//                 gl::GL_TEXTURE_2D,
//                 0,
//                 internal_format,
//                 0,
//                 0,
//                 params.width as _,
//                 params.height as _,
//                 0,
//             );
//         }
//     }

//     /// Returns an [Image] from the pixel data in this texture.
//     ///
//     /// This operation can be expensive.
//     pub fn get_texture_data(&self) -> Image {
//         let ctx = get_quad_ctx();
//         let (width, height) = ctx.texture_size(self.raw_miniquad_id());
//         let mut image = Image {
//             width: width as _,
//             height: height as _,
//             bytes: vec![0; width as usize * height as usize * 4],
//         };
//         ctx.texture_read_pixels(self.raw_miniquad_id(), &mut image.bytes);
//         image
//     }
// }

pub(crate) struct Batcher {
    unbatched: Vec<Texture2D>,
    atlas: crate::text::atlas::Atlas,
}

impl Batcher {
    pub fn new(ctx: &mut dyn miniquad::RenderingBackend) -> Batcher {
        Batcher {
            unbatched: vec![],
            atlas: crate::text::atlas::Atlas::new(ctx, miniquad::FilterMode::Linear),
        }
    }

    // pub fn add_unbatched(&mut self, texture: &Texture2D) {
    //     self.unbatched.push(texture.weak_clone());
    // }

    // pub fn get(&mut self, texture: &Texture2D) -> Option<(Texture2D, Rect)> {
    //     let id = SpriteKey::Texture(texture.raw_miniquad_id());
    //     let uv_rect = self.atlas.get_uv_rect(id)?;
    //     Some((Texture2D::unmanaged(self.atlas.texture()), uv_rect))
    // }
}

/// Build an atlas out of all currently loaded texture
/// Later on all draw_texture calls with texture available in the atlas will use
/// the one from the atlas
/// NOTE: the GPU memory and texture itself in Texture2D will still be allocated
/// and Texture->Image conversions will work with Texture2D content, not the atlas
pub fn build_textures_atlas() {
    // let context = get_context();

    // for texture in context.texture_batcher.unbatched.drain(0..) {
    //     let sprite: Image = texture.get_texture_data();
    //     let id = SpriteKey::Texture(texture.raw_miniquad_id());

    //     context.texture_batcher.atlas.cache_sprite(id, sprite);
    // }

    // let texture = context.texture_batcher.atlas.texture();
    // let (w, h) = get_quad_ctx().texture_size(texture);
    // crate::telemetry::log_string(&format!("Atlas: {} {}", w, h));
    unimplemented!()
}

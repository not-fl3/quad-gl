use miniquad::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(pub [u8; 4]);

pub const LIGHTGRAY: Color = Color([200, 200, 200, 255]);
pub const GRAY: Color = Color([130, 130, 130, 255]);
pub const DARKGRAY: Color = Color([80, 80, 80, 255]);
pub const YELLOW: Color = Color([253, 249, 0, 255]);
pub const GOLD: Color = Color([255, 203, 0, 255]);
pub const ORANGE: Color = Color([255, 161, 0, 255]);
pub const PINK: Color = Color([255, 109, 194, 255]);
pub const RED: Color = Color([230, 41, 55, 255]);
pub const MAROON: Color = Color([190, 33, 55, 255]);
pub const GREEN: Color = Color([0, 228, 48, 255]);
pub const LIME: Color = Color([0, 158, 47, 255]);
pub const DARKGREEN: Color = Color([0, 117, 44, 255]);
pub const SKYBLUE: Color = Color([102, 191, 255, 255]);
pub const BLUE: Color = Color([0, 121, 241, 255]);
pub const DARKBLUE: Color = Color([0, 82, 172, 255]);
pub const PURPLE: Color = Color([200, 122, 255, 255]);
pub const VIOLET: Color = Color([135, 60, 190, 255]);
pub const DARKPURPLE: Color = Color([112, 31, 126, 255]);
pub const BEIGE: Color = Color([211, 176, 131, 255]);
pub const BROWN: Color = Color([127, 106, 79, 255]);
pub const DARKBROWN: Color = Color([76, 63, 47, 255]);
pub const WHITE: Color = Color([255, 255, 255, 255]);
pub const BLACK: Color = Color([0, 0, 0, 255]);
pub const BLANK: Color = Color([0, 0, 0, 0]);
pub const MAGENTA: Color = Color([255, 0, 255, 255]);

const MAX_VERTICES: usize = 10000;
const MAX_INDICES: usize = 5000;

struct DrawCall {
    vertices: [Vertex; MAX_VERTICES],
    indices: [u16; MAX_INDICES],

    vertices_count: usize,
    indices_count: usize,

    clip: Option<(i32, i32, i32, i32)>,
    texture: Texture,

    model: glam::Mat4,
    projection: glam::Mat4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    u: f32,
    v: f32,
    color: Color,
}

impl Vertex {
    pub fn new(x: f32, y: f32, z: f32, u: f32, v: f32, color: Color) -> Vertex {
        Vertex {
            x,
            y,
            z,
            u,
            v,
            color,
        }
    }
}

impl DrawCall {
    fn new(texture: Texture, projection: glam::Mat4, model: glam::Mat4) -> DrawCall {
        DrawCall {
            vertices: [Vertex::new(0., 0., 0., 0., 0., Color([0, 0, 0, 0])); MAX_VERTICES],
            indices: [0; MAX_INDICES],
            vertices_count: 0,
            indices_count: 0,
            clip: None,
            texture,
            projection,
            model,
        }
    }

    fn vertices(&self) -> &[Vertex] {
        &self.vertices[0..self.vertices_count]
    }

    fn indices(&self) -> &[u16] {
        &self.indices[0..self.indices_count]
    }
}

struct GlState {
    texture: Texture,
    clip: Option<(i32, i32, i32, i32)>,
    projection: glam::Mat4,
    model_stack: Vec<glam::Mat4>,
}

impl GlState {
    fn model(&self) -> glam::Mat4 {
        *self.model_stack.last().unwrap()
    }
}

pub struct QuadGl {
    pipeline: Pipeline,

    draw_calls: Vec<DrawCall>,
    draw_calls_bindings: Vec<Bindings>,
    draw_calls_count: usize,
    state: GlState,

    white_texture: Texture,
}

impl QuadGl {
    pub fn new(ctx: &mut miniquad::Context) -> QuadGl {
        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::META);

        let pipeline = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("position", VertexFormat::Float3),
                VertexAttribute::new("texcoord", VertexFormat::Float2),
                VertexAttribute::new("color0", VertexFormat::Byte4),
            ],
            shader,
            PipelineParams {
                color_blend: Some((
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );

        let white_texture = Texture::from_rgba8(ctx, 1, 1, &[255, 255, 255, 255]);

        QuadGl {
            pipeline,
            state: GlState {
                clip: None,
                texture: white_texture,
                projection: glam::Mat4::identity(),
                model_stack: vec![glam::Mat4::identity()],
            },
            draw_calls: Vec::with_capacity(200),
            draw_calls_bindings: Vec::with_capacity(200),
            draw_calls_count: 0,
            white_texture,
        }
    }

    /// Reset internal state to known default
    pub fn reset(&mut self) {
        self.state.clip = None;
        self.state.texture = self.white_texture;
        self.state.projection = glam::Mat4::identity();
        self.state.model_stack = vec![glam::Mat4::identity()];

        self.draw_calls_count = 0;
    }

    pub fn draw(&mut self, ctx: &mut miniquad::Context) {
        for _ in 0..self.draw_calls.len() - self.draw_calls_bindings.len() {
            let vertex_buffer = Buffer::stream(
                ctx,
                BufferType::VertexBuffer,
                MAX_VERTICES * std::mem::size_of::<Vertex>(),
            );
            let index_buffer = Buffer::stream(
                ctx,
                BufferType::IndexBuffer,
                MAX_INDICES * std::mem::size_of::<u16>(),
            );
            let bindings = Bindings {
                vertex_buffers: vec![vertex_buffer],
                index_buffer: index_buffer,
                images: vec![],
            };

            self.draw_calls_bindings.push(bindings);
        }
        assert_eq!(self.draw_calls_bindings.len(), self.draw_calls.len());

        ctx.begin_default_pass(PassAction::Nothing);

        let (width, height) = ctx.screen_size();

        for (dc, bindings) in self.draw_calls[0..self.draw_calls_count]
            .iter_mut()
            .zip(self.draw_calls_bindings.iter_mut())
        {
            bindings.vertex_buffers[0].update(ctx, dc.vertices());
            bindings.index_buffer.update(ctx, dc.indices());
            bindings.images = vec![dc.texture];

            ctx.apply_pipeline(&self.pipeline);
            if let Some(clip) = dc.clip {
                ctx.apply_scissor_rect(clip.0, height as i32 - (clip.1 + clip.3), clip.2, clip.3);
            } else {
                ctx.apply_scissor_rect(0, 0, width as i32, height as i32);
            }
            ctx.apply_bindings(&bindings);
            ctx.apply_uniforms(&shader::Uniforms {
                projection: dc.projection,
                model: dc.model,
            });
            ctx.draw(0, dc.indices_count as i32, 1);

            dc.vertices_count = 0;
            dc.indices_count = 0;
        }

        ctx.end_render_pass();

        self.draw_calls_count = 0;
    }

    pub fn push_model_matrix(&mut self, matrix: glam::Mat4) {
        self.state.model_stack.push(self.state.model() * matrix);
    }

    pub fn pop_model_matrix(&mut self) {
        if self.state.model_stack.len() > 1 {
            self.state.model_stack.pop();
        }
    }

    pub fn geometry(&mut self, vertices: &[Vertex], indices: &[u16]) {
        let previous_dc_ix = if self.draw_calls_count == 0 {
            None
        } else {
            Some(self.draw_calls_count - 1)
        };
        let previous_dc = previous_dc_ix.and_then(|ix| self.draw_calls.get(ix));

        if previous_dc.map_or(true, |draw_call| {
            draw_call.texture != self.state.texture
                || draw_call.clip != self.state.clip
                || draw_call.model != self.state.model()
                || draw_call.projection != self.state.projection
                || draw_call.vertices_count >= MAX_VERTICES - vertices.len()
                || draw_call.indices_count >= MAX_INDICES - indices.len()
        }) {
            if self.draw_calls_count >= self.draw_calls.len() {
                self.draw_calls.push(DrawCall::new(
                    self.state.texture,
                    self.state.projection,
                    self.state.model(),
                ));
            }
            self.draw_calls[self.draw_calls_count].texture = self.state.texture;
            self.draw_calls[self.draw_calls_count].vertices_count = 0;
            self.draw_calls[self.draw_calls_count].indices_count = 0;
            self.draw_calls[self.draw_calls_count].clip = self.state.clip;
            self.draw_calls[self.draw_calls_count].projection = self.state.projection;
            self.draw_calls[self.draw_calls_count].model = self.state.model();

            self.draw_calls_count += 1;
        };
        let dc = &mut self.draw_calls[self.draw_calls_count - 1];

        for i in 0..vertices.len() {
            dc.vertices[dc.vertices_count + i] = vertices[i];
        }

        for i in 0..indices.len() {
            dc.indices[dc.indices_count + i] = indices[i] + dc.vertices_count as u16;
        }
        dc.vertices_count += vertices.len();
        dc.indices_count += indices.len();
        dc.texture = self.state.texture;
    }
}

/// Texture, data stored in GPU memory
#[derive(Clone, Copy, Debug)]
pub struct Texture2D {
    texture: miniquad::Texture,
}

impl Texture2D {
    pub fn empty() -> Texture2D {
        Texture2D {
            texture: miniquad::Texture::empty(),
        }
    }

    pub fn update(&mut self, ctx: &mut miniquad::Context, image: &Image) {
        assert_eq!(self.texture.width, image.width as u32);
        assert_eq!(self.texture.height, image.height as u32);

        self.texture.update(ctx, &image.bytes);
    }

    pub fn width(&self) -> f32 {
        self.texture.width as f32
    }

    pub fn height(&self) -> f32 {
        self.texture.height as f32
    }

    pub fn from_file_with_format<'a>(
        ctx: &mut miniquad::Context,
        bytes: &[u8],
        format: Option<image::ImageFormat>,
    ) -> Texture2D {
        let img = if let Some(fmt) = format {
            image::load_from_memory_with_format(&bytes, fmt)
                .unwrap_or_else(|e| panic!(e))
                .to_rgba()
        } else {
            image::load_from_memory(&bytes)
                .unwrap_or_else(|e| panic!(e))
                .to_rgba()
        };
        let width = img.width() as u16;
        let height = img.height() as u16;
        let bytes = img.into_raw();

        Self::from_rgba8(ctx, width, height, &bytes)
    }

    pub fn from_rgba8(
        ctx: &mut miniquad::Context,
        width: u16,
        height: u16,
        bytes: &[u8],
    ) -> Texture2D {
        let texture = miniquad::Texture::from_rgba8(ctx, width, height, &bytes);

        Texture2D { texture }
    }
}

/// Image, data stored in CPU memory
pub struct Image {
    pub bytes: Vec<u8>,
    pub width: u16,
    pub height: u16,
}

impl Image {
    pub fn empty() -> Image {
        Image {
            width: 0,
            height: 0,
            bytes: vec![],
        }
    }

    pub fn gen_image_color(width: u16, height: u16, color: Color) -> Image {
        let mut bytes = vec![0; width as usize * height as usize * 4];
        for i in 0..width as usize * height as usize {
            for c in 0..4 {
                bytes[i * 4 + c] = color.0[c];
            }
        }
        Image {
            width,
            height,
            bytes,
        }
    }

    pub fn update(&mut self, bytes: &[Color]) {
        assert!(self.width as usize * self.height as usize == bytes.len());

        for i in 0..bytes.len() {
            self.bytes[i * 4] = bytes[i].0[0];
            self.bytes[i * 4 + 1] = bytes[i].0[1];
            self.bytes[i * 4 + 2] = bytes[i].0[2];
            self.bytes[i * 4 + 3] = bytes[i].0[3];
        }
    }
    pub fn width(&self) -> usize {
        self.width as usize
    }

    pub fn height(&self) -> usize {
        self.height as usize
    }

    pub fn get_image_data(&mut self) -> &mut [Color] {
        use std::slice;

        unsafe {
            slice::from_raw_parts_mut(
                self.bytes.as_mut_ptr() as *mut Color,
                self.width as usize * self.height as usize,
            )
        }
    }
}

mod shader {
    use miniquad::{ShaderMeta, UniformBlockLayout, UniformType};

    pub const VERTEX: &str = r#"#version 100
    attribute vec3 position;
    attribute vec2 texcoord;
    attribute vec4 color0;

    varying lowp vec2 uv;
    varying lowp vec4 color;

    uniform mat4 Model;
    uniform mat4 Projection;

    void main() {
        gl_Position = Projection * Model * vec4(position, 1);
        color = color0 / 255.0;
        uv = texcoord;
    }"#;

    pub const FRAGMENT: &str = r#"#version 100
    varying lowp vec4 color;
    varying lowp vec2 uv;
    
    uniform sampler2D Texture;

    void main() {
        gl_FragColor = color * texture2D(Texture, uv) ;
    }"#;

    pub const META: ShaderMeta = ShaderMeta {
        images: &["Texture"],
        uniforms: UniformBlockLayout {
            uniforms: &[
                ("Projection", UniformType::Mat4),
                ("Model", UniformType::Mat4),
            ],
        },
    };

    #[repr(C)]
    #[derive(Debug)]
    pub struct Uniforms {
        pub projection: glam::Mat4,
        pub model: glam::Mat4,
    }
}

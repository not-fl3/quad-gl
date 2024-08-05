use crate::{
    camera::{self, Camera},
    color::Color,
    error::Error,
    image,
    material::Material,
    math::{vec2, vec3, Mat4, Quat, Vec2, Vec3},
    telemetry, text,
    texture::Texture2D,
    tobytes::ToBytes,
    QuadGl,
};

use crate::material::shaders::{preprocess_shader, PreprocessorConfig};

use miniquad::*;

use std::sync::{Arc, Mutex};

pub mod frustum;

#[derive(Clone)]
pub struct NodeData {
    pub vertex_buffers: Vec<miniquad::BufferId>,
    pub index_buffer: miniquad::BufferId,
}

#[derive(Clone, Debug)]
pub struct Uniform {
    name: String,
    uniform_type: UniformType,
    byte_offset: usize,
    byte_size: usize,
}

#[derive(Clone)]
pub struct Shader {
    pub shader: miniquad::ShaderId,
    pub pipeline: miniquad::Pipeline,
    pub uniforms: Vec<Uniform>,
    pub uniforms_data: Vec<u8>,
}

impl Shader {
    pub fn new(
        ctx: &mut miniquad::Context,
        mut uniforms: Vec<(String, UniformType, usize)>,
        fragment: Option<&str>,
        vertex: Option<&str>,
    ) -> Shader {
        let mut max_offset = 0;

        let mut meta = shader::meta().clone();
        for uniform in &uniforms {
            meta.uniforms
                .uniforms
                .push(miniquad::UniformDesc::new(&uniform.0, uniform.1));
        }
        let mut max_offset = 0;
        for miniquad::UniformDesc {
            name,
            uniform_type,
            array_count,
        } in shader::meta().uniforms.uniforms.into_iter().rev()
        {
            uniforms.insert(0, (name.to_owned(), uniform_type, array_count));
        }

        let uniforms = uniforms
            .iter()
            .scan(0, |offset, uniform| {
                let byte_size = uniform.1.size() * uniform.2;
                let uniform = Uniform {
                    name: uniform.0.clone(),
                    uniform_type: uniform.1,
                    byte_offset: *offset,
                    byte_size,
                };
                *offset += byte_size;
                max_offset = *offset;

                Some(uniform)
            })
            .collect();

        let vertex = preprocess_shader(
            &vertex.unwrap_or(shader::VERTEX),
            &PreprocessorConfig {
                includes: vec![(
                    "common_vertex.glsl".to_string(),
                    include_str!("common_vertex.glsl").to_string(),
                )],
            },
        );
        let defines = vec![
            "HAS_METALLIC_ROUGHNESS_MAP".to_string(),
            "HAS_NORMAL_MAP".to_string(),
        ];
        let shader = shadermagic::transform(
            fragment.unwrap_or(shader::FRAGMENT),
            &vertex,
            &meta,
            &shadermagic::Options {
                defines,
                ..Default::default()
            },
        )
        .unwrap();
        let shader = shadermagic::choose_appropriate_shader(&shader, &ctx.info());
        if let miniquad::ShaderSource::Glsl { fragment, vertex } = shader {
            //miniquad::warn!("{}", vertex);
        };
        let shader = ctx
            .new_shader(shader, meta)
            .unwrap_or_else(|e| panic!("Failed to load shader: {}", e));

        let pipeline = ctx.new_pipeline(
            &[
                BufferLayout::default(),
                BufferLayout::default(),
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("in_position", VertexFormat::Float3, 0),
                VertexAttribute::with_buffer("in_uv", VertexFormat::Float2, 1),
                VertexAttribute::with_buffer("in_normal", VertexFormat::Float3, 2),
                VertexAttribute::with_buffer("in_inst", VertexFormat::Float3, 3),
            ],
            shader,
            PipelineParams {
                depth_test: Comparison::LessOrEqual,
                depth_write: true,
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),

                ..Default::default()
            },
        );

        Shader {
            shader,
            pipeline,
            uniforms,
            uniforms_data: vec![0; max_offset],
        }
    }

    pub fn default(ctx: &mut miniquad::Context) -> Shader {
        Self::new(ctx, vec![], None, None)
    }

    /// Set GPU uniform value for this material.
    /// "name" should be from "uniforms" list used for material creation.
    /// Otherwise uniform value would be silently ignored.
    pub fn set_uniform<T: ToBytes>(&mut self, name: &str, uniform: T) {
        let uniform_meta = self.uniforms.iter().find(
            |Uniform {
                 name: uniform_name, ..
             }| uniform_name == name,
        );
        if uniform_meta.is_none() {
            eprintln!("Trying to set non-existing uniform: {}", name);
            return;
        }
        let uniform_meta = uniform_meta.unwrap();
        let uniform_format = uniform_meta.uniform_type;
        let uniform_byte_size = uniform_meta.byte_size;
        let uniform_byte_offset = uniform_meta.byte_offset;

        if uniform_byte_size != uniform_byte_size {
            eprintln!(
                "Trying to set uniform {} sized {} bytes value of {} bytes",
                name,
                std::mem::size_of::<T>(),
                uniform_byte_size
            );
            return;
        }
        let data: &[u8] = uniform.to_bytes().as_ref();
        for i in 0..uniform_byte_size {
            self.uniforms_data[uniform_byte_offset + i] = data[i];
        }
    }
}

#[derive(Clone)]
pub struct Material2 {
    pub color: [f32; 4],
    pub base_color_texture: Option<Texture2D>,
    pub emissive_texture: Option<Texture2D>,
    pub normal_texture: Option<Texture2D>,
    pub occlusion_texture: Option<Texture2D>,
    pub metallic_roughness_texture: Option<Texture2D>,
    pub metallic: f32,
    pub roughness: f32,
    pub shader: Shader,
}
#[derive(Clone)]
pub struct Node {
    pub name: String,
    pub data: Vec<NodeData>,
    pub materials: Vec<Material2>,
    pub transform: Transform,
}

#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Clone)]
pub struct Model {
    pub nodes: Vec<Node>,
    pub aabb: AABB,
}

pub struct Model2 {
    pub model: Model,
    pub transform: Transform,
    pub world_aabb: AABB,
}

#[derive(Debug, Clone)]
pub enum ShadowSplit {
    Orthogonal,
    PSSM2,
    PSSM4,
}

#[derive(Clone)]
pub struct ShadowCaster {
    pub direction: Vec3,
    pub split: ShadowSplit,
}

pub struct Scene {
    pub(crate) quad_ctx: Arc<Mutex<Box<dyn miniquad::RenderingBackend>>>,
    pub(crate) fonts_storage: Arc<Mutex<text::FontsStorage>>,

    pub(crate) cameras: Vec<camera::Camera>,
    pub(crate) models: Vec<Model2>,
    pub(crate) shadow_casters: Vec<ShadowCaster>,

    pub(crate) white_texture: miniquad::TextureId,
    pub(crate) black_texture: miniquad::TextureId,

    pub(crate) shadowmap: crate::shadowmap::ShadowMap,
    //pub(crate) default_material: Material,
}

async fn load_string(path: &str) -> Result<String, Error> {
    unimplemented!()
}

async fn load_file(path: &str) -> Result<Vec<u8>, Error> {
    unimplemented!()
}

#[derive(Clone, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
}
impl Default for Transform {
    fn default() -> Transform {
        Transform {
            translation: vec3(0.0, 0.0, 0.0),
            scale: vec3(1.0, 1.0, 1.0),
            rotation: Quat::IDENTITY,
        }
    }
}
impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Model2 {
    fn update_aabb(&mut self) {
        let aabb = self.model.aabb;
        let min = self.transform.matrix().transform_point3(aabb.min);
        let max = self.transform.matrix().transform_point3(aabb.max);
        self.world_aabb = AABB { min, max };
    }
}
#[derive(Clone)]
pub struct ModelHandle(usize);

impl Scene {
    pub fn aabb(&self, h: &ModelHandle) -> AABB {
        self.models[h.0].world_aabb
    }
    pub fn set_translation(&mut self, h: &ModelHandle, pos: Vec3) {
        self.models[h.0].transform.translation = pos;
        self.models[h.0].update_aabb();
    }
    pub fn set_rotation(&mut self, h: &ModelHandle, rotation: Quat) {
        self.models[h.0].transform.rotation = rotation;
        self.models[h.0].update_aabb();
    }
    pub fn set_scale(&mut self, h: &ModelHandle, scale: Vec3) {
        self.models[h.0].transform.scale = scale;
        self.models[h.0].update_aabb();
    }

    pub fn translation(&self, h: &ModelHandle) -> Vec3 {
        self.models[h.0].transform.translation
    }
    pub fn rotation(&self, h: &ModelHandle) -> Quat {
        self.models[h.0].transform.rotation
    }

    pub fn materials(&mut self, h: &ModelHandle) -> impl Iterator<Item = &mut Material2> {
        self.models[h.0]
            .model
            .nodes
            .iter_mut()
            .map(|n| n.materials.iter_mut())
            .flatten()
    }

    pub fn update_multi_positions(&mut self, h: &ModelHandle, positions: &[Vec3]) {
        let mut model = &mut self.models[h.0];
        let mut ctx = self.quad_ctx.lock().unwrap();
        for mut child in &mut model.model.nodes {
            for mut bindings in &mut child.data {
                let old_vec_size = ctx.buffer_size(bindings.vertex_buffers[3]) as i32 / 12;
                let new_vec_size = positions.len();
                if old_vec_size != new_vec_size as i32 {
                    bindings.vertex_buffers[3] =
                        ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Stream, unsafe {
                            BufferSource::slice(positions)
                        });
                } else {
                    ctx.buffer_update(bindings.vertex_buffers[3], BufferSource::slice(positions));
                }
            }
        }
    }

    pub fn update_child(&mut self, h: &ModelHandle, name: &str, f: impl Fn(&mut Transform)) {
        let model = &mut self.models[h.0];
        for child in &mut model.model.nodes {
            if child.name == name {
                f(&mut child.transform)
            }
        }
    }
}

impl Scene {
    pub(crate) fn new(
        ctx: Arc<Mutex<Box<dyn miniquad::RenderingBackend>>>,
        fonts_storage: Arc<Mutex<text::FontsStorage>>,
    ) -> Scene {
        let quad_ctx = ctx.clone();
        let mut ctx = ctx.lock().unwrap();
        // let shader = ctx
        //     .new_shader(
        //         ShaderSource::Glsl {
        //             vertex: shader::VERTEX,
        //             fragment: shader::FRAGMENT,
        //         },
        //         shader::meta(),
        //     )
        //     .unwrap_or_else(|e| panic!("Failed to load shader: {}", e));

        // let default_material = Material::new2(
        //     &mut **ctx,
        //     shader,
        //     PipelineParams {
        //         depth_test: Comparison::LessOrEqual,
        //         depth_write: true,
        //         ..Default::default()
        //     },
        //     vec![],
        //     vec![],
        // )
        // .unwrap();

        Scene {
            white_texture: ctx.new_texture_from_rgba8(1, 1, &[255, 255, 255, 255]),
            black_texture: ctx.new_texture_from_rgba8(1, 1, &[0, 0, 0, 0]),
            fonts_storage: fonts_storage.clone(),

            cameras: vec![],
            models: vec![],
            shadow_casters: vec![],

            shadowmap: crate::shadowmap::ShadowMap::new(ctx.as_mut()),
            //default_material,
            quad_ctx,
        }
    }
}

impl Scene {
    // pub fn add_camera(&mut self, camera: camera::Camera) -> CameraHandle {
    //     self.cameras.push(camera);
    //     CameraHandle(self.cameras.len() - 1)
    // }

    pub fn add_shadow_caster(&mut self, shadow_caster: ShadowCaster) {
        self.shadow_casters.push(shadow_caster);
    }

    pub fn add_model(&mut self, model: &Model) -> ModelHandle {
        self.models.push(Model2 {
            model: model.clone(),
            transform: Transform {
                translation: vec3(0.0, 0.0, 0.0),
                scale: vec3(1., 1., 1.),
                rotation: Quat::IDENTITY,
            },
            world_aabb: model.aabb,
        });
        ModelHandle(self.models.len() - 1)
    }

    // pub fn add_multi_model(&mut self, model: &Model, multi_position: Vec<Vec3>) -> ModelHandle {
    //     self.models.push(Model2 {
    //         model: model.clone(),
    //         transform: Transform {
    //             translation: vec3(0.0, 0.0, 0.0),
    //             scale: vec3(1., 1., 1.),
    //             rotation: Quat::IDENTITY,
    //         },
    //         world_aabb: model.aabb,
    //         multi_position: Some(multi_position),
    //     });
    //     ModelHandle(self.models.len() - 1)
    // }

    // pub fn fullscreen_canvas(&self, ix: usize) -> sprite_layer::SpriteLayer {
    //     // fn pixel_perfect_render_state() -> RenderState {
    //     //     let (w, h) = (
    //     //         crate::window::screen_width(),
    //     //         crate::window::screen_height(),
    //     //     );
    //     //     RenderState {
    //     //         camera: crate::camera::Camera::Camera2D {
    //     //             rotation: 0.,
    //     //             zoom: vec2(1. / w * 2., -1. / h * 2.),
    //     //             target: vec2(w / 2., h / 2.),
    //     //             offset: vec2(0., 0.),
    //     //         },
    //     //         ..Default::default()
    //     //     }
    //     // }

    //     //let render_state = pixel_perfect_render_state();
    //     // self.data.layers.lock()[ix].render_pass(None);
    //     // self.data.layers.lock()[ix].clear_draw_calls();

    //     //SpriteLayer::new(self.ctx.clone(), ix)
    //     unimplemented!()
    // }

    // pub fn canvas(&self, render_state: RenderState) -> SpriteLayer {
    //     let mut gl = self.layers.lock()..pop().unwrap();
    //     let render_pass = render_state.render_target.as_ref().map(|rt| rt.render_pass);
    //     gl.render_pass(render_pass);

    //     SpriteLayer::new(self, gl, render_state)
    // }

    pub fn clear(&self, color: Color) {
        let mut ctx = self.quad_ctx.lock().unwrap();
        let clear = PassAction::clear_color(color.r, color.g, color.b, color.a);

        ctx.begin_default_pass(clear);
        ctx.end_render_pass();
    }

    // pub fn clear2(&mut self, ctx: &Context2, color: Color) {
    //     let mut ctx = self.quad_ctx.lock().unwrap();
    //     let clear = PassAction::clear_color(color.r, color.g, color.b, color.a);

    //     if let Some(pass) = render_state.render_target.as_ref().map(|rt| rt.render_pass) {
    //         ctx.begin_pass(Some(pass), clear);
    //     } else {
    //         ctx.begin_default_pass(clear);
    //     }
    //     ctx.end_render_pass();
    // }

    pub(crate) fn draw_canvas(&self, ix: usize) {
        // let mut ctx = self.data.quad_ctx.lock().unwrap();

        // let (width, height) = miniquad::window::screen_size();

        // let screen_mat = glam::Mat4::orthographic_rh_gl(0., width, height, 0., -1., 1.);
        // let canvas = &mut self.data.layers.lock()[ix];
        // canvas.draw(&mut **ctx, screen_mat);

        unimplemented!()
    }

    pub(crate) fn draw_model(
        ctx: &mut miniquad::Context,
        white_texture: TextureId,
        black_texture: TextureId,
        model: &mut Model2,
        camera: &camera::Camera,
        shadow_proj: [Mat4; 4],
        shadow_cascades: [f32; 4],
        shadowmap: [TextureId; 4],
        shadow_casters: [i32; 4],
        clipping_planes: [frustum::Plane; 6],
    ) {
        // unsafe {
        //     miniquad::gl::glPolygonMode(miniquad::gl::GL_FRONT_AND_BACK, miniquad::gl::GL_LINE);
        // }

        let transform = model.transform.matrix();
        let aabb = model.world_aabb;
        let m = &model;
        let model = &mut model.model;
        if clipping_planes.iter().any(|p| !p.clip(aabb)) {
            return;
        }
        for node in &mut model.nodes {
            for (bindings, material) in node.data.iter_mut().zip(node.materials.iter_mut()) {
                let cubemap = match camera.environment {
                    crate::camera::Environment::Skybox(ref cubemap) => Some(cubemap.texture),
                    _ => None,
                };
                let or_white = |t: &Option<Texture2D>| {
                    t.as_ref().map_or(white_texture, |t| t.raw_miniquad_id())
                };
                let or_black = |t: &Option<Texture2D>| {
                    t.as_ref().map_or(black_texture, |t| t.raw_miniquad_id())
                };
                let images = [
                    or_white(&material.base_color_texture),
                    or_black(&material.emissive_texture),
                    or_white(&material.occlusion_texture),
                    or_white(&material.normal_texture),
                    or_white(&material.metallic_roughness_texture),
                    cubemap.unwrap_or(white_texture),
                    shadowmap[0],
                    shadowmap[1],
                    shadowmap[2],
                    shadowmap[3],
                ];
                ctx.apply_pipeline(&material.shader.pipeline);
                assert_eq!(bindings.vertex_buffers.len(), 4);
                ctx.apply_bindings_from_slice(
                    &bindings.vertex_buffers,
                    bindings.index_buffer,
                    &images,
                );

                let (proj, view) = camera.proj_view();

                let projection = proj * view;
                let time = (miniquad::date::now()) as f32;
                let time = glam::vec4(time, time.sin(), time.cos(), 0.);

                let model_matrix = transform * node.transform.matrix();
                let model_matrix_inverse = model_matrix.inverse();
                // ctx.apply_uniforms(UniformsSource::table(&shader::Uniforms {
                //     projection,
                //     shadow_projection: shadow_proj,
                //     model: model_matrix,
                //     model_inverse: model_matrix_inverse,
                //     color: material.color,
                //     shadow_cascades,
                //     shadow_casters,
                //     material: [material.metallic, material.roughness, 0.0, 0.0],
                //     camera_pos: camera.position,
                // }));
                material.shader.set_uniform("Projection", projection);
                // TODO: implement the array thing
                material
                    .shader
                    .set_uniform("ShadowProjection", &shadow_proj[..]);
                material.shader.set_uniform("Model", model_matrix);
                material
                    .shader
                    .set_uniform("ModelInverse", model_matrix_inverse);
                material.shader.set_uniform("Color", material.color);
                material
                    .shader
                    .set_uniform("ShadowCascades", shadow_cascades);
                material.shader.set_uniform("ShadowCasters", shadow_casters);
                material.shader.set_uniform(
                    "Material",
                    [material.metallic, material.roughness, 0.0, 0.0],
                );
                material
                    .shader
                    .set_uniform("CameraPosition", camera.position);
                ctx.apply_uniforms_from_bytes(
                    material.shader.uniforms_data.as_ptr(),
                    material.shader.uniforms_data.len(),
                );
                let buffer_size = ctx.buffer_size(bindings.index_buffer) as i32 / 2;
                let multi_size = ctx.buffer_size(bindings.vertex_buffers[3]) as i32 / 12;
                ctx.draw(0, buffer_size, multi_size);
            }
        }

        // unsafe {
        //     use miniquad::gl;
        //     gl::glPolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_FILL);
        // }
    }

    // pub fn set_transform(&self, model: usize, transform: Mat4) {
    //     self.models[model].1 = transform;
    // }

    pub fn draw(&mut self, camera: &Camera) {
        let _z = telemetry::ZoneGuard::new("Scene::draw");

        let clipping_planes = frustum::projection_planes(camera);
        let (proj, view) = camera.proj_view();
        let mut clear_action = PassAction::Nothing;
        {
            let _z = telemetry::ZoneGuard::new("environment");

            if let crate::camera::Environment::Skybox(ref cubemap) = camera.environment {
                cubemap.draw(&mut **self.quad_ctx.lock().unwrap(), &proj, &view);
            }

            if let crate::camera::Environment::SolidColor(color) = camera.environment {
                clear_action = PassAction::clear_color(color.r, color.g, color.b, color.a);
            }

            unsafe {
                miniquad::gl::glFlush();
                miniquad::gl::glFinish();
            }
        }
        let mut ctx = self.quad_ctx.lock().unwrap();

        let mut shadow_proj = Default::default();
        let mut cascade_clips = Default::default();
        let casters_count = self.shadow_casters.len();
        let mut split_count = 0;
        if let Some(shadow_caster) = self.shadow_casters.get(0) {
            split_count = match shadow_caster.split {
                ShadowSplit::Orthogonal => 1,
                ShadowSplit::PSSM2 => 2,
                ShadowSplit::PSSM4 => 4,
            };
            let _z = telemetry::ZoneGuard::new("shadows");
            (shadow_proj, cascade_clips) = self.shadowmap.draw_shadow_pass(
                ctx.as_mut(),
                &self.models[..],
                &camera,
                shadow_caster,
                clipping_planes,
            );

            unsafe {
                miniquad::gl::glFlush();
                miniquad::gl::glFinish();
            }
        }

        if let Some(pass) = camera.render_target.as_ref().map(|rt| rt.render_pass) {
            ctx.begin_pass(Some(pass), clear_action);
        } else {
            ctx.begin_default_pass(clear_action);
        }

        {
            let _z = telemetry::ZoneGuard::new("models");
            for model in &mut self.models {
                Scene::draw_model(
                    ctx.as_mut(),
                    self.white_texture,
                    self.black_texture,
                    model,
                    camera,
                    shadow_proj,
                    cascade_clips,
                    [
                        self.shadowmap.depth_img[0],
                        self.shadowmap.depth_img[1],
                        self.shadowmap.depth_img[2],
                        self.shadowmap.depth_img[3],
                    ],
                    [casters_count as _, split_count as _, 0, 0],
                    clipping_planes,
                );
            }
            unsafe {
                miniquad::gl::glFlush();
                miniquad::gl::glFinish();
            }
        }
        ctx.end_render_pass();
    }

    pub fn draw_shadow_debug(&mut self) {
        let mut ctx = self.quad_ctx.lock().unwrap();

        self.shadowmap
            .dbg
            .draw(ctx.as_mut(), &self.shadowmap.depth_img[..]);
    }
}

pub mod shader {
    use crate::math::Vec3;
    use miniquad::{ShaderMeta, UniformBlockLayout, UniformDesc, UniformType};

    pub const VERTEX: &str = include_str!("vertex.glsl");
    pub const FRAGMENT: &str = include_str!("fragment.glsl");
    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec![
                "Albedo".to_string(),
                "Emissive".to_string(),
                "Occlusion".to_string(),
                "Normal".to_string(),
                "MetallicRoughness".to_string(),
                "Environment".to_string(),
                "ShadowMap0".to_string(),
                "ShadowMap1".to_string(),
                "ShadowMap2".to_string(),
                "ShadowMap3".to_string(),
            ],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("Projection", UniformType::Mat4),
                    UniformDesc::array(UniformDesc::new("ShadowProjection", UniformType::Mat4), 4),
                    UniformDesc::new("Model", UniformType::Mat4),
                    UniformDesc::new("ModelInverse", UniformType::Mat4),
                    UniformDesc::new("Color", UniformType::Float4),
                    UniformDesc::new("ShadowCascades", UniformType::Float4),
                    UniformDesc::new("ShadowCasters", UniformType::Int4),
                    UniformDesc::new("Material", UniformType::Float4),
                    UniformDesc::new("CameraPosition", UniformType::Float3),
                ],
            },
        }
    }

    // #[repr(C)]
    // pub struct Uniforms {
    //     pub projection: glam::Mat4,
    //     pub shadow_projection: [glam::Mat4; 4],
    //     pub model: glam::Mat4,
    //     pub model_inverse: glam::Mat4,
    //     pub color: [f32; 4],
    //     pub shadow_cascades: [f32; 4],
    //     pub shadow_casters: [i32; 4], // count, split, 0, 0
    //     pub material: [f32; 4],       // metallic, roughness, 0, 0,
    //     pub camera_pos: glam::Vec3,
    // }
}

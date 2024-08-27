// //! 3D shapes and models, loading 3d models from files, drawing 3D primitives.

use crate::{
    math::{vec2, vec3, Vec2, Vec3},
    scene::{self, Model, Node, NodeData, Scene, Transform},
    texture::Texture2D,
};
use miniquad::*;

use std::sync::Arc;

pub struct CpuMesh {
    pub vertices: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<u16>,
}

pub fn sphere(radius: f32, rings: u32, slices: u32) -> CpuMesh {
    let scale = vec3(radius, radius, radius);
    let mut vertices = vec![];
    let mut indices = vec![];
    let mut uvs = vec![];
    let mut normals = vec![];

    // (i, j + 1)   (i + 1, j + 1)
    //
    // (i, j)       (i + 1, j)
    //
    // i      j
    // i + 1, j + 1
    // i + 1, j
    //
    // i,     j
    // i    , j + 1
    // i + 1, j + 1

    let mut geometry = |v: &[_], i| {
        for ix in i {
            indices.push((vertices.len() + ix) as u16);
        }
        for &(v, uv, n) in v {
            vertices.push(v);
            uvs.push(uv);
            normals.push(n);
        }
    };
    for i in 0..rings + 1 {
        for j in 0..slices + 1 {
            use std::f32::consts::PI;

            let pi34 = PI / 2. * 3.;
            let pi2 = PI * 2.;
            let i = i as f32;
            let j = j as f32;
            let rings: f32 = rings as _;
            let slices: f32 = slices as _;

            let v1 = vec3(
                (pi34 + (PI / (rings + 1.)) * i).cos() * (j * pi2 / slices).sin(),
                (pi34 + (PI / (rings + 1.)) * i).sin(),
                (pi34 + (PI / (rings + 1.)) * i).cos() * (j * pi2 / slices).cos(),
            );
            let uv1 = vec2(i / rings, j / slices);
            let v2 = vec3(
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).cos() * ((j + 1.) * pi2 / slices).sin(),
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).sin(),
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).cos() * ((j + 1.) * pi2 / slices).cos(),
            );
            let uv2 = vec2((i + 1.) / rings, (j + 1.) / slices);
            let v3 = vec3(
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).cos() * (j * pi2 / slices).sin(),
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).sin(),
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).cos() * (j * pi2 / slices).cos(),
            );
            let uv3 = vec2((i + 1.) / rings, j / slices);

            geometry(
                &[
                    ((v1 * scale), uv1, v1.normalize()),
                    ((v2 * scale), uv2, v2.normalize()),
                    ((v3 * scale), uv3, v3.normalize()),
                ],
                &[0, 1, 2],
            );

            let v1 = vec3(
                (pi34 + (PI / (rings + 1.)) * i).cos() * (j * pi2 / slices).sin(),
                (pi34 + (PI / (rings + 1.)) * i).sin(),
                (pi34 + (PI / (rings + 1.)) * i).cos() * (j * pi2 / slices).cos(),
            );
            let uv1 = vec2(i / rings, j / slices);
            let v2 = vec3(
                (pi34 + (PI / (rings + 1.)) * (i)).cos() * ((j + 1.) * pi2 / slices).sin(),
                (pi34 + (PI / (rings + 1.)) * (i)).sin(),
                (pi34 + (PI / (rings + 1.)) * (i)).cos() * ((j + 1.) * pi2 / slices).cos(),
            );
            let uv2 = vec2(i / rings, (j + 1.) / slices);
            let v3 = vec3(
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).cos() * ((j + 1.) * pi2 / slices).sin(),
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).sin(),
                (pi34 + (PI / (rings + 1.)) * (i + 1.)).cos() * ((j + 1.) * pi2 / slices).cos(),
            );
            let uv3 = vec2((i + 1.) / rings, (j + 1.) / slices);

            geometry(
                &[
                    ((v1 * scale), uv1, v1.normalize()),
                    ((v2 * scale), uv2, v2.normalize()),
                    ((v3 * scale), uv3, v3.normalize()),
                ],
                &[0, 1, 2],
            );
        }
    }

    CpuMesh {
        vertices,
        uvs,
        normals,
        indices,
    }
}

pub fn square() -> CpuMesh {
    let width = 1.0;
    let length = 1.0;
    let indices = vec![0u16, 1, 2, 0, 2, 3];

    let vertices = vec![
        vec3(-width / 2., 0., -length / 2.),
        vec3(-width / 2., 0., length / 2.),
        vec3(width / 2., 0., length / 2.),
        vec3(width / 2., 0., -length / 2.),
    ];
    let uvs = vec![vec2(0., 1.), vec2(0., 0.), vec2(1., 0.), vec2(1., 1.)];
    let normals = vec![
        vec3(0., 1., 0.),
        vec3(0., 1., 0.),
        vec3(0., 1., 0.),
        vec3(0., 1., 0.),
    ];

    CpuMesh {
        vertices,
        uvs,
        normals,
        indices,
    }
}

pub(crate) fn mesh(
    quad_ctx: &mut miniquad::Context,
    CpuMesh {
        vertices,
        uvs,
        normals,
        indices,
    }: CpuMesh,
    texture: Option<Arc<Texture2D>>,
) -> Model {
    let vertex_buffer = quad_ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&vertices),
    );
    let normals_buffer = quad_ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&normals),
    );
    let uvs_buffer = quad_ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&uvs),
    );
    let index_buffer = quad_ctx.new_buffer(
        BufferType::IndexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&indices),
    );
    let shader = shadermagic::transform(
        crate::scene::shader::FRAGMENT,
        crate::scene::shader::VERTEX,
        &crate::scene::shader::meta(),
        &shadermagic::Options {
            defines: vec![],
            ..Default::default()
        },
    )
    .unwrap();
    let shader = shadermagic::choose_appropriate_shader(&shader, &quad_ctx.info());
    if let miniquad::ShaderSource::Glsl { fragment, vertex } = shader {
        //miniquad::warn!("{}", fragment);
    };
    let shader = quad_ctx
        .new_shader(shader, scene::shader::meta())
        .unwrap_or_else(|e| panic!("Failed to load shader: {}", e));

    let pipeline = quad_ctx.new_pipeline(
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

    let instancing = vec![vec3(0.0, 0.0, 0.0)];
    let instancing_buffer =
        quad_ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, unsafe {
            BufferSource::slice(&instancing[..])
        });

    let data = NodeData {
        vertex_buffers: vec![vertex_buffer, uvs_buffer, normals_buffer, instancing_buffer],
        index_buffer,
    };
    let material = scene::Material2 {
        color: [1.0, 1.0, 1.0, 1.0],
        base_color_texture: texture,
        emissive_texture: None,
        normal_texture: None,
        occlusion_texture: None,
        metallic_roughness_texture: None,
        metallic: 0.01,
        roughness: 0.8,
        shader: scene::Shader::default(quad_ctx),
    };

    let mut aabb = crate::scene::AABB {
        min: vec3(std::f32::MAX, std::f32::MAX, std::f32::MAX),
        max: vec3(-std::f32::MAX, -std::f32::MAX, -std::f32::MAX),
    };
    for vertex in &vertices {
        aabb.min = aabb.min.min(*vertex);
        aabb.max = aabb.max.max(*vertex);
    }
    Model {
        nodes: vec![Node {
            name: "root".to_string(),
            data: vec![data],
            materials: vec![material],
            transform: Transform::default(),
        }],
        aabb,
    }
}

impl crate::QuadGl {
    pub fn mesh(
        &self,
        m: CpuMesh,
        texture: Option<Arc<Texture2D>>,
    ) -> Model {
        mesh(self.quad_ctx.lock().unwrap().as_mut(), m, texture)
    }
}

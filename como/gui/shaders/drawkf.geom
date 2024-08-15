/*
* This file was derived from part of DeepFactors by Jan Czarnowski.
*
* Copyright (C) 2020 Imperial College London
* 
* The use of the code within this file and all code within files that make up
* the software that is DeepFactors is permitted for non-commercial purposes
* only.  The full terms and conditions that apply to the code within this file
* are detailed within the LICENSE file and at
* <https://www.imperial.ac.uk/dyson-robotics-lab/projects/deepfactors/deepfactors-license>
* unless explicitly stated. By downloading this file you agree to comply with
* these terms.
*
* If you wish to use any of this code for commercial purposes then please
* email researchcontracts.engineering@imperial.ac.uk.
*
*/

#version 430 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

// Uniforms and textures
uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform vec4 cam; // fx, fy, ux, uy
uniform int width;
uniform int height;
uniform float slt_thresh = 0.075;
uniform int crop_pix = 30;

uniform sampler2D image;
uniform sampler2D depth;
uniform sampler2D valid;

// Output to fragment shader
out fData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
}
vertex;

vec3 triangle_normal(vec4 a, vec4 b, vec4 c)
{
  vec3 A = c.xyz - a.xyz;
  vec3 B = b.xyz - a.xyz;
  return normalize(cross(A, B));
}

vec4 unproject(vec4 cam, vec2 point, float d)
{
  float fx = cam[0];
  float fy = cam[1];
  float cx = cam[2];
  float cy = cam[3];
  vec3 ray = vec3((point.x - cx) / fx, (point.y - cy) / fy, 1);
  return vec4(ray * d, 1);
}

vec4 lift(vec2 point, float depth, vec4 cam)
{
  return unproject(cam, point, depth);
}

struct PixelData
{
  vec3 color;
  float depth;
  bool valid;
};

PixelData fetch_pixel(ivec2 loc)
{
  PixelData data;
  data.color = texelFetch(image, loc, 0).xyz;
  data.depth = texelFetch(depth, loc, 0).x;
  data.valid = texelFetch(valid, loc, 0).x >= 0;
  return data;
}

void main(void)
{
  mat4 m_view = m_camera * m_model;
  mat4 mvp = m_proj * m_view;

  // get (x,y) pixel location from primitive id
  int y = gl_PrimitiveIDIn / int(width);
  int x = gl_PrimitiveIDIn - y * int(width);

  if (x < crop_pix || x > int(width) - crop_pix ||
      y < crop_pix || y > int(height) - crop_pix)
    return;

  // texelFetch uses bottom-left as origin, but we've uploaded
  // the image flipped, so we can just indexing with top-left as origin

  ivec2 topleft = ivec2(x, y);
  ivec2 topright = ivec2(x + 1, y);
  ivec2 botleft = ivec2(x, y + 1);
  ivec2 botright = ivec2(x + 1, y + 1);

  PixelData topleft_data = fetch_pixel(topleft);
  PixelData topright_data = fetch_pixel(topright);
  PixelData botleft_data = fetch_pixel(botleft);
  PixelData botright_data = fetch_pixel(botright);

  // need to lift 4 points around and generate triangles
  // NOTE: this is in the camera frame
  vec4 topleft_pt = lift(vec2(topleft), topleft_data.depth, cam);
  vec4 topright_pt = lift(vec2(topright), topright_data.depth, cam);
  vec4 botleft_pt = lift(vec2(botleft), botleft_data.depth, cam);
  vec4 botright_pt = lift(vec2(botright), botright_data.depth, cam);

  // calculate normals in the camera frame
  vec3 n1 = triangle_normal(topleft_pt, botleft_pt, topright_pt);
  vec3 n2 = triangle_normal(topright_pt, botleft_pt, botright_pt);
  vec3 ray = normalize(vec3((x - cam[2]) / cam[0], (y - cam[3]) / cam[1], 1));
  if (abs(dot(n1, ray)) < slt_thresh) // invalidate too slanted triangles
    return;
  if (abs(dot(n2, ray)) < slt_thresh) // invalidate too slanted triangles
    return;


  // average two triangle normals for the quad
  n1 = triangle_normal(m_view*topleft_pt, m_view*botleft_pt, m_view*topright_pt);
  n2 = triangle_normal(m_view*topright_pt, m_view*botleft_pt, m_view*botright_pt);
  vec3 normal = (n1 + n2) / 2.0;
  // normal is computed in the opencv coordinate frame, switch to opengl by flipping y
  normal[1] *= -1;

  // transform the points into opengl camera frame
  topright_pt = mvp * topright_pt;
  topleft_pt = mvp * topleft_pt;
  botright_pt = mvp * botright_pt;
  botleft_pt = mvp * botleft_pt;


  if (!topleft_data.valid || !botright_data.valid || !topright_data.valid || !botleft_data.valid) {
    return;
  }
  // CCW
  gl_Position = topleft_pt;
  vertex.pos = topleft_pt.xyz;
  vertex.color = topleft_data.color;
  vertex.normal = normal;
  EmitVertex();

  gl_Position = botleft_pt;
  vertex.pos = botleft_pt.xyz;
  vertex.color = botleft_data.color;
  vertex.normal = normal;
  EmitVertex();

  gl_Position = topright_pt;
  vertex.pos = topright_pt.xyz;
  vertex.color = topright_data.color;
  vertex.normal = normal;
  EmitVertex();

  gl_Position = botright_pt;
  vertex.pos = botright_pt.xyz;
  vertex.color = botright_data.color;
  vertex.normal = normal;

  EmitVertex();

  EndPrimitive();

}

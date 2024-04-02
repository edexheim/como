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
uniform mat4  mvp;
uniform vec4  cam; // fx, fy, ux, uy
uniform int   width;
uniform int   height;
uniform float slt_thresh;
uniform int   crop_pix;

uniform sampler2D image;
uniform sampler2D depth;
uniform sampler2D valid;

// Output to fragment shader
out fData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
} vertex;

vec3 triangle_normal(vec4 a, vec4 b, vec4 c)
{
  vec3 A = c.xyz - a.xyz;
  vec3 B = b.xyz - a.xyz;
  return normalize(cross(A,B));
}

vec4 lift(vec2 point, float depth, vec4 cam)
{
  // transform to normalized image plane
  vec3 ray;
  ray.x = (point.x - cam[2]) / cam[0];
  ray.y = (point.y - cam[3]) / cam[1];
  ray.z = 1.0;

  // reproject
  return vec4(ray * depth, 1);
}

struct PixelData
{
  vec2 pos;
  vec3 color;
  float depth;
  bool valid;
};

PixelData validate_pixel(PixelData data)
{
  // if (data.depth < 0.5 || data.depth > 10) // if depth is too small or too big
  //   data.valid = false;

  int border = 4;
  if (data.pos.x < border || data.pos.x > width-border ||   // cut the border
      data.pos.y < border || data.pos.y > height-border)
  {
    data.valid = false;
  }

  return data;
}

PixelData fetch_pixel(ivec2 loc)
{
  struct PixelData data;
  data.pos = loc;
  data.color = texelFetch(image, loc, 0).xyz;
  data.depth = texelFetch(depth, loc, 0).x;
  data.valid = texelFetch(valid, loc, 0).x > 0;
  return validate_pixel(data);
}

void main(void)
{
  // get (x,y) pixel location from primitive id
  int y = gl_PrimitiveIDIn / int(width);
  int x = gl_PrimitiveIDIn - y * int(width);

  if (x < crop_pix || x > int(width) - crop_pix ||
      y < crop_pix || y > int(height) - crop_pix)
    return;

  ivec2 topright = ivec2(x,   y);
  ivec2 topleft  = ivec2(x-1, y);
  ivec2 botright = ivec2(x,   y+1);
  ivec2 botleft  = ivec2(x-1, y+1);

  PixelData topright_data = fetch_pixel(topright);
  PixelData topleft_data  = fetch_pixel(topleft);
  PixelData botright_data = fetch_pixel(botright);
  PixelData botleft_data  = fetch_pixel(botleft);

  // need to lift 4 points around and generate triangles
  // NOTE: this is in the camera frame
  vec4 topright_pt = lift(vec2(topright), topright_data.depth, cam);
  vec4 topleft_pt  = lift(vec2(topleft), topleft_data.depth, cam);
  vec4 botright_pt = lift(vec2(botright), botright_data.depth, cam);
  vec4 botleft_pt  = lift(vec2(botleft), botleft_data.depth, cam);

  // calculate normals in the camera frame
  vec3 n1 = triangle_normal(topright_pt, topleft_pt, botright_pt);
  vec3 n2 = triangle_normal(topleft_pt, botleft_pt, botright_pt);

  vec3 ray = normalize(vec3((x - cam[2])/cam[0],(y-cam[3])/cam[1],1));

  float angle1 = abs(dot(n1, ray));
  float angle2 = abs(dot(n2, ray));

  bool topright_triangle_valid = true;
  bool bottomleft_triangle_valid = true;
  if (angle1 < slt_thresh) // invalidate too slanted triangles
    topright_triangle_valid = false;
  if (angle2 < slt_thresh) // invalidate too slanted triangles
    bottomleft_triangle_valid = false;

  if (!topright_triangle_valid && !bottomleft_triangle_valid)
    return;

  // transform the points into opengl camera frame
  topright_pt = mvp * topright_pt;
  topleft_pt  = mvp * topleft_pt;
  botright_pt = mvp * botright_pt;
  botleft_pt  = mvp * botleft_pt;

  vec3 normal;
  if (topright_triangle_valid && bottomleft_triangle_valid)
    normal = (n1 + n2) / 2.0;
  else if (topright_triangle_valid)
    normal = n1;
  else if (bottomleft_triangle_valid)
    normal = n2;
  //normal = (mvp * vec4(normal, 1.0f)).xyz;  // doesnt work

  // Emit the vertices
  //
  //  topleft      topright
  //          +--+
  //          |\ |
  //          | \|
  //          +--+
  //  botleft      botright
  //
  // Algorithm:
  // if topright is not valid then just dont emit it
  // if topleft or botright is not valid then do not emit anything
  // complete the second triangle only when botleft is valid
  if (topright_data.valid && topright_triangle_valid)
  {
    gl_Position = topright_pt;
    vertex.pos    = topright_pt.xyz;
    vertex.color  = topright_data.color;
    vertex.normal = normal;
    EmitVertex();
  }

  if (!topleft_data.valid || !botright_data.valid)
    return;

  gl_Position = topleft_pt;
  vertex.pos    = topleft_pt.xyz;
  vertex.color  = topleft_data.color;
  vertex.normal = normal;
  EmitVertex();

  gl_Position = botright_pt;
  vertex.pos    = botright_pt.xyz;
  vertex.color  = botright_data.color;
  vertex.normal = normal;
  EmitVertex();

  if (botleft_data.valid && bottomleft_triangle_valid)
  {
    gl_Position = botleft_pt;
    vertex.pos    = botleft_pt.xyz;
    vertex.color  = botleft_data.color;
    vertex.normal = normal;
    EmitVertex();
  }

  EndPrimitive();
}

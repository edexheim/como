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

#version 330 core

uniform vec3 lightpos;
uniform bool phong_enabled;
uniform bool normals_render;
uniform mat4 mvp;

in fData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
} frag;

out vec4 out_color;

vec3 normal2rgb(vec3 normal) {
  vec3 rgb = 0.5*(1.0 + normal);
  return rgb;
}

// TODO: paramatrize some stuff inside here
vec3 phong_shading(vec3 incolor, vec3 normal)
{
  vec3 lightpos_cam = (mvp * vec4(lightpos, 1.0f)).xyz;

  // weights
  float ka = 0.9; // ambient
  float kd = 0.0; // diffuse
  float ks = 0.3; // specular
  float shininess = 32.0;

  // colors
  vec3 ambient_color = incolor;
  vec3 diffuse_color = incolor;
  vec3 specular_color = vec3(1.0, 1.0, 1.0);

  // ambient term
  vec3 ambient = ka * ambient_color;

  // diffuse term
  vec3 lightDir = normalize(lightpos_cam - frag.pos);
  float NdotL = dot(normal, lightDir);
  float lambertian = max(NdotL, 0.0);
  vec3 diffuse = kd * lambertian * diffuse_color;

  // specular term
  vec3 rVector = normalize(2.0 * normal * dot(normal, lightDir) - lightDir);
  vec3 viewVector = normalize(-frag.pos);
  float RdotV = dot(rVector, viewVector);
  float spec_angle = max(RdotV, 0.0);
  vec3 specular = ks * pow(spec_angle, shininess) * specular_color;

  return ambient + diffuse + specular;
}

void main()
{
  if (phong_enabled)
  {
    out_color = vec4(phong_shading(frag.color, frag.normal), 1.);
  }
  else if (normals_render) 
  {
    out_color = vec4(normal2rgb(frag.normal), 1.);
  }
  else
  {
    out_color = vec4(frag.color, 1.);
  }
}

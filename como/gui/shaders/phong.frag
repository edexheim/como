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
uniform mat4 m_model;
uniform mat4 m_camera;
// opengl gl left-hand, x-right, y-up, z-fwd
uniform vec3 lightpos = vec3(0, 0.3, -1);
uniform vec3 phong = vec3(0.5, 0.4, 3);
uniform float spec = 3;
uniform bool texmap = true;
uniform bool shownormal = false;
uniform vec3 basecolor = vec3(1, 1, 1);
in fData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
} frag;

out vec4 out_color;


void main() {
    if (shownormal) {
      // mat4 m_view = m_camera * m_model;
      // mat3 m_normal = inverse(transpose(mat3(m_view)));
      // vec3 N = m_normal * normalize(frag.normal);
      vec3 N = normalize(frag.normal);
      out_color = vec4(N * 0.5 + 0.5, 1.0);
    } else {
      float kA = phong[0];
      float kD = phong[1];
      float kS = phong[2];

      vec3 color = texmap ? frag.color : basecolor;
      // color = vec4(1, 1, 1, 1);

      vec3 L = normalize(lightpos - frag.pos);
      vec3 N = normalize(frag.normal);
      float lambertian = max(dot(N, L), 0.0);
      float specular = 0;
      if (lambertian > 0.0) {
          // vec3 R = reflect(-L, N);
          vec3 R = 2 * (L * N) * N - L;
          vec3 V = normalize(-frag.pos);
          specular = pow(max(dot(R, V), 0.0), spec);
      // out_color = vec4(color * kA + basecolor * lambertian * kD + kS * specular, 1.0);
      }
      out_color = vec4(vec3(color * (kA + lambertian * kD + kS * specular)), 1.0);
    }
}

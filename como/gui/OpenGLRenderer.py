import os

import numpy as np
import torch
from OpenGL import GL as gl
from OpenGL.GL import *

from . import util


class OpenGLRenderer:
    def __init__(self, w, h):
        super().__init__()

        self.width_ = w
        self.height_ = h

        gl.glViewport(0, 0, w, h)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        # self.program_mtime = 0
        try:
            self.program = util.load_shaders(
                os.path.join(cur_path, "shaders/empty.vert"),
                os.path.join(cur_path, "shaders/drawkf.geom"),
                os.path.join(cur_path, "shaders/phong.frag"),
            )
        except Exception as e:
            print("failed to compile")
        # Initialize textures
        self.rgb_tid = util.Reinitialise(w, h, "rgb")
        self.depth_tid = util.Reinitialise(w, h, "float")
        self.valid_tid = util.Reinitialise(w, h, "float")

    def set_frag_vars(self, M, V, P):
        lightpos = np.array([0.0, 0.0, 0.0])

        # util.set_uniform_v3(self.program, lightpos, "lightpos")
        util.set_uniform_1int(self.program, 1, "phong_enabled")
        util.set_uniform_1int(self.program, 0, "normals_render")
        util.set_uniform_mat4(self.program, M.numpy(), "m_model")  # TODO: Also in geom?
        util.set_uniform_mat4(
            self.program, V.numpy(), "m_camera"
        )  # TODO: Also in geom?
        util.set_uniform_mat4(self.program, P.numpy(), "m_proj")  # TODO: Also in geom?

    def set_geom_vars(self, M, V, P, K):
        h = self.height_
        w = self.width_

        intrinsics = torch.as_tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        util.set_uniform_mat4(self.program, M.numpy(), "m_model")  # TODO: Also in geom?
        util.set_uniform_mat4(
            self.program, V.numpy(), "m_camera"
        )  # TODO: Also in geom?
        util.set_uniform_mat4(self.program, P.numpy(), "m_proj")  # TODO: Also in geom?

        # util.set_uniform_mat4(self.program, mvp, "mvp")  # TODO: Also in frag?
        util.set_uniform_v4(self.program, intrinsics, "cam")
        util.set_uniform_1int(self.program, w, "width")
        util.set_uniform_1int(self.program, h, "height")
        util.set_uniform_1f(self.program, 0.2, "slt_thresh")
        util.set_uniform_1int(self.program, 0, "crop_pix")

    def update_kf_textures(self, kf_rgb, kf_depth):
        # Prepare textures
        kf_valid = torch.ones_like(kf_depth)
        h = self.height_
        w = self.width_

        # Prep order!
        rgb_np = kf_rgb.permute(1, 2, 0).cpu().numpy()
        rgb_np_uint8 = np.ascontiguousarray((rgb_np * 255).astype(np.uint8))
        depth_np = kf_depth.permute(1, 2, 0).cpu().numpy()
        valid_np = kf_valid.permute(1, 2, 0).cpu().numpy()

        # Subtextures
        util.Upload(self.rgb_tid, rgb_np_uint8, h, w, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        util.Upload(self.depth_tid, depth_np, h, w, gl.GL_RED, gl.GL_FLOAT)
        util.Upload(self.valid_tid, valid_np, h, w, gl.GL_RED, gl.GL_FLOAT)

    #  This functon was derived from part of DeepFactors by Jan Czarnowski.

    #  Copyright (C) 2020 Imperial College London

    #  The use of the code within this file and all code within files that make up
    #  the software that is DeepFactors is permitted for non-commercial purposes
    #  only.  The full terms and conditions that apply to the code within this file
    #  are detailed within the LICENSE file and at
    #  <https://www.imperial.ac.uk/dyson-robotics-lab/projects/deepfactors/deepfactors-license>
    #  unless explicitly stated. By downloading this file you agree to comply with
    #  these terms.

    #  If you wish to use any of this code for commercial purposes then please
    #  email researchcontracts.engineering@imperial.ac.uk.

    def render_keyframe(self, kf_rgb, kf_depth, kf_pose, K, V, P):
        self.update_kf_textures(kf_rgb, kf_depth)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        # program_mtime = self.mtime_shader()
        # if self.program_mtime < program_mtime:
        #     try:
        #         self.program = util.load_shaders(
        #             os.path.join(cur_path, "shaders/empty.vert"),
        #             os.path.join(cur_path, "shaders/drawkf.geom"),
        #             os.path.join(cur_path, "shaders/phong.frag"),
        #         )
        #         print("Recompiled")
        #     except Exception as e:
        #         print("failed to compile")
        #         print(e)
        #     self.program_mtime = program_mtime
        # Activate program
        gl.glUseProgram(self.program)

        # TODO: Get relative pose between viewpoint and keyframe pos
        # T = P @ V
        # mvp = T @ kf_pose.numpy()
        M = kf_pose
        # Fill uniforms
        self.set_frag_vars(M, V, P)
        self.set_geom_vars(M, V, P, K)

        # Texture bank ids
        util.set_uniform_1int(self.program, 0, "image")
        util.set_uniform_1int(self.program, 1, "depth")
        util.set_uniform_1int(self.program, 2, "valid")

        # Setup texture banks
        gl.glActiveTexture(gl.GL_TEXTURE0)
        util.Bind(self.rgb_tid)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        util.Bind(self.depth_tid)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        util.Bind(self.valid_tid)

        # Draw keyframe
        gl.glDrawArrays(gl.GL_POINTS, 0, self.width_ * self.height_)

    # def mtime_shader(self):
    #     cur_path = os.path.dirname(os.path.abspath(__file__))
    #     os.path.join(cur_path, "shaders/empty.vert")
    #     os.path.join(cur_path, "shaders/drawkf.geom")
    #     os.path.join(cur_path, "shaders/phong.frag")

    #     return max(
    #         [
    #             os.path.getmtime(os.path.join(cur_path, "shaders/empty.vert")),
    #             os.path.getmtime(os.path.join(cur_path, "shaders/drawkf.geom")),
    #             os.path.getmtime(os.path.join(cur_path, "shaders/phong.frag")),
    #         ]
    #     )

import numpy as np
import OpenGL.GL.shaders as shaders
from OpenGL.GL import *


def load_shaders(vs, gs, fs):
    vertex_shader = open(vs, "r").read()
    geometry_shader = open(gs, "r").read()
    fragment_shader = open(fs, "r").read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(geometry_shader, GL_GEOMETRY_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name), 1, GL_FALSE, (content.T).astype(np.float32)
    )


def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(glGetUniformLocation(shader, name), content)


def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, name), content)


def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name), contents[0], contents[1], contents[2]
    )


def set_uniform_v4(shader, contents, name):
    glUseProgram(shader)
    glUniform4f(
        glGetUniformLocation(shader, name),
        contents[0],
        contents[1],
        contents[2],
        contents[3],
    )


# Replicating Pangolin functions below from
# https://github.com/stevenlovegrove/Pangolin/blob/52e84ae29b76929d69cff818d05ec5516270d8ca/components/pango_opengl/include/pangolin/gl/gl.hpp#L174


def Bind(tid):
    glBindTexture(GL_TEXTURE_2D, tid)


# glTexImage2D creates the storage for the texture
def Reinitialise(w, h, tex_type):
    tid = glGenTextures(1)
    Bind(tid)

    border = 0

    if tex_type == "rgb":
        int_format = GL_RGBA8
        glformat = GL_RGB
        gltype = GL_UNSIGNED_BYTE
        data = np.zeros((h, w, 3), dtype=np.uint8)
    elif tex_type == "float":
        int_format = GL_R32F
        glformat = GL_RED
        gltype = GL_FLOAT
        data = np.zeros((h, w), dtype=np.float32)

    glTexImage2D(GL_TEXTURE_2D, 0, int_format, w, h, border, glformat, gltype, data)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

    return tid


# glTexSubImage2D only modifies pixel data within the texture
def Upload(tid, data, h, w, data_format, data_type):
    Bind(tid)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, data_format, data_type, data)

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os.path as osp
import torch

ROOT = osp.dirname(osp.abspath(__file__))

has_cuda = torch.cuda.is_available()

include_dirs = [
    osp.join(ROOT, "como/backend/include"),
]

sources = [
    "como/backend/src/cov.cpp",
    "como/backend/src/cov_cpu.cpp",
    "como/backend/src/depth_cov_backends.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("como/backend/src/cov_gpu.cu")
    extra_compile_args["cxx"].append("-DBACKEND_WITH_CUDA=1")
    extra_compile_args["nvcc"] = [
        "-O3",
        "-gencode=arch=compute_86,code=sm_86",
        "-DBACKEND_WITH_CUDA=1",
    ]
    ext_modules = [
        CUDAExtension(
            "como_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    ext_modules = [
        CppExtension(
            "como_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="como",
    version="0.1.0",
    author="edexheim",
    packages=["como"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

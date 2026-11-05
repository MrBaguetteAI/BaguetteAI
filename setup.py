from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "BaguetteAI.util._core",
        ["src/BaguetteAI/util/_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
)

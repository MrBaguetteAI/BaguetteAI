from setuptools import setup, Extension
import pybind11

ext_modules = [

    Extension(
        "BaguetteAI.neural.neural_functions",
        ["src/BaguetteAI/neural/neural_functions.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
)

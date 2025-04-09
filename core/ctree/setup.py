from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# noinspection PyInterpreter
extensions = [
    Extension(
        "cytree",
        ["cytree.pyx"],
        extra_compile_args=['-O3', '-std=c++11'],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(ext_modules=cythonize(extensions))

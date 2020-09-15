from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("cython_wathed", ["cython_wathed.pyx"]
              #, define_macros=[('CYTHON_TRACE', '1')]
             )
]

setup(
    name = "cython_wathed",
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
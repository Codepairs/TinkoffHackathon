#from distutils.core import Extension, setup


import distutils.core

import numpy
from Cython.Build import cythonize
# define an extension that will be cythonized and compiled
ext = distutils.core.Extension(name="algorithm_cython", sources=["algorithm_cython.pyx"], include_dirs=[numpy.get_include()], define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
distutils.core.setup(ext_modules=cythonize(ext, compiler_directives={'language_level': 3, "always_allow_keywords": True}))

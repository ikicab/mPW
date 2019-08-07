# -*- coding: utf-8 -*-
"""
@author: Barbara Ikica
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
 
setup(
    ext_modules = cythonize(Extension("mPW", sources=["mPW.pyx"], language="c++"),
    compiler_directives={'language_level' : 3}),
    include_dirs=[numpy.get_include()]
)

#python setup.py build_ext --inplace
#cython -a mPW.pyx
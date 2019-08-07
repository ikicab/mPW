# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:28:49 2019

@author: barbe
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
#cython -a PWC.pyx
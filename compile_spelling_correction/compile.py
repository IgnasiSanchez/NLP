"""
python3 compile.py build_ext --inplace
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

p1 = os.getcwd()
p2 = "Spelling_Correction_c.pyx"
path = os.path.join(p1,p2)

setup(ext_modules=cythonize(p2),include_dirs=[numpy.get_include()])

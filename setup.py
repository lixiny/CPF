from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy as np

# get np include path
np_include_path = np.get_include()

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    "hocontact.utils.libmesh.triangle_hash",
    sources=["hocontact/utils/libmesh/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[np_include_path],
)

# Gather all extension modules
ext_modules = [
    triangle_hash_module,
]

setup(
    name="hocontact",
    version="0.0.1",
    python_requires=">=3.6.0",
    ext_modules=cythonize(ext_modules),
    cmdclass={"build_ext": BuildExtension},
)

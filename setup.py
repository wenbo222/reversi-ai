from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "reversi_ai._reversi_ai",
        ["reversi_ai.cpp"],
        cxx_std=20,
    ),
]

setup(
    name="reversi_ai",
    version="0.1.3",
    author="Wenbo Wang",
    author_email="wangwenbo205@gmail.com",
    description="An AI for a custom version of Reversi",
    url="https://github.com/wenbo222/reversi-ai",
    packages=find_packages(),
    package_data={
        "reversi_ai": ["reversi_ai.dat"],
    },
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
)
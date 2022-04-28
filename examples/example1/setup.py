"""Setup for example model."""

from setuptools import find_packages, setup

requires = ["mesa_ret", "mesa"]

version = "v0a"
with open("Readme.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="ret_example",
    version=version,
    description="Example model for RET",
    long_package_description=readme,
    packages=find_packages(),
    install_requires=requires,
    package_data={"ret_example": ["images/*.png", "images/*/*.svg"]},
)

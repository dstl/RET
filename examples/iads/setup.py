"""Setup for IADS model."""

from setuptools import find_packages, setup

requires = ["mesa_ret", "mesa"]
extras_require = {"dev": ["flake8", "black"]}

version = "20220303"
with open("Readme.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="iads",
    version=version,
    description="IADS demonstration",
    long_package_description=readme,
    packages=find_packages(),
    install_requires=requires,
    package_data={"iads": ["images/*.png", "images/*/*.svg"]},
)

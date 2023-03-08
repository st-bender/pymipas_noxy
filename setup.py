from codecs import open
from os import path
import re

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

name = "mipas_noxy"
meta_path = path.join("src", name, "__init__.py")
here = path.abspath(path.dirname(__file__))

extras_require = {
    "tests": ["pytest"],
}
extras_require["all"] = sorted(
    {v for req in extras_require.values() for v in req}
)


# Approach taken from
# https://packaging.python.org/guides/single-sourcing-package-version/
# and the `attrs` package https://www.attrs.org/
# https://github.com/python-attrs/attrs
def read(*parts):
    """
    Builds an absolute path from *parts* and and return the contents of the
    resulting file.  Assumes UTF-8 encoding.
    """
    with open(path.join(here, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, *path):
    """
    Extract __*meta*__ from *path* (can have multiple components)
    """
    meta_file = read(*path)
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        meta_file,
        re.M,
    )
    if not meta_match:
        raise RuntimeError("__{meta}__ string not found.".format(meta=meta))
    return meta_match.group(1)


# Get the long description from the README file
long_description = read("README.md")
version = find_meta("version", meta_path)

if __name__ == "__main__":
    setup(
        name=name,
        version=version,
        description="MIPAS level-2 NOx/NOy calculater",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/st-bender/pymipas_noxy",
        author="Stefan Bender",
        author_email="sbender@iaa.es",
        license="GPLv2",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Physics",
            "Topic :: Utilities",
        ],
        packages=find_packages("src"),
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=[
            "numpy",
            "xarray",
            "netCDF4",
            "tomli",
        ],
        extras_require=extras_require,
        scripts=[],
        entry_points={
            "console_scripts": [
                "mipas_noxy = mipas_noxy.__main__:main",
            ]
        },
        options={
            "bdist_wheel": {"universal": True},
        },
        zip_safe=False,
    )

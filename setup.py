import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynetan", 
    version="1.0.2",
    author="Marcelo C. R. Melo",
    author_email="melomcr@gmail.com",
    description="A Python implementation for Dynamical Network Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data = True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
    python_requires='>=3.7',
    install_requires=[
        "MDAnalysis",
        "SciPy",
        "NumPy",
        "nglview",
        "colorama",
        "pandas",
        "ipywidgets",
        "networkx",
        "numba",
        "h5py",
        "pympler",
        "python-louvain",
        "tzlocal",
        "recommonmark"
    ],
)


============
Installation
============

Installing with `pip`
---------------------

To install this package and all required Python packages, simply run::

    $ pip install dynetan

Requirements
------------

The core package requires Python 3.7 or greater and the follwoing python packages:

- MDAnalysis 
- SciPy 
- NumPy 
- pandas 
- networkx 
- numba 
- h5py 
- python-louvain

The following packages are not necessary for the core fucntionalyty, but are suggested for use along with jupyter notebooks: 

- ipywidgets 
- colorama 
- nglview 
- rpy2 
- pympler
- tzlocal

Build the package from source:
-------------------------------

Ensure pip, setuptools, and wheel are up to date::

    $ python -m pip install --upgrade pip setuptools wheel

Create a Wheel file locally::

    $ python3 setup.py sdist bdist_wheel

Troubleshooting
---------------

Installing with `pip`
^^^^^^^^^^^^^^^^^^^^^

- If during the installation process with `pip` you find the error `fatal error: Python.h: No such file or directory`, make sure your Linux distribution has the necessary development packages for Python. They contain header files that are needed for the compilation of packages such as MDAnalysis. These packages will be listed as "python3-dev" or similar. For example, in Fedora 32, one can use the command `dnf install python3-devel` to install additional system packages with Python headers.

- Similarly, if during the installation process you find the error `gcc: fatal error: cannot execute ‘cc1plus’: execvp: No such file or directory`, make sure you have development tools for gcc and c++. For example, in Fedora 32, one can use the command `dnf install gcc-c++` to install additional system packages with c++ development tools.

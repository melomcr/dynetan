=========
Reference
=========

..    :Release: |release|
..    :Date: |today|

The documentation provided here (attempts to) follow the Google style of code
documentation, and is built using Sphinx and its Napoleon module.

Process Trajectory Data
-----------------------

This dedicated class controls all trajectory processing necessary for Dynamical
Network Analysis.

.. automodule:: dynetan.proctraj
   :members:

Save and Load Data
----------------------

This dedicated class stores and recovers results from Dynamical Network Analysis.

.. automodule:: dynetan.datastorage
   :members:
   
Contact Detection
-----------------

This module contains auxiliary functions for the parallel calculation of node contacts.

.. automodule:: dynetan.contact
   :members:

Generalized Correlations
------------------------

This module contains auxiliary functions for the parallel calculation of
generalized correlation coefficients.

.. automodule:: dynetan.gencor
   :members:

Network Properties
-------------------

This module contains auxiliary functions for the parallel calculation of network
properties.

.. automodule:: dynetan.network
   :members:
   
Toolkit
--------

This module contains auxiliary functions for manipulation of atom selections,
acquiring pre-calculated cartesian distances, and user interface in jupyter
notebooks.

.. automodule:: dynetan.toolkit
   :members:

Visualization
--------------

This module contains auxiliary functions for visualization of the system and
network analysis results.

.. automodule:: dynetan.viz
   :members:

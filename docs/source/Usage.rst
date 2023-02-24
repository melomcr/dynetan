========
Usage
========

This package was built to provide an updated and enhanced Python implementation
of the Dynamical Network Analysis method, for the analysis of Molecular Dynamics
simulations. The package was optimized for both interactive use through Jupyter
Notebooks (see :ref:`Tutorial <Tutorial>`) and for command-line-interface use
(such as in scripts for remote execution. The package allows for extensive
customization of analysis to suit research-specific needs.

We present below an overview of the process of analysing an MD simulation,
using the OMP decarboxylase example that was examined in the reference
publication (see :ref:`Citing <Citing>`) and the associated
:ref:`Tutorial <Tutorial>`.

Load your simulation data by creating a :py:class:`~dynetan.proctraj.DNAproc` object::

    # Load the python package
    import os
    import dynetan

    # Create the object that processes MD trajectories.
    dnap = DNAproc()


Select the location of simulation files::

    # Path where input files will searched and results be written.
    workDir = "./TutorialData/"

    # PSF file name
    psfFile = os.path.join(workDir, "decarboxylase.0.psf")

    # DCD file name
    dcdFiles = [os.path.join(workDir, "decarboxylase.1.dcd")]

Select the number of windows into which your trajectory will be split.
This can correspond to a long contiguous simulation or multiple independent
concatenated replicas of the same system::

    # Number of windows created from full simulation.
    numWinds = 4

    # Sampled frames per window (for detection of structural waters)
    numSampledFrames = 10

Select a ligand to be analysed and segment IDs for the biomolecules to be studied.
For automatic detection of structural water molecules, provide the name of the
solvent residue::

    ligandSegID = "OMP"

    # Segment IDs for regions that will be studied.
    segIDs = ["OMP","ENZY"]

    # Residue name for solvent molecule(s)
    h2oName = ["TIP3"]

Set the node groups for user-defined residues::

    # Network Analysis will make one node per protein residue (in the alpha carbon)
    # For other residues, the user must specify atom(s) that will represent a node.
    customResNodes = {}
    customResNodes["TIP3"] = ["OH2"]
    customResNodes["OMP"] = ["N1","P"]

    # We also need to know the heavy atoms that compose each node group.

    usrNodeGroups = {}

    usrNodeGroups["TIP3"] = {}
    usrNodeGroups["TIP3"]["OH2"] = set("OH2 H1 H2".split())

    usrNodeGroups["OMP"] = {}
    usrNodeGroups["OMP"]["N1"] = set("N1 C2 O2 N3 C4 O4 C5 C6 C7 OA OB".split())
    usrNodeGroups["OMP"]["P"] = set("P OP1 OP2 OP3 O5' C5' C4' O4' C1' C3' C2' O2' O3'".split())

Define the parameters that will control contact detection::

    # Cutoff for contact map (In Angstroms)
    cutoffDist = 4.5

    # Minimum contact persistance (In ratio of total trajectory frames)
    contactPersistence = 0.75

Finally, load all data to the :py:class:`~dynetan.proctraj.DNAproc` object::

    ### Load info to object

    dnap.setNumWinds(numWinds)
    dnap.setNumSampledFrames(numSampledFrames)
    dnap.setCutoffDist(cutoffDist)
    dnap.setContactPersistence(contactPersistence)
    dnap.seth2oName(h2oName)
    dnap.setSegIDs(segIDs)

    dnap.setCustomResNodes(customResNodes)
    dnap.setUsrNodeGroups(usrNodeGroups)


In its simplest form, the code will load the MD simulation, detect structural
water molecules, and create a network representation of the nodes selected so far::

    dnap.loadSystem(psfFile,dcdFiles)

    dnap.selectSystem(withSolvent=True)

    dnap.prepareNetwork()

After the nodes and node groups are selected, the system is aligned, contacts
are detected, and the calculation of correlation coefficients can begin::

    dnap.alignTraj(inMemory=True)

    dnap.findContacts(stride=1)

    dnap.calcCor(ncores=1)

With the correlation matrix of each simulation window, we create graph
representations for each simulation window, and calculate network properties
such as optimal paths, betweenness and communities::

    dnap.calcGraphInfo()

    dnap.calcOptPaths(ncores=1)

    dnap.calcBetween(ncores=1)

    dnap.calcCommunities()

To automate the detection of edges between two separate subunits of a biomolecular
complex, we can specify segment IDs and request the identification of interface
connections::

    dnap.interfaceAnalysis(selAstr="segid ENZY", selBstr="segid OMP")

Finally, all data can be saved to disk::

    dnap.saveData(fullPathRoot)


All the interactive visualization of the structure and network nodes and edges,
optimal paths, communities, and high resolution rendering are performed through
jupyter notebooks. Please refer to the :ref:`Tutorial <Tutorial>` for
detailed examples.

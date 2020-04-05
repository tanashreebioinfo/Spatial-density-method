# Method-development1 (1. Kernel-density-based-spatial distribution estimation(SPD))

This repository consist Python codes written for calculating spatial density distribution using kernel density estimation apparoch. This methods are useful to capture the probabble atomic poistions around solute. example of urea solution aroubd N-methylacetamide moleucles is shown.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes. 

### Prerequisites

Python libraries required to use the code

```
*mdtraj 
*sys, math, random, copy, pickle, re
*numpy 
*matplotlib
*scipy 
*mayavi 
*mpl3d
*sklearn
*joblib

```

### Installing
Python 3 or above version is required to setup the dependencies

```
use commands below to activate loacal environment anaconda
conda create -n method python=3.6
source activate method

e.g. to install python libraries listed above

conda install -c omnia mdtraj  (similarly install other libraries using conda or pip)



## Running the codes

spatial-density-distribution.py takes 3 standard inputs from user i.e. dcd (obtained using NAMD simulation program), pdb file, model residue name (can be obatained from segid column of PDB), and psf (toplogy file)

To run the automated tests for this system
cp.sh can be used as:
bash cp.sh
please note that cp.sh takes arguments from another bash files copies data into successive folders and use python code for generating SPDs

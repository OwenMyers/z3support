
This repo contains plotting and analysis tools for all things related to 
the Quantum Dimer Pentamer Model (QDPM) project.

# Setup

Install anaconda

Make virtual environment

```
conda create -n z3support python=3
```

You probably need to initialize your shell

```
conda init bash
``` 

Change `bash` to whichever shell you use if not bash

Now activate the environment

```
conda activate z3support
```

Install the requirements

```
pip install -r requirements.txt
```

and set the rest up for develop 

```
python setup.py develop
```

# Plot Lattices

* Place lattice files in the `lattices` directory
* Run `python plot_lattices.py` with no arguments
* You will find figures in the `figures/lattices`

Note:
You have to set the lattice size in the python script. Has to be hard coded.
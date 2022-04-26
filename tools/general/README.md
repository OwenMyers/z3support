# Plotting Lattice Configurations


Note: The prefix for python I'm using is just to make sure the
person running ensures they have the expected python3 binary

`cd` in to this directory

See if you already have `virtualenv`

```
/usr/bin/python3 -m virtualenv --version
```

Install it if you don't

```
/usr/bin/python3 -m pip install virtualenv
```

Create a virtual environment here called `venv`


```
/usr/bin/python3 -m virtualenv venv
```

activate

```
source venv/bin/activate[.zsh, .csh, etc]
```


Now that it is activated `which python` should return the python in 
the `venv` path. 

Upgrate your virtualenv pip (default pip will be the virtualenv pip)

```
pip install --upgrade pip
```

cd into the root of the repo

```
cd ../../
```

install the local files

```
python setup.py develop
```

come back here

```
cd tools/general/
```

Install requirements

```
pip install -r requirements.txt
```

Make sure you have your lattice files in `./lattices/`

run

```
python plot_lattices.py
```

and output will be in `figures/lattices/`
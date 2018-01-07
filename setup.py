
import os
from setuptools import setup

# Keeping comment from documentatoin site because it is informative:
# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "z3support",
    version = "0.0.1",
    author = "Owen Myers",
    author_email = "oweenm@gmail.com",
    description = ("QDPM and Z3 string net plotting and analysis"),
    #license = "BSD",
    url = "https://github.com/OwenMyers/z3support",
    packages=['z3support'],
    long_description=read('README.md'),
)

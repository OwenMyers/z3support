Prepare Data
============
There are several aspects of the lattice configurations that can make
data prep challenging.
* We are exploring degrees of freedom on the links of the lattice which does not
  fit nicely into a pixel -like matrix
* The number of degrees of freedom can be different between systems so we need to
  represent them "fairly" to the algorithm (some kind of normalization)
* Degrees of freedom may be relative to a vertex


This is a script that makes preperation easy and can be modified to incorperate
additional prep steps depending on the system. For now it can be used as follows:

Usage notes (asking for help on the script like `python transform_data_for_autoencoder.py --help`
provides the following as well):

.. code-block:: txt

   usage: transform_data_for_autoencoder.py [-h] --src SRC --parse-type
                                            PARSE_TYPE --destination DESTINATION
                                            [--truncate TRUNCATE]
   
   optional arguments:
     -h, --help            show this help message and exit
     --src SRC             Path to directory containing configuration files
     --parse-type PARSE_TYPE What system configurations are you parsing
     --destination DESTINATION Path to save transformed file


The above creates files that can be easily ingested by the rest of the autoencoder. See next
for setting up and running the autoencoder.

Function and Code Docs
======================
Below are the function docs for the tools used to prep the data

:mod:`src.transform_data_for_autoencoder` -- Prepare Data
---------------------------------------------------------

.. automodule:: src.transform_data_for_autoencoder
   :members:
   :undoc-members:
   :show-inheritance:

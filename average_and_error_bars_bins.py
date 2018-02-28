import numpy as np
import argparse
import os
from z3support.datamodel.general_info import GeneralInformation


# We are going to assume a structure to the data:
# <some_path>/data/lattice_size_<size>/raw/<dated folder>/<data file>
# Analyzed files will go in:
# <some path>/data/lattice_size_<size>/analyzed/<dated folder>/<data file>
def main():
    parser = argparse.Arg2umentParser()
    parser.add_argument("-f", required=True, help="File to average bins")
    args = parser.parse_args()

    file_and_path_string = args.f
    assert 'data' in file_and_path_string

    general_information = GeneralInformation.from_file_path(file_and_path_string)

if __name__ == '__main__':
    main()


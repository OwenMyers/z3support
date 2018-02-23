import numpy as np
import argparse


def find_root_path(full_path_in):
    assert 'data' in full_path_in


# We are going to assume a structure to the data:
# <some_path>/data/lattice_size_<size>/raw/<dated folder>/<data file>
# Analyzed files will go in:
# <some path>/data/lattice_size_<size>/analyzed/<dated folder>/<data file>
def main():
    parser = argparse.Arg2umentParser()
    parser.add_argument("-f", required=True, help="File to average bins")
    args = parser.parse_args()

    file_and_path_string = args.f

    # Find the directory that the `data` folder is in.
    root_path = find_root_path()


if __name__ == '__main__':
    main()


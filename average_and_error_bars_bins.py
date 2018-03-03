import numpy as np
import argparse
import os
from z3support.datamodel.general_info import GeneralInformation
from z3support import data_tools


def create_if_necessary_analysis_path(general_information):

    analysis_path = os.path.join(general_information.lattice_size_path,
                                 "analysis")

    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)

    return analysis_path


def create_if_necessary_dated_path(analysis_path, general_information):

    dated_path = os.path.join(analysis_path,
                              general_information.date_as_string())

    if not os.path.exists(dated_path):
        os.mkdir(dated_path)

    return dated_path


# We are going to assume a structure to the data:
# <some_path>/data/lattice_size_<size>/raw/<dated folder>/<data file>
# Analyzed files will go in:
# <some path>/data/lattice_size_<size>/analyzed/<dated folder>/<data file>
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="File to average bins")
    args = parser.parse_args()

    file_and_path_string = args.f
    assert 'data' in file_and_path_string

    general_information = GeneralInformation.from_file_path(file_and_path_string)
    analysis_path = create_if_necessary_analysis_path(general_information)
    dated_path = create_if_necessary_dated_path(analysis_path)

    f = open(file_and_path_string, "r")
    header_line = f.readline()
    data = np.genfromtxt(f, delimiter=',')

    data_tools.check_data(data)
    number_of_real_vertices = data_tools.find_number_of_real_vertices_from_data(data)

    assert number_of_real_vertices == \
        (general_information.system_size.x * general_information.system_size.y) / 2

    data = data.reshape(-1, number_of_real_vertices, 6)

    for d in data:
        print(d)

    print("Done")


if __name__ == '__main__':
    main()


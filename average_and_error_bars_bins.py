import numpy as np
import argparse
import os
from z3support.datamodel.general_info import GeneralInformation
from z3support import data_tools


def average_and_error_data(data, number_of_real_vertices):
    """
    From the correctly shaped data find the average and the error.

    Average over the bins and also find the error bars for each. Pay careful attention to the shape
    of the data and make sure you don't forget your factor of 1/sqrt(N)

    :param data: numpy array with shape (N bins, N real vertices, 6) where 6 is for
        x loc, y loc, North, East, South, West.
    :param number_of_real_vertices: an unnecessary parameter because it can be found from
        len(data[0,:,0]) but it makes things more readable.
    :return: two 2D arrays: one of the averages, the other of the error bars. shape of
        each is (N real vertices, 6) because we are keeping the x, y locations for
        convenience.
    """

    averages = np.zeros([number_of_real_vertices, 6])
    error_bars = np.zeros([number_of_real_vertices, 6])

    number_of_bins = data.shape[0]

    for i in range(number_of_real_vertices):
        for j in range(6):
            current_average = data[:, i, j].mean()
            averages[i, j] = current_average

            current_error_bar = data[:, i, j].std() / np.sqrt(float(number_of_bins))
            error_bars[i, j] = current_error_bar
            if j < 2:
                assert current_error_bar < data_tools.EPSILON

    return averages, error_bars


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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", default=None, help="File to average bins")
    group.add_argument("-d", default=None, help="Directory containing files. Operate on all CSV files in directory")
    args = parser.parse_args()

    file_and_path_string = args.f
    path_with_multiple_files = args.d

    file_and_path_list = []
    if file_and_path_string is not None:
        file_and_path_list.append(file_and_path_string)
    if path_with_multiple_files is not None:
        for potential_file in os.listdir(path_with_multiple_files):
            if potential_file[-4:] == ".csv":
                this_file_and_path = os.path.join(path_with_multiple_files, potential_file)
                file_and_path_list.append(this_file_and_path)

    for file_and_path_string in file_and_path_list:
        print("Working on file: {}".format(file_and_path_string))
        assert 'data' in file_and_path_string

        general_information = GeneralInformation.from_file_path(file_and_path_string)
        analysis_path = create_if_necessary_analysis_path(general_information)
        dated_path = create_if_necessary_dated_path(analysis_path, general_information)

        f = open(file_and_path_string, "r")
        header_line = f.readline().rstrip()
        data = np.genfromtxt(f, delimiter=',')

        data_tools.check_data(data)
        number_of_real_vertices = data_tools.find_number_of_real_vertices_from_data(data)

        assert number_of_real_vertices == \
            (general_information.system_size.x * general_information.system_size.y) / 2

        data = data.reshape(-1, number_of_real_vertices, 6)

        averages, error_bars = average_and_error_data(data, number_of_real_vertices)

        averages_file_and_path = os.path.join(dated_path,
                                              general_information.file_name_no_extension
                                                + "_averages_over_bins.csv"
                                              )
        error_bars_file_and_path = os.path.join(dated_path,
                                                general_information.file_name_no_extension
                                                    + "_error_bars.csv"
                                                )

        np.savetxt(averages_file_and_path, averages, delimiter=",", header=header_line, comments='')
        np.savetxt(error_bars_file_and_path, error_bars, delimiter=",", header=header_line, comments='')

    print("Done")


if __name__ == '__main__':
    main()


import os
import argparse
import pandas as pd
import tqdm
import numpy as np
import logging


def string_to_number_directions_1(r, column):
    """
    If this function is being used then the data has strings representing directions in each element of the "image"
    matrix. This is one way of converting those strings to numbers for the CNN

    Also, this is supposed to be used in a pandas apply; useful context for the arguments.

    :param r: a row from a pandas ``apply``
    :param column: The column of the data to work on
    :return: integer of the string to int map
    """
    link_str = r[column]
    if link_str == 'B':
        return 1
    elif link_str == 'N':
        return 2
    elif link_str == 'S':
        return 3
    elif link_str == 'E':
        return 4
    elif link_str == 'W':
        return 5


def string_to_number_directions_2(r, column):
    """
    If this function is being used then the data has strings representing directions in each element of the "image"
    matrix. This is one way of converting those strings to numbers for the CNN

    Also, this is supposed to be used in a pandas apply; useful context for the arguments.

    :param r: a row from a pandas ``apply``
    :param column: The column of the data to work on
    :return: integer of the string to int map
    """
    link_str = r[column]
    if link_str == 'B':
        return 1
    elif link_str == 'N':
        return 2
    elif link_str == 'S':
        return 3
    elif link_str == 'E':
        return 2
    elif link_str == 'W':
        return 3


def apply_string_to_number_all_directions(df_in):
    """
    This function uses the string-to-number mapping functions to convert a matrix with string elements to a matrix
    with number elements. Strings represent some direction in a lattice but we need to represent them numerically
    for the CNN

    :param df_in: the dataframe containing the matrix with string elements
    :return: Returns a copy of the dataframe after the string-to-int mapping has been made
    """
    df_copy = df_in.copy()
    df_copy['n_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'N'), axis=1)
    df_copy['e_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'E'), axis=1)
    df_copy['s_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'S'), axis=1)
    df_copy['w_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'W'), axis=1)
    return df_copy


def determine_lattice_size(df_in):
    """
    Using the input data, determine the size of the lattice.

    :param df_in: dataframe containing the data (representation of a lattice configuration.
    :return: ``int`` of the lattice size (assuming square lattice)
    """
    # Assume square lattice
    max_x = df_in['x'].values.max()
    max_y = df_in['y'].values.max()
    if max_x != max_y:
        raise ValueError('Expecting x==y dimensions')
    return max_x + 1


def check_if_exists(cur_val, proposed_val, v=False):
    """
    Helper function to help check if there are any inconsistencies in the lattice "description".
    For context A data file that comes from the z3 work will have rows that correspond to a vertex. That means that in
    the data file a given link will be represented in 2 rows, one for each vertex on either side of the link. This
    function helps us check that a links representation is consistent.

    :param cur_val: The established (previous) vertex's notion of this links value
    :param proposed_val: The current "working" vertex's notion of this links value
    :param v: verbose or not boolean
    :return: None, but will raise a ``ValueError`` if there is a problem.
    """
    if v:
        print('  In check_if_exists\n')
        print(f'    cur_val {cur_val}')
        print(f'    proposed_val {proposed_val}')
    if cur_val == 0:
        pass
    elif cur_val != proposed_val:
        raise ValueError("Discovered inconsistency in representation.")


def create_full_numerical_representation(df_in, lattice_size, v=False):
    """
    Creates a matrix of numbers that can be interpreted by a CNN auto encoder.

    Requires running the ``string_to_number_directions`` function first.

    Checks for consistency in the plaquette representations of the configurations.

    :param df_in: is the dataframe of a plaquette representation of a configuration for which you have run
        the ``string_to_number_directions`` on
    :type DataFrame:
    :param lattice_size: Lattice length or width (assumed square)
    :type int:
    :param v: Verbose or not
    :type bool:

    :return: A numpy matrix with zeros representing the vertices and centers of plaquetts and the number system as described
        by ``string_to_number_directions`` representing the links.
    """
    df_working = df_in.copy()

    # We are currently setting this in the settings file so leave this line commented out for now.
    # l = determine_lattice_size(df_working)

    # This is the matrix that we will return this matrix
    m = np.zeros([2 * lattice_size, 2 * lattice_size])
    for i in range(lattice_size):
        for j in range(lattice_size):
            cur_row = df_working.loc[j, i]

            # For all entries we will check for consistency between the plaquetts. E.g. bottom(top) of the previous
            # row of plaquetts with the top(bottom) of the current row -> these need to be the same and if they are not
            # their is either a problem with the way you are writing the plaquetts to file, or with the algorithm
            # generating the configurations.
            horizontal_index_x = j * 2 + 1
            horizontal_index_y = -(i * 2) - 1
            vertical_index_x = j * 2
            vertical_index_y = -(i * 2 + 1) - 1
            if v:
                print(f'i (y): {i}')
                print(f'j (x): {j}')
                print(f'horizontal_index_x {horizontal_index_x}')
                print(f'horizontal_index_y {horizontal_index_y}')
                print(f'vertical_index_x {vertical_index_x}')
                print(f'vertical_index_y {vertical_index_y}')

            # horizontal
            check_if_exists(m[horizontal_index_y, horizontal_index_x], cur_row['s_number'], v=v)
            m[horizontal_index_y, horizontal_index_x] = cur_row['s_number']
            check_if_exists(m[-((-horizontal_index_y + 2) % (2 * lattice_size)), horizontal_index_x],
                            cur_row['n_number'], v=v)
            m[-((-horizontal_index_y + 2) % (2 * lattice_size)), horizontal_index_x] = cur_row['n_number']

            # vertical
            check_if_exists(m[vertical_index_y, vertical_index_x], cur_row['w_number'], v=v)
            m[vertical_index_y, vertical_index_x] = cur_row['w_number']
            check_if_exists(m[vertical_index_y, (vertical_index_x + 2) % (2 * lattice_size)], cur_row['e_number'], v=v)
            m[vertical_index_y, (vertical_index_x + 2) % (2 * lattice_size)] = cur_row['e_number']
            if v:
                print('current m:\n')
                print(m)
    return m


def final_output(func):
    """
    A decorator to aid in final logging, simple sanity checks, and writing of the final matrix to file

    :param func: The function that is being decorated
    :return:
    """
    def wrapper_final_output():
        logging.info("In final output decorator")
        save_file_name, matrix_list = func()
        logging.info(f"    Save file name: {save_file_name}")
        logging.info(f"    Number of matrices in output: {len(matrix_list)}")
        if len(matrix_list) == 0:
            raise ValueError('No files successfully transformed. Check src directory.')
        full_save_path = os.path.join(DESTINATION, save_file_name + '.npy')
        logging.info(f"    Trying to save to {full_save_path}\n...")
        np.save(full_save_path, np.array(matrix_list))
        logging.info("    Done saving and done in decorator wrapper")

    return wrapper_final_output


@final_output
def parse_owen_z3():
    """
    Function to parse z3 data

    :return: <data name/type>, a list of all of the matrices to train on
    """
    matrix_list = []
    file_list = os.listdir(SRC)

    lattice_size = None
    for i, cur_file in enumerate(tqdm.tqdm(file_list)):
        if (TRUNCATE > 0) and (i > TRUNCATE):
            break
        # noinspection SpellCheckingInspection
        if ('.csv' != cur_file[-4:]) or ('plaquett' not in cur_file):
            continue
        current_df = pd.read_csv(os.path.join(SRC, cur_file))
        if i == 0:
            lattice_size = determine_lattice_size(current_df)
        current_df = apply_string_to_number_all_directions(current_df)
        current_df.set_index(['x', 'y'], inplace=True)
        current_matrix = create_full_numerical_representation(current_df, lattice_size)
        matrix_list.append(current_matrix)

    return 'z3_data', matrix_list


def z2_row_to_matrix(row, lattice_size):
    """
    Currently the z2 data is organized as a full configuration per line. One file contains multiple configurations
    and each, as a line, can be transformed into a matrix. That transformation, from a row to a matrix is what we
    do in this function.

    :param row: row to be transformed into matrix
    :param lattice_size:
    :return:
    """
    cur_transformed_matrix = np.zeros([lattice_size * 2, lattice_size * 2])
    count = 0
    for k in range(lattice_size):
        for j in range(lattice_size):
            raw_horizontal_index = count
            raw_vertical_index = count + 1
            transformed_horizontal_index_y = 2 * j + 1
            transformed_horizontal_index_x = k * 2

            transformed_vertical_index_y = 2 * j
            transformed_vertical_index_x = k * 2 + 1

            # Transformed value notes
            # Input data will have a blank or spin down as -1. spit up or not blank will be 1
            # For CNN rep. of data 0s are centers of plaquetts and vertices so we can use zero. We want it to be
            # commensurate with string_to_number_directions_2 so
            # if -1 -> 1
            # if 1 -> 3 (the max representation in z3. this at least gets you the max difference though we probably
            # need to re-think it...
            # ^ TODO
            to_set_horizontal = None
            if row[raw_horizontal_index] < 0:
                to_set_horizontal = 1
            elif row[raw_horizontal_index] > 0:
                to_set_horizontal = 3
            else:
                # we get zero
                raise ValueError('Got unexpected 0 in horizontal z2 data')
            to_set_vertical = None
            if row[raw_vertical_index] < 0:
                to_set_vertical = 1
            elif row[raw_vertical_index] > 0:
                to_set_vertical = 3
            else:
                # we get zero
                raise ValueError('Got unexpected 0 in vertical z2 data')
            cur_transformed_matrix[transformed_horizontal_index_x, transformed_horizontal_index_y] = to_set_horizontal
            cur_transformed_matrix[transformed_vertical_index_x, transformed_vertical_index_y] = to_set_vertical

            count += 2
    return cur_transformed_matrix


@final_output
def parse_as_rows_z2():
    """
    Parses the z2 data. Uses ``z2_row_to_matrix`` on each row. Each row represents a full configuration
    :return:
    """
    matrix_list = []
    file_list = os.listdir(SRC)

    lattice_size = None
    for i, cur_file in enumerate(tqdm.tqdm(file_list)):
        if (TRUNCATE > 0) and (i > TRUNCATE):
            break
        if '.txt' != cur_file[-4:]:
            continue
        current_df = pd.read_csv(os.path.join(SRC, cur_file), delimiter=' ', header=None)
        if i == 0:
            lattice_size = int(np.sqrt((current_df.shape[1] - 1) / 2))
        for j in range(current_df.shape[0]):
            row = current_df.iloc[j]
            matrix = z2_row_to_matrix(row, lattice_size)
            matrix_list.append(matrix)

    return 'z2_data', matrix_list


def main():
    if PARSE_TYPE == 'owen_z3':
        parse_owen_z3()
    if PARSE_TYPE == 'as_rows_z2':
        parse_as_rows_z2()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Path to directory containing configuration files', required=True)
    parser.add_argument('--parse-type', type=str, help='What system configurations are you parsing', required=True)
    parser.add_argument('--destination', type=str, help='Path to save transformed file', required=True)
    parser.add_argument('--truncate', type=int, help="Number of files to process if you don't want all", default=0)

    args = parser.parse_args()

    allowed_parse_type_list = ['owen_z3', 'as_rows_z2']
    if args.parse_type not in allowed_parse_type_list:
        raise ValueError("Don't know how to parse this type of file")
    PARSE_TYPE = args.parse_type

    if not os.path.exists(args.src):
        raise ValueError('Invalid src destination')

    SRC = args.src
    DESTINATION = args.destination
    TRUNCATE = args.truncate

    main()

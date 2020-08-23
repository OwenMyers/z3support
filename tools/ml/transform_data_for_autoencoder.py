import os
import argparse
import pandas as pd
import tqdm
import numpy as np
import logging


def string_to_number_directions_1(r, column):
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
    df_copy = df_in.copy()
    df_copy['n_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'N'), axis=1)
    df_copy['e_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'E'), axis=1)
    df_copy['s_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'S'), axis=1)
    df_copy['w_number'] = df_copy.apply(lambda r: string_to_number_directions_2(r, 'W'), axis=1)
    return df_copy


def determine_lattice_size(df_in):
    # Assume square lattice
    max_x = df_in['x'].values.max()
    max_y = df_in['y'].values.max()
    if max_x != max_y:
        raise ValueError('Expecting x==y dimensions')
    return max_x + 1


def check_if_exists(cur_val, proposed_val, v=False):
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

    Arguments:
        df_in (DataFrame): is the dataframe of a plaquette representation of a configuration for which you have run
            the ``string_to_number_directions`` on.
        lattice_size (int): Lattice length or width (assumed square).
        v (bool): Verbose or not.

    Returns:
        A numpy matrix with zeros representing the vertices and centers of plaquetts and the number system as described
        by ``string_to_number_directions``
        representing the links.
    """
    df_working = df_in.copy()

    # l = determine_lattice_size(df_working)
    # Will return this matrix
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

            cur_transformed_matrix[transformed_horizontal_index_x, transformed_horizontal_index_y] = \
                row[raw_horizontal_index]
            cur_transformed_matrix[transformed_vertical_index_x, transformed_vertical_index_y] = \
                row[raw_vertical_index]

            count += 2
    return cur_transformed_matrix


@final_output
def parse_as_rows_z2():
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

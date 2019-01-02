"""
Plot some value like the link density as a function of the link weighting parameter.
In the Rust code this parameter is the "link_number_tuning" parameter.
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from colors import color_list
from z3support.data_tools import variance


def check_and_return_array(array_or_list_in):
    if isinstance(array_or_list_in, list):
        a = np.array(array_or_list_in)
    else:
        assert isinstance(array_or_list_in, np.ndarray)
        a = array_or_list_in
    return a


def random_population(array_or_list_in):
    a = check_and_return_array(array_or_list_in)
    sample = np.random.choice(a, len(a), replace=True)
    return sample


def bootstrap(array_or_list_in, n, value_type='mean'):
    """
    Randomly sample from array_or_list_in n times.
    
    Args:
        array_or_list_in (np.ndarray, list): The original sample to bootstrap over.
        n (int):
    """
    a = check_and_return_array(array_or_list_in)
    values = []
    for i in range(n):
        sample = random_population(a)
        if value_type == 'median':
            cur_value = np.median(sample)
        elif value_type == 'mean':
            cur_value = np.mean(sample)
        values.append(cur_value)
    return np.array(values)


def main():

    root_dir = os.path.join('data', 'variable_link_weights')
    lattice_size_list = os.listdir(root_dir)

    for i, cur_lattice_size_dir in enumerate(lattice_size_list):
        if 'lattice_size' not in cur_lattice_size_dir:
            continue
        working_dir = os.path.join('data', 'variable_link_weights', cur_lattice_size_dir)
        print('working directory {}'.format(working_dir))

        split_num = cur_lattice_size_dir.split('_')
        dir_list = os.listdir(working_dir)

        weights = []
        variances = []
        errs = []
        for cur_weight_dir in dir_list:
            if 'weight' not in cur_weight_dir:
                continue

            cur_weight_dir_full_path = os.path.join(working_dir, cur_weight_dir)
            print("Current working path: {}".format(cur_weight_dir_full_path))

            weight = float(cur_weight_dir.split('_')[1].replace('pt', '.'))
            weights.append(np.sqrt(weight))
            #weights.append(weight)

            winding_number_path = os.path.join(
                cur_weight_dir_full_path,
                'winding_number_variance_estimator.csv'
            )

            df = pd.read_csv(winding_number_path)

            horz_varience_list = df['Horizontal'].tolist()
            horz_varience = bootstrap(horz_varience_list, 100, value_type='mean')

            variances.append(horz_varience.mean()/float(split_num[-1]))
            print('cur var {}'.format(horz_varience.mean()))
            errs.append(horz_varience.std())
            #errs.append(np.array(horz_varience_list).std()/np.sqrt(float(len(horz_varience_list))))
            print('cur err {}'.format(horz_varience.std()))

        if len(weights) == 0:
            continue
        weights, variances = zip(*sorted(zip(weights, variances)))

        print('plotting for lattice size {}'.format(split_num))
        plt.errorbar(
            weights,
            variances,
            errs,
            capsize=2,
            marker='o',
            markersize=3,
            color=color_list[i],
            label=r'$' + split_num[-1] + r'\times' + split_num[-2] + r'$'
        )

    plt.xlabel(r'$\sqrt{\alpha}$')
    plt.ylabel(r'$\rho_s$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
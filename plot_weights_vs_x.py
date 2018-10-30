"""
Plot some value like the link density as a function of the link weighting parameter.
In the Rust code this parameter is the "link_number_tuning" parameter.
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def main():
    lattice_size_dir = 'lattice_size_4_4'
    working_dir = os.path.join('data', 'variable_link_weights', 'lattice_size_4_4')
    split_num = lattice_size_dir.split('_')
    total_number_links =  int(split_num[-1]) * int(split_num[-2]) * 2
    dir_list = os.listdir(working_dir)

    weights = []
    densities = []
    errors = []
    for cur_weight_dir in dir_list:
        if 'weight' not in cur_weight_dir:
            continue

        cur_weight_dir_full_path = os.path.join(working_dir, cur_weight_dir)
        print("Current working path: {}".format(cur_weight_dir_full_path))

        weight = float(cur_weight_dir.split('_')[1].replace('pt', '.'))
        weights.append(weight)

        total_counts_path = os.path.join(cur_weight_dir_full_path, 'total_link_count_estimator.csv')

        df = pd.read_csv(total_counts_path)
        average_count = df['Average Total Link Counts'].mean()
        error = df['Average Total Link Counts'].std()/np.sqrt(len(df))
        density = average_count / float(total_number_links)

        densities.append(density)
        errors.append(error)

        print("avg link count: {}".format(average_count))
        print("error: {}".format(error))

    weights, densities, errors = zip(*sorted(zip(weights, densities, errors)))

    plt.errorbar(weights, densities, errors, capsize=2)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Link Density')
    plt.show()


if __name__ == "__main__":
    main()
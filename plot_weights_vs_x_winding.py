"""
Plot some value like the link density as a function of the link weighting parameter.
In the Rust code this parameter is the "link_number_tuning" parameter.
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from colors import color_list


def main():

    root_dir = os.path.join('data', 'variable_link_weights')
    lattice_size_list = os.listdir(root_dir)

    for i, cur_lattice_size_dir in enumerate(lattice_size_list):
        if 'lattice_size' not in cur_lattice_size_dir:
            continue
        working_dir = os.path.join('data', 'variable_link_weights', cur_lattice_size_dir)
        split_num = cur_lattice_size_dir.split('_')
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
            winding_number_path = os.path.join(
                cur_weight_dir_full_path,
                'winding_number_count_estimator.csv'
            )

            df = pd.read_csv(total_counts_path)
            average_count = df['Average Total Link Counts'].mean()
            error = df['Average Total Link Counts'].std()/np.sqrt(len(df))
            density = average_count / float(total_number_links)

            densities.append(density)
            errors.append(error)

            print("avg link count: {}".format(average_count))
            print("error: {}".format(error))

        weights, densities, errors = zip(*sorted(zip(weights, densities, errors)))

        plt.errorbar(
            weights,
            densities,
            errors,
            capsize=2,
            color=color_list[i],
            label=r'$' + split_num[-1] + r'\times' + split_num[-2] + r'$'
        )

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Link Density')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
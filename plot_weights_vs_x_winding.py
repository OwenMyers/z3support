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


def main():

    root_dir = os.path.join('data', 'variable_link_weights')
    lattice_size_list = os.listdir(root_dir)

    for i, cur_lattice_size_dir in enumerate(lattice_size_list):
        if 'lattice_size' not in cur_lattice_size_dir:
            continue
        working_dir = os.path.join('data', 'variable_link_weights', cur_lattice_size_dir)
        split_num = cur_lattice_size_dir.split('_')
        dir_list = os.listdir(working_dir)

        weights = []
        variances = []
        for cur_weight_dir in dir_list:
            if 'weight' not in cur_weight_dir:
                continue

            cur_weight_dir_full_path = os.path.join(working_dir, cur_weight_dir)
            print("Current working path: {}".format(cur_weight_dir_full_path))

            weight = float(cur_weight_dir.split('_')[1].replace('pt', '.'))
            weights.append(weight)

            winding_number_path = os.path.join(
                cur_weight_dir_full_path,
                'winding_number_count_estimator.csv'
            )

            df = pd.read_csv(winding_number_path)
            average_horz_winding = df['Horizontal'].mean()
            average_vert_winding = df['Vertical'].mean()
            print('average_horz_winding {}'.format(average_horz_winding))
            print('average_vert_winding {}'.format(average_vert_winding))
            horz_varience = variance(np.array(df['Horizontal']))
            vert_varience = variance(np.array(df['Vertical']))
            print('horz_varience {}'.format(horz_varience))
            print('vert_varience {}'.format(vert_varience))

            variances.append(vert_varience)

        if len(weights) == 0:
            continue
        weights, variances = zip(*sorted(zip(weights, variances)))

        plt.plot(
            weights,
            variances,
            color=color_list[i],
            label=r'$' + split_num[-1] + r'\times' + split_num[-2] + r'$'
        )

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Winding Number Varience')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
import argparse
from z3support.datamodel.general_info import GeneralInformation
from z3support.datamodel.vertex import Vertex
from z3support.datamodel.point import Point
import csv
from plot_lattices import draw_lattice_backround
from plot_lattices import LINK_LENGTH
from plot_lattices import adjusted_figure
from matplotlib.collections import PatchCollection
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# plt.style.use('aps')
# pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
# matplotlib.rcParams.update(pgf_with_rc_fonts)


def get_color_out_list(file_and_path):
    color_out_list = []
    with open(file_and_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            location = Point(int(row['x']),
                             int(row['y']))
            n, e, s, w = int(row['N']), int(row['E']), int(row['S']), int(row['W'])
            if n == 1:
                color_out_list.append((location, 'N'))
            if e == 1:
                color_out_list.append((location, 'E'))
            if s == 1:
                color_out_list.append((location, 'S'))
            if w == 1:
                color_out_list.append((location, 'W'))

    return color_out_list


def remove_colored_out(
                color_out_list,
                rectangle_patches,
                estimator_values,
                locations,
                directions):

    new_rectangle_patches = []
    new_estimator_values = []

    for i in range(len(rectangle_patches)):
        for j, color_out_tuple in enumerate(color_out_list):
            cur_color_out_location = color_out_tuple[0]
            cur_color_out_direction = color_out_tuple[1]

            if (cur_color_out_location.x == locations[i].x) and (cur_color_out_location.y == locations[i].y):
                if cur_color_out_direction == directions[i]:
                    continue

            new_rectangle_patches.append(rectangle_patches[i])
            new_estimator_values.append(estimator_values[i])
    return new_rectangle_patches, new_estimator_values


def get_colored_out_collection(color_out_list, general_information):

    patches = []
    for color_out_tuple in color_out_list:
        cur_color_out_location = color_out_tuple[0]
        cur_color_out_direction = color_out_tuple[1]
        tmp_value_dict = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
        tmp_value_dict[cur_color_out_direction] = 1

        for_vertex_info_dict = {'x': cur_color_out_location.x,
                                'y': cur_color_out_location.y,
                                'N': tmp_value_dict['N'],
                                'E': tmp_value_dict['E'],
                                'S': tmp_value_dict['S'],
                                'W': tmp_value_dict['W']
                                }

        vertex = Vertex(general_information.system_size)
        vertex.subtract_off_from_link(LINK_LENGTH * 0.15)
        vertex.estimator_fill_from_csv_row(for_vertex_info_dict)
        vertex.make_patches_to_plot(LINK_LENGTH, link_width_factor=0.15)

        for i, cur_patch in enumerate(vertex.rect_patches):
            if vertex.directions[i] == cur_color_out_direction:
                if vertex.values[i] > 0.5:
                    patches.append(cur_patch)

    collection = PatchCollection(patches)
    collection.set_color('Orange')
    return collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="File with averaged estimator")
    parser.add_argument("--blot", required=False, default=None,
                        help="Can specify `h` or `v` to blot out horizontal or vertical links respectively")
    parser.add_argument("--color-out", required=False, default=None,
                        help="Specify a file containing a list of links to color out and remove from heat map.")

    args = parser.parse_args()
    blot_out = args.blot
    color_out = args.color_out
    if color_out is not None:
        color_out_list = get_color_out_list(color_out)
    file_and_path_string = args.f
    assert 'data' in file_and_path_string

    general_information = GeneralInformation.from_file_path(file_and_path_string)
    assert general_information.system_size.x == general_information.system_size.y, \
        "Can't handel rectangular lattices right now."

    fig, ax = adjusted_figure()
    draw_lattice_backround(general_information.system_size.x, ax)

    rectangle_patches = []
    estimator_values = []
    # Point objects
    locations = []
    # Strings "N","E","S","W"
    directions = []

    with open(file_and_path_string, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            vertex = Vertex(general_information.system_size)
            # makes the links smaller
            vertex.subtract_off_from_link(LINK_LENGTH * 0.15)
            if blot_out == "h":
                vertex.ignore_horizontal_links = True
            if blot_out == "v":
                vertex.ignore_vertical_links = True
            vertex.estimator_fill_from_csv_row(row)
            vertex.make_patches_to_plot(LINK_LENGTH, link_width_factor=0.15)

            rectangle_patches += vertex.rect_patches
            estimator_values += vertex.values
            locations += [vertex.int_location] * len(vertex.rect_patches)
            directions += vertex.directions

    assert len(rectangle_patches) == len(estimator_values)
    assert len(rectangle_patches) == len(locations)
    assert len(rectangle_patches) == len(directions)

    if color_out is not None:
        rectangle_patches, estimator_values = remove_colored_out(
                color_out_list,
                rectangle_patches,
                estimator_values,
                locations,
                directions,
        )

    collection = PatchCollection(rectangle_patches)
    # collection.set_color('grey')estimator_values_as_array
    estimator_values_as_array = np.array(estimator_values)
    collection.set_array(estimator_values_as_array)
    collection.set_cmap('Blues')
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax)

    if color_out is not None:
        colored_out_collection = get_colored_out_collection(color_out_list, general_information)
        ax.add_collection(colored_out_collection)

    plt.axis('equal')
    plt.axis('off')

    ax.set_xlim([-0.5, float(general_information.system_size.x) - 0.5])
    ax.set_ylim([-0.5, float(general_information.system_size.x) - 0.5])

    # plt.show()
    file_path, file_name = os.path.split(file_and_path_string)
    full_fig_name = os.path.join('figures',
                                 'estimators',
                                 file_name.split('.')[0] + '.png')
    plt.savefig(full_fig_name, dpi=300)
    plt.close(fig)

    print("Done")


if __name__ == '__main__':
    main()

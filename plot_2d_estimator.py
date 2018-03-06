import argparse
from z3support.datamodel.general_info import GeneralInformation
from z3support.datamodel.vertex import Vertex
import csv
import matplotlib.pyplot as plt
import matplotlib
from .plot_lattices import draw_lattice_backround
from .plot_lattices import LINK_LENGTH
from .plot_lattices import L
from .plot_lattices import adjusted_figure
from matplotlib.collections import PatchCollection
import os

plt.style.use('aps')
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, help="File with averaged estimator")
    args = parser.parse_args()

    file_and_path_string = args.f
    assert 'data' in file_and_path_string

    general_information = GeneralInformation.from_file_path(file_and_path_string)

    fig, ax = adjusted_figure()
    draw_lattice_backround(ax)

    rectangle_patches = []
    estimator_values = []
    with open(file_and_path_string, "r") as f:
        reader = csv.DictReader(f)

        for row in reader():
            vertex = Vertex(general_information.system_size)
            vertex.estimator_fill_from_csv_row(row)
            vertex.make_patches_to_plot(LINK_LENGTH, link_width_factor=0.15)

            rectangle_patches += vertex.rect_patches
            estimator_values += vertex.values

    collection = PatchCollection(rectangle_patches)
    collection.set_color('grey')
    ax.add_collection(collection)

    plt.axis('equal')
    plt.axis('off')

    ax.set_xlim([-0.5, float(L) - 0.5])
    ax.set_ylim([-0.5, float(L) - 0.5])

    #plt.show()
    file_path, file_name = os.path.split(file_and_path_string)
    full_fig_name = os.path.join('figures',
                                 'estimators',
                                 file_name.split('.')[0] + '.png')
    plt.savefig(full_fig_name, dpi=300)
    plt.close(fig)

    print("Done")


if __name__ == '__main__':
    main()


from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from z3support.datamodel.vertex import Vertex
from z3support.datamodel.point import Point
import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#plt.style.use('aps')
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)


L = 6
LINE_SIZE = 0.05
# link length 
LINK_LENGTH = 1.0

# Two types of files
# 1) a file for each configuration; a row for each vertex
# 2) all in one file; a row for each configuration

FILE_TYPE = 2

def adjusted_figure():
    fig = plt.figure()
    #fig.subplots_adjust(bottom=0.14,left=0.135,right=0.98,top=0.97)
    fig.subplots_adjust(bottom=0.05, left=0.05, right=0.99, top=0.95)
    ax = fig.add_subplot(111)

    return fig, ax


def draw_lattice_backround(square_lattice_size, ax):
    patches = []

    # horizontal lines
    for i in range(square_lattice_size):
        loc = (-LINK_LENGTH/2.0, float(i) - LINE_SIZE/2.0)
        cur_horz_rect = Rectangle(loc, square_lattice_size, LINE_SIZE, color='k')
        patches.append(cur_horz_rect) 
    # vertical lines
    for i in range(square_lattice_size):
        loc = (i - LINE_SIZE/2.0, -LINK_LENGTH/2.0)
        cur_vert_rect = Rectangle(loc, LINE_SIZE, square_lattice_size, color='k')
        patches.append(cur_vert_rect) 

    collection = PatchCollection(patches)
    collection.set_color('k')
    ax.add_collection(collection)


def plot_test_points(ax, lat_size):
    rect_patches = []
    tri_patches = []

    location = Point(0, 1)
    vertex = Vertex(lat_size)
    vertex.location = location
    vertex.N = 'Out'
    vertex.E = 'Out'
    vertex.S = 'Out'
    vertex.W = 'Out'
    vertex.make_patches_to_plot(LINK_LENGTH)

    rect_patches += vertex.rect_patches
    tri_patches += vertex.tri_patches

    location = Point(3, 3)
    vertex = Vertex(lat_size)
    vertex.location = location
    vertex.N = 'In'
    vertex.E = 'In'
    vertex.S = 'In'
    vertex.W = 'In'
    vertex.make_patches_to_plot(LINK_LENGTH)

    rect_patches += vertex.rect_patches
    tri_patches += vertex.tri_patches
    
    collection = PatchCollection(rect_patches)
    collection.set_color('grey')
    ax.add_collection(collection)

    draw_lattice_backround(L, ax)

    collection = PatchCollection(tri_patches)
    collection.set_color('black')
    ax.add_collection(collection)

def list_of_vertices_on_sublattice(l, shaped_array):
    lat_size = Point(l, l)
    vertex_list = []
    for i in range(l):
        for j in range(l):
            # only make a vertex for one sublattice
            if (i + j) % 2 == 0:
                cur_v = Vertex(lat_size)

def main():
    lat_size = Point(L, L)

    testing = False

    lattice_files = os.listdir('lattices')

    for cur_f in lattice_files:
        rect_patches = []
        tri_patches = []
        if not (('.csv' in cur_f) or ('.txt' in cur_f)):
            continue

        fig, ax = adjusted_figure()

        if testing:
            plot_test_points(ax, lat_size)
            break

        full_f_name = os.path.join('lattices', cur_f)

        with open(full_f_name, 'r') as csv_file:

            if FILE_TYPE == 1:
                reader = csv.DictReader(csv_file)

                for row in reader:

                    vertex = Vertex(lat_size)
                    vertex.fill_from_csv_row(row)
                    vertex.make_patches_to_plot(LINK_LENGTH, link_width_factor=0.15)

                    rect_patches += vertex.rect_patches
                    tri_patches += vertex.tri_patches
            elif FILE_TYPE == 2:
                config_array = np.genfromtxt(csv_file, delimiter=' ')
                print(config_array)
                

            collection = PatchCollection(rect_patches)
            collection.set_color('grey')
            ax.add_collection(collection)

            collection = PatchCollection(tri_patches)
            collection.set_color('black')
            ax.add_collection(collection)

        draw_lattice_backround(L, ax)

        plt.axis('equal')
        plt.axis('off')
        #ax.set_aspect(1.0)


        ax.set_xlim([-0.5, float(L) - 0.5])
        ax.set_ylim([-0.5, float(L) - 0.5])

        #plt.show()
        if not os.path.exists('figures'):
            os.mkdir('figures')
        if not os.path.exists(os.path.join('figures', 'lattices')):
            os.mkdir(os.path.join('figures', 'lattices'))

        full_fig_name = os.path.join('figures', 'lattices', cur_f.split('.')[0] + '.png')
        plt.savefig(full_fig_name, dpi=300)
        plt.close(fig)

    if testing:
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    main()

import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from z3support.datamodel.vertex import Vertex
from z3support.datamodel.point import Point
import os
import csv

L = 4
LINE_SIZE = 0.05
# link length 
LINK_LENGTH = 1.0

def plot_vertex(x, y, row, ax):
    vertex = Vertex(x, y)
    vertex.fill_from_csv_row(row)
    vertex.get_patches_to_plot(LINK_LENGTH)

def draw_lattice_backround(ax):
    patches = []

    # horizontal lines
    for i in range(L):
        loc = (-LINK_LENGTH/2.0,float(i) - LINE_SIZE/2.0)
        cur_horz_rect = Rectangle(loc, L, LINE_SIZE, color='k')
        patches.append(cur_horz_rect) 
    # vertical lines
    for i in range(L):
        loc = (i - LINE_SIZE/2.0, -LINK_LENGTH/2.0)
        cur_vert_rect = Rectangle(loc, LINE_SIZE, L, color='k')
        patches.append(cur_vert_rect) 


    collection = PatchCollection(patches)
    collection.set_color('k')
    ax.add_collection(collection)

def plot_test_points(ax, lat_size):
    rect_patches = []
    tri_patches = []

    location = Point(1, 1)
    vert = Vertex(location, lat_size)
    vert.N = 'In'
    vert.E = 'In'
    vert.S = 'In'
    vert.W = 'In'
    vert.make_patches_to_plot(LINK_LENGTH)

    rect_patches += vert.rect_patches
    tri_patches += vert.tri_patches

    location = Point(2, 2)
    vert = Vertex(location, lat_size)
    vert.N = 'Out'
    vert.E = 'Out'
    vert.S = 'Out'
    vert.W = 'Out'
    vert.make_patches_to_plot(LINK_LENGTH)

    rect_patches += vert.rect_patches
    tri_patches += vert.tri_patches
    
    collection = PatchCollection(rect_patches)
    collection.set_color('grey')
    ax.add_collection(collection)

    draw_lattice_backround(ax)

    collection = PatchCollection(tri_patches)
    collection.set_color('black')
    ax.add_collection(collection)


def main():
    lat_size = Point(4, 4)
    
    rect_patches = []
    tri_patches = []

    lattice_files = os.listdir('lattices')

    for cur_f in lattice_files:
        if '.csv' not in cur_f:
            continue
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        full_f_name = os.path.join('lattices', cur_f)

        with open(full_f_name, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for row in reader:
            
                vert = Vertex(lat_size)
                vert.fill_from_csv_row(row)
                vert.make_patches_to_plot(LINK_LENGTH)

                rect_patches += vert.rect_patches
                tri_patches += vert.tri_patches

        collection = PatchCollection(rect_patches)
        collection.set_color('grey')
        ax.add_collection(collection)

        draw_lattice_backround(ax)

        collection = PatchCollection(tri_patches)
        collection.set_color('black')
        ax.add_collection(collection)


        plt.axis('equal')
        #plt.axis('off')
        #plt.tight_layout()

        #plt.show()
        full_fig_name = os.path.join('figures', 'lattices', cur_f.split('.')[0] + '.png')
        plt.savefig(full_fig_name )

if __name__ == "__main__":
    main()

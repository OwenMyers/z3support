
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from datamodel.vertex import Vertex

L = 4
LINE_SIZE = 0.05
# link length 
LL = 1.0

def plot_vertex(x, y, row, ax):
    vertex = Vertex(x, y)
    vertex.fill_from_csv_row(row)
    vertex.get_patches_to_plot(LL)

def draw_lattice_backround(ax):
    patches = []

    for i in range(L):
        loc = (-LL/2.0,i)
        cur_horz_rect = Rectangle(loc, L, LINE_SIZE, color='k')
        patches.append(cur_horz_rect) 
    for i in range(L):
        loc = (i,-LL/2.0)
        cur_horz_rect = Rectangle(loc, LINE_SIZE, L, color='k')
        patches.append(cur_horz_rect) 


    collection = PatchCollection(patches)

    ax.add_collection(collection)


def main():
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    draw_lattice_backround(ax)

    plt.axis('equal')
    #plt.axis('off')
    #plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

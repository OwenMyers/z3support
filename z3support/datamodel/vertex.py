import matplotlib.patches as mpatches
from .point import Point
import numpy as np

class Vertex:
    def __init__(self, location, size):
        """
        :param location: The cartesian location of the verx.
        :type location: Point (see datamodel).
        :param size: The size (width, height) of the lattic.
        :type size: Point (see datamodel).
        """
        self.location = location
        self.size = size

        self.N = None
        self.E = None
        self.S = None
        self.W = None

        # rectangular patces list
        self.rect_patches = []
        # triangles which show the direction of links
        self.tri_patches = []

    def fill_from_csv_row(self, row):
        self.N = row['N']
        self.E = row['E']
        self.S = row['S']
        self.W = row['W']

    def full_north_patch(self, link_length, link_width):
        if self.N == 'Blank':
            return
        elif self.N == 'Out':
            orientation = 0.0
        elif self.N == 'In':
            orientation = np.pi
        # Triangle stuff
        center_x = self.location.x
        center_y = self.location.y + link_length/2.0
        triangle = mpatches.RegularPolygon((center_x,center_y), 3, radius=link_width/2.0, orientation=orientation)

        # Rectangle stuff
        lower_left_x = self.location.x - link_width/2.0
        lower_left_y = self.location.y
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_width, link_length)

        self.rect_patches.append(p)
        self.tri_patches.append(triangle)

    def half_north_patch(self, link_length, link_width):
        if self.N == 'Blank':
            return
        elif self.N == 'Out':
            orientation = 0.0
        elif self.N == 'In':
            orientation = np.pi
        # Triangle stuff
        center_x = self.location.x
        center_y = self.location.y + link_length/2.0
        triangle = mpatches.RegularPolygon((center_x,center_y), 3, radius=link_width/2.0, orientation=orientation)

        # Rectangle stuff
        lower_left_x = self.location.x - link_width/2.0
        lower_left_y = self.location.y
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_width, link_length/2.0)

        self.rect_patches.append(p)
        self.tri_patches.append(triangle)
 
    def full_south_patch(self, link_length, link_width):
        if self.S == 'Blank':
            return
        lower_left_x = self.location.x - link_width/2.0
        lower_left_y = self.location.y - link_length
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_width, link_length)
        self.rect_patches.append(p)

    def half_south_patch(self, link_length, link_width):
        if self.S == 'Blank':
            return
        lower_left_x = self.location.x - link_width/2.0
        lower_left_y = self.location.y - link_length/2.0
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_width, link_length/2.0)
        self.rect_patches.append(p)

    def full_east_patch(self, link_length, link_width):
        if self.E == 'Blank':
            return
        lower_left_x = self.location.x 
        lower_left_y = self.location.y - link_width/2.0
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_length, link_width)
        self.rect_patches.append(p)

    def half_east_patch(self, link_length, link_width):
        if self.E == 'Blank':
            return
        lower_left_x = self.location.x
        lower_left_y = self.location.y - link_width/2.0
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_length/2.0, link_width)
        self.rect_patches.append(p)

    def full_west_patch(self, link_length, link_width):
        if self.W == 'Blank':
            return
        lower_left_x = self.location.x - link_length
        lower_left_y = self.location.y - link_width/2.0
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_length, link_width)
        self.rect_patches.append(p)

    def half_west_patch(self, link_length, link_width):
        if self.W == 'Blank':
            return
        lower_left_x = self.location.x - link_length/2.0
        lower_left_y = self.location.y - link_width/2.0
        p = rounded_rect_patch(lower_left_x, lower_left_y, link_length/2.0, link_width)
        self.rect_patches.append(p)


    def make_patches_to_plot(self, link_length): 
        link_width = float(link_length) * 0.2

        if self.vertex_at_boundry_question() == "not at boundary":
            self.full_north_patch(link_length, link_width) 
            self.full_south_patch(link_length, link_width) 
            self.full_east_patch(link_length, link_width) 
            self.full_west_patch(link_length, link_width) 

        if self.vertex_at_boundry_question() == "down left":
            self.full_north_patch(link_length, link_width)
            self.half_south_patch(link_length, link_width)
            self.full_east_patch(link_length, link_width)
            self.half_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "up left":
            self.half_north_patch(link_length, link_width)
            self.full_south_patch(link_length, link_width)
            self.full_east_patch(link_length, link_width)
            self.half_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "up right":
            self.half_north_patch(link_length, link_width)
            self.full_south_patch(link_length, link_width)
            self.half_east_patch(link_length, link_width)
            self.full_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "down right":
            self.full_north_patch(link_length, link_width)
            self.half_south_patch(link_length, link_width)
            self.half_east_patch(link_length, link_width)
            self.full_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "left":
            self.full_north_patch(link_length, link_width)
            self.full_south_patch(link_length, link_width)
            self.full_east_patch(link_length, link_width)
            self.half_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "right":
            self.full_north_patch(link_length, link_width)
            self.full_south_patch(link_length, link_width)
            self.half_east_patch(link_length, link_width)
            self.full_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "down":
            self.full_north_patch(link_length, link_width)
            self.half_south_patch(link_length, link_width)
            self.full_east_patch(link_length, link_width)
            self.full_west_patch(link_length, link_width)

        if self.vertex_at_boundry_question() == "up":
            self.half_north_patch(link_length, link_width)
            self.full_south_patch(link_length, link_width)
            self.full_east_patch(link_length, link_width)
            self.full_west_patch(link_length, link_width)

    def vertex_at_boundry_question(self):
        """
        This function really just makes other parts of the code more readable.
        """
    
        if (self.location.x == 0) and (self.location.y == 0):
            return "down left"
        elif (self.location.x == 0) and (self.location.y == (self.size.y-1)):
            return "up left"
        elif (self.location.x == (self.size.x-1)) and (self.location.y == (self.size.y-1)):
            return "up right"
        elif (self.location.x == (self.size.x-1)) and (self.location.y == 0):
            return "down right"
        elif self.location.x == 0:
            return "left"
        elif self.location.x == (self.size.x - 1):
            return "right"
        elif self.location.y == 0:
            return "down"
        elif self.location.y == (self.size.y - 1):
            return "up"
        else:
            return "not at boundary"

def rounded_rect_patch(x, y, width, height):
    fancybox = mpatches.FancyBboxPatch(
          (x, y), width, height,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02))
    
    return fancybox
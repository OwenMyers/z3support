import matplotlib.patches as mpatches
from point import Point

class Vertex:
    def __init__(self, x, y):
        # The location of the center of the vertex
        self.location = Point(x, y)

        self.N = None
        self.E = None
        self.S = None
        self.W = None

    def fill_from_csv_row(row):
        self.N = row['N']
        self.E = row['E']
        self.S = row['S']
        self.W = row['W']

    def get_patches_to_plot(link_length): 
        link_width = float(link_length) * 0.2

        patches = []

        # North link 
        if self.N != 'Blank':
            lower_left_x = self.location.x - link_width/2.0
            lower_left_y = self.location.y
            p = rounded_rect_patch(lower_left_x, 
                                   lower_left_y, 
                                   link_width, 
                                   link_length)
            patches.append(p)

        # South link
        if self.S != 'Blank':
            lower_left_x = self.location.x - link_width/2.0
            lower_left_y = self.location.y - link_length
            p = rounded_rect_patch(lower_left_x, 
                                   lower_left_y, 
                                   link_width, 
                                   link_length)

        # East link
        if self.E != 'Blank':
            lower_left_x = self.location.x - link_length
            lower_left_y = self.location.y - link_width/2.0
            p = rounded_rect_patch(lower_left_x, 
                                   lower_left_y, 
                                   link_length, 
                                   link_width)

        # East link
        if self.E != 'Blank':
            lower_left_x = self.location.x
            lower_left_y = self.location.y - link_width/2.0
            p = rounded_rect_patch(lower_left_x, 
                                   lower_left_y, 
                                   link_length, 
                                   link_width)


def rounded_rect_patch(x, y, width, height):
    fancybox = mpatches.FancyBboxPatch(
          (x, y), width, height,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02))
    
    return fancybox
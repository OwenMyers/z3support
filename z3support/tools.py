

def vertex_at_boundry_question(location, size):
    """
    :param location: The cartesian location of the verx.
    :type location: Point (see datamodel).
    :param size: The size (width, height) of the lattic.
    :type size: Point (see datamodel).
    """

    if (location.x != 0) and (location.y != 0):
        return "not at boundary"
    elif (location.x == 0) and (location.y == 0):
        return "down left"
    elif (location.x == 0) and (location.y == (size.y-1)):
        return "up left"
    elif (location.x == (size.x-1)) and (location.y == (size.y-1)):
        return "up right"
    elif (location.x == (size.x-1)) and (location.y == 0):
        return "down right"
    elif location.x == 0:
        return "left"
    elif location.x == (size.x - 1):
        return "right"
    elif location.y == 0:
        return "down"
    elif location.y == (size.y - 1):
        return "up"
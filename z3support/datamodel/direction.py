
def flip_inout(direction):

    if direction == 'Blank':
        return 'Blank'
    if direction == 'In':
        return 'Out'
    if direction == 'Out':
        return 'In'
    if direction == "no orientation":
        return "no orientation"

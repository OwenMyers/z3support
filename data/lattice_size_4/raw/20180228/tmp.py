
import numpy as np

EPSILON = 1e-5

def find_lattice_size_from_data(data):

    counting_lines = 0

    for i, line in enumerate(data):
        if i == 0:
            continue

        counting_lines += 1
        
        if (line[0] < EPSILON) and (line[1] < EPSILON):
            return counting_lines

    raise Exception("Looks like, at best, there is only one bin") 
        
def check_data(data):
    assert data[0][0] < EPSILON
    assert data[0][1] < EPSILON
    assert len(data[0]) == 6


def main():
    f = open("density_estimator.csv", "r")
    header_line = f.readline()
    data = np.genfromtxt(f, delimiter=',')

    check_data(data)

    lattice_size = find_lattice_size_from_data(data)
    print("Lattice size", lattice_size)

    data = data.reshape(-1, lattice_size, 6)

    for d in data:
        print(d)

if __name__ == main():
    main()
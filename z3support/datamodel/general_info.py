from .point import Point
import os
from datetime import datetime


class GeneralInformation():
    def __init__(self, system_size, file_and_path, date):
        self.system_size = Point(0, 0)
        self.file_and_path = ''
        self.date = None

    @classmethod
    def from_file_path(cls, file_and_path):
        path_to_file, file_name = os.path.split(file_and_path)
        raw_path, date = os.path.split(path_to_file)
        lattice_size_path, raw = os.path.split(raw_path)
        data_path, lattice_size = os.path.split(lattice_size_path)

        date = datetime.strptime(date, '%Y-%m-%d')
        lattice_size_y = int(lattice_size.split('_')[-1])
        lattice_size_x = int(lattice_size.split('_')[-2])
        system_size = Point(lattice_size_x, lattice_size_y)

        return cls(system_size, file_and_path, date)


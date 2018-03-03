from .point import Point
import os
from datetime import datetime

class GeneralInformation:

    def __init__(self,
                 system_size,
                 file_and_path,
                 date,
                 lattice_size_path,
                 lattice_size):
        """
        :param system_size:
        :type system_size: Point object
        :param file_and_path:
        :type file_and_path: string
        :param date:
        :type date: datetime datetime
        :param lattice_size_path:
        :type lattice_size_path: string
        """

        self.system_size = system_size
        self.file_and_path = file_and_path
        self.date = date
        self.lattice_size_path = lattice_size_path

    def date_as_string(self):
        return self.date.strftime("%Y-%m-%d_%H:%M:%S")

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

        return cls(system_size,
                   file_and_path,
                   date,
                   lattice_size_path)


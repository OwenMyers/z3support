from unittest import TestCase
import os
from tools.ml import grid_search as gs
import numpy as np


class Test(TestCase):
    def test_import_data_shuffle(self):
        z2_file_name = '/tmp/tmp_z2.npy'
        z3_file_name = '/tmp/tmp_z3.npy'
        # make fake data sets. Fake output of transform data for autoencoder
        z2_array = np.array([[[0, 1, 0, 1], [0, 2, 0, 2], [1, 0, 1, 0], [2, 0, 2, 0]]])
        z3_array = np.array([[[1, 0, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]])

        np.save(z2_file_name, z2_array)
        np.save(z3_file_name, z3_array)

        list_data = [z2_file_name, z3_file_name]
        indices = (0, 1)
        normed_original_data = [z2_array/2.0, z3_array/2.0]
        c = 0
        while indices == (0, 1):
            balanced_and_shuffled, indices = gs.import_data(list_data)
            c += 1
            if c > 1000:
                self.fail("Problem with shuffle")

        for i, current_row in enumerate(balanced_and_shuffled):
            for j, current_check in enumerate(current_row):
                for k, current_check_2 in enumerate(current_check):
                    self.assertListEqual(list(current_check_2), list(normed_original_data[indices[i]][j, k]))

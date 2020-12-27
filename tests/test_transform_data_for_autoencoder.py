from unittest import TestCase
import numpy as np
from tools.ml.src.transform_data_for_autoencoder import z2_row_to_matrix


class Test(TestCase):
    def test_z2_row_to_matrix(self):
        matrix = np.array([
            np.array([0.0, 1.0, 0.0, 2.0]),
            np.array([3.0, 0.0, 4.0, 0.0]),
            np.array([0.0, 5.0, 0.0, 6.0]),
            np.array([7.0, 0.0, 8.0, 0.0]),
        ])
        matrix_as_row = np.array([1, 3, 2, 4, 5, 7, 6, 8])
        out = z2_row_to_matrix(matrix_as_row, 2)

        self.assertIsNone(np.testing.assert_array_almost_equal(out, matrix))

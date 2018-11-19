from numba import jitclass
from numba.types import int32, float64
from .. import common_utils


@jitclass([('_element_values', float64[:]),
        ('_element_column_indices', int32[:]),
        ('_row_first_element_indices', int32[:])])
class CSRMatrix:

    def __init__(self, element_values, element_column_indices, row_first_element_indices):

        self._element_values = element_values
        self._element_column_indices = element_column_indices
        self._row_first_element_indices = row_first_element_indices

    @property
    def element_values(self):

        return self._element_values

    @property
    def element_column_indices(self):

        return self._element_column_indices

    @property
    def row_first_element_indices(self):

        return self._row_first_element_indices

    @property
    def size(self):

        return len(self._row_first_element_indices) - 1


csr_matrix_nb_type = common_utils.get_numba_type(CSRMatrix)

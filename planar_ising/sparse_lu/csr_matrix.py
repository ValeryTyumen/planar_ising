class CSRMatrix:

    def __init__(self, signs, logs, column_indices, row_first_element_indices):

        self._signs = signs
        self._logs = logs
        self._column_indices = column_indices
        self._row_first_element_indices = row_first_element_indices

    @property
    def signs(self):

        return self._signs

    @property
    def logs(self):

        return self._logs

    @property
    def column_indices(self):

        return self._column_indices

    @property
    def row_first_element_indices(self):

        return self._row_first_element_indices

    @property
    def size(self):

        return len(self._row_first_element_indices) - 1

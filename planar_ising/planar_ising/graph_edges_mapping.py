class GraphEdgesMapping:

    def __init__(self, first_dual_edges_mapping, second_dual_edges_mapping):

        self._first = first_dual_edges_mapping
        self._second = second_dual_edges_mapping

    @property
    def size(self):

        return self._first.shape[0]

    @property
    def first(self):

        return self._first

    @property
    def second(self):

        return self._second

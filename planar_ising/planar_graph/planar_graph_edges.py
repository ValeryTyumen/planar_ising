import numpy as np


class PlanarGraphEdges:
    """
    A list of planar graph edges. For each edge two incident vertices and next/previous edge
    indices in the rotating order around each incident vertex are stored.

    Notes
    -----
    The implemented data structure is described in R. J. Lipton, R. E. Tarjan,
    "A separator theorem for planar graphs", tech. rep., Stanford, CA, USA, 1977. 
    """

    def __init__(self, capacity):

        self._vertex1, self._vertex2, self._vertex1_next_edge_index, \
                self._vertex1_previous_edge_index, self._vertex2_next_edge_index, \
                self._vertex2_previous_edge_index = self._allocate_data(capacity)
        self._size = 0

    @property
    def size(self):

        return self._size

    @property
    def vertex1(self):

        return self._vertex1[:self._size]

    @property
    def vertex2(self):

        return self._vertex2[:self._size]

    def append(self, vertex1, vertex2):

        self._vertex1[self._size] = vertex1
        self._vertex2[self._size] = vertex2

        self._size += 1

    def extend(self, edges):

        self._vertex1 = np.concatenate((self._vertex1[:self._size], edges._vertex1[:edges._size]))
        self._vertex2 = np.concatenate((self._vertex2[:self._size], edges._vertex2[:edges._size]))
        self._vertex1_next_edge_index = np.concatenate((self._vertex1_next_edge_index[:self._size],
                edges._vertex1_next_edge_index[:edges._size]))
        self._vertex1_previous_edge_index = \
                np.concatenate((self._vertex1_previous_edge_index[:self._size],
                edges._vertex1_previous_edge_index[:edges._size]))
        self._vertex2_next_edge_index = np.concatenate((self._vertex2_next_edge_index[:self._size],
                edges._vertex2_next_edge_index[:edges._size]))
        self._vertex2_previous_edge_index = \
                np.concatenate((self._vertex2_previous_edge_index[:self._size],
                edges._vertex2_previous_edge_index[:edges._size]))

        self._size += edges.size

    def increase_capacity(self, capacity):

        vertex1, vertex2, vertex1_next_edge_index, vertex1_previous_edge_index, \
                vertex2_next_edge_index, vertex2_previous_edge_index = \
                self._allocate_data(capacity)

        vertex1[:self._size] = self._vertex1[:self._size]
        vertex2[:self._size] = self._vertex2[:self._size]
        vertex1_next_edge_index[:self._size] = self._vertex1_next_edge_index[:self._size]
        vertex1_previous_edge_index[:self._size] = self._vertex1_previous_edge_index[:self._size]
        vertex2_next_edge_index[:self._size] = self._vertex2_next_edge_index[:self._size]
        vertex2_previous_edge_index[:self._size] = self._vertex2_previous_edge_index[:self._size]

        self._vertex1 = vertex1
        self._vertex2 = vertex2
        self._vertex1_next_edge_index = vertex1_next_edge_index
        self._vertex1_previous_edge_index = vertex1_previous_edge_index
        self._vertex2_next_edge_index = vertex2_next_edge_index
        self._vertex2_previous_edge_index = vertex2_previous_edge_index

    def _allocate_data(self, capacity):

        vertex1 = np.zeros(capacity, dtype=np.int32)
        vertex2 = np.zeros(capacity, dtype=np.int32)
        vertex1_next_edge_index = np.zeros(capacity, dtype=np.int32)
        vertex1_previous_edge_index = np.zeros(capacity, dtype=np.int32)
        vertex2_next_edge_index = np.zeros(capacity, dtype=np.int32)
        vertex2_previous_edge_index = np.zeros(capacity, dtype=np.int32)

        return vertex1, vertex2, vertex1_next_edge_index, vertex1_previous_edge_index, \
                vertex2_next_edge_index, vertex2_previous_edge_index

    def get_opposite_vertex(self, edge_index, vertex):

        if vertex == self._vertex1[edge_index]:
            return self._vertex2[edge_index]

        return self._vertex1[edge_index]

    def set_next_edge(self, edge_index, vertex, other_edge_index):

        self._set_next_edge_only(edge_index, vertex, other_edge_index)
        self._set_previous_edge_only(other_edge_index, vertex, edge_index)

    def set_previous_edge(self, edge_index, vertex, other_edge_index):

        self._set_previous_edge_only(edge_index, vertex, other_edge_index)
        self._set_next_edge_only(other_edge_index, vertex, edge_index)

    def _set_next_edge_only(self, edge_index, vertex, other_edge_index):

        if vertex == self._vertex1[edge_index]:
            self._vertex1_next_edge_index[edge_index] = other_edge_index
        else:
            self._vertex2_next_edge_index[edge_index] = other_edge_index

    def _set_previous_edge_only(self, edge_index, vertex, other_edge_index):

        if vertex == self._vertex1[edge_index]:
            self._vertex1_previous_edge_index[edge_index] = other_edge_index
        else:
            self._vertex2_previous_edge_index[edge_index] = other_edge_index

    def get_next_edge_index(self, edge_index, vertex):

        if vertex == self._vertex1[edge_index]:
            return self._vertex1_next_edge_index[edge_index]

        return self._vertex2_next_edge_index[edge_index]

    def get_previous_edge_index(self, edge_index, vertex):

        if vertex == self._vertex1[edge_index]:
            return self._vertex1_previous_edge_index[edge_index]

        return self._vertex2_previous_edge_index[edge_index]

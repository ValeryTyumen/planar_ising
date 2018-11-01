import numpy as np
from numba import jit, jitclass
from numba.types import void, int32, boolean
from lipton_tarjan import planar_graph_nb_type
from . import numba_utils, utils


@jitclass([('_graph', planar_graph_nb_type),
        ('_spin_values', int32[:]),
        ('_component_indices', int32[:]),
        ('_next_vertices_in_components', int32[:]),
        ('_component_sizes', int32[:]),
        ('_component_first_vertices', int32[:]),
        ('_components_count', int32)])
class PartialSpinConfiguration:

    def __init__(self, graph):

        self._graph = graph
        size = graph.size

        self._spin_values = np.zeros(size, dtype=np.int32)
        self._component_indices = utils.repeat_int(-1, size)
        self._next_vertices_in_components = utils.repeat_int(-1, size)
        self._component_sizes = np.zeros(size, dtype=np.int32)
        self._component_first_vertices = np.zeros(size, dtype=np.int32)
        self._components_count = 0

    @property
    def spin_values(self):

        return self._spin_values

    def set_spin_pair(self, edge_index, set_equal_values):

        vertex1 = self._graph.edges.vertex1[edge_index]
        vertex2 = self._graph.edges.vertex2[edge_index]

        if self._spin_values[vertex1] != 0 and self._spin_values[vertex2] != 0:

            component_index1 = self._component_indices[vertex1]
            component_index2 = self._component_indices[vertex2]

            if component_index1 == component_index2:
                return

            if self._component_sizes[component_index1] < self._component_sizes[component_index2]:
                vertex1, vertex2 = vertex2, vertex1
                component_index1, component_index2 = component_index2, component_index1

            spin_value1 = self._spin_values[vertex1]
            spin_value2 = self._spin_values[vertex2]

            invert_component_spin_values = ((set_equal_values and spin_value1 != spin_value2) or \
                    ((not set_equal_values) and spin_value1 == spin_value2))

            _change_component_index(component_index2, component_index1,
                    invert_component_spin_values, self._spin_values, self._component_indices,
                    self._next_vertices_in_components, self._component_sizes,
                    self._component_first_vertices)

            return

        if self._spin_values[vertex1] == 0:
            vertex1, vertex2 = vertex2, vertex1

        if self._spin_values[vertex1] == 0:

            self._spin_values[vertex1] = 1

            component_index1 = self._components_count

            self._component_indices[vertex1] = component_index1
            self._component_sizes[component_index1] = 1
            self._component_first_vertices[component_index1] = vertex1
            self._components_count += 1

        else:
            component_index1 = self._component_indices[vertex1]

        if set_equal_values:
            self._spin_values[vertex2] = self._spin_values[vertex1]
        else:
            self._spin_values[vertex2] = -self._spin_values[vertex1]

        self._component_indices[vertex2] = component_index1
        self._next_vertices_in_components[vertex2] = \
                self._component_first_vertices[component_index1]

        self._component_sizes[component_index1] += 1
        self._component_first_vertices[component_index1] = vertex2


@jit(void(int32, int32, boolean, int32[:], int32[:], int32[:], int32[:], int32[:]), nopython=True)
def _change_component_index(component_index, new_component_index, invert_component_spin_values,
        spin_values, component_indices, next_vertices_in_components, component_sizes,
        component_first_vertices):

    first_vertex = component_first_vertices[component_index]
    last_vertex = -1

    current_vertex = first_vertex

    while current_vertex != -1:

        if invert_component_spin_values:
            spin_values[current_vertex] = -spin_values[current_vertex]

        component_indices[current_vertex] = new_component_index

        last_vertex = current_vertex

        current_vertex = next_vertices_in_components[current_vertex]

    next_vertices_in_components[last_vertex] = component_first_vertices[new_component_index]
    component_first_vertices[new_component_index] = first_vertex
    component_sizes[new_component_index] += component_sizes[component_index]


partial_spin_configuration_nb_type = numba_utils.get_numba_type(PartialSpinConfiguration)

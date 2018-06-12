import numpy as np
from numba import jit, jitclass
from numba.types import void, int32, boolean
from lipton_tarjan import planar_graph_nb_type
from . import utils


@jitclass([('_graph', planar_graph_nb_type),
        ('_spin_values', int32[:]),
        ('_component_indices', int32[:]),
        ('_component_sizes', int32[:]),
        ('_components_count', int32)])
class PartialSpinConfiguration:

    def __init__(self, graph):

        self._graph = graph

        size = graph.size
        self._spin_values = np.zeros(size, dtype=np.int32)
        self._component_indices = utils.repeat_int(-1, size)
        self._component_sizes = np.zeros(size, dtype=np.int32)
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

            _change_component_index(self._graph, self._component_indices, self._spin_values,
                    vertex2, component_index1, invert_component_spin_values)

            self._component_sizes[component_index1] += self._component_sizes[component_index2]

            return

        if self._spin_values[vertex1] == 0:
            vertex1, vertex2 = vertex2, vertex1

        if self._spin_values[vertex1] == 0:

            self._spin_values[vertex1] = 1

            component_index1 = self._components_count

            self._component_indices[vertex1] = component_index1
            self._component_sizes[component_index1] = 1
            self._components_count += 1

        else:
            component_index1 = self._component_indices[vertex1]

        if set_equal_values:
            self._spin_values[vertex2] = self._spin_values[vertex1]
        else:
            self._spin_values[vertex2] = -self._spin_values[vertex1]

        self._component_indices[vertex2] = component_index1
        self._component_sizes[component_index1] += 1

partial_spin_configuration_nb_type = PartialSpinConfiguration.class_type.instance_type

@jit(void(planar_graph_nb_type, int32[:], int32[:], int32, int32, boolean), nopython=True)
def _change_component_index(graph, component_indices, spin_values, component_vertex,
        new_component_index, invert_component_spin_values):

    component_index = component_indices[component_vertex]

    stack = [component_vertex]
    component_indices[component_vertex] = new_component_index

    while len(stack) != 0:

        vertex = stack.pop()

        if invert_component_spin_values:
            spin_values[vertex] = -spin_values[vertex]

        for adjacent_vertex in graph.get_adjacent_vertices(vertex):
            if component_indices[adjacent_vertex] == component_index:
                component_indices[adjacent_vertex] = new_component_index
                stack.append(adjacent_vertex)

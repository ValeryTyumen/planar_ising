import numpy as np
from numba import jit, jitclass
from numba.types import void, List, int32, boolean
from .planar_graph_edges import planar_graph_edges_nb_type
from .planar_graph import planar_graph_nb_type
from .. import common_utils


@jitclass([('_left_stack', List(int32)), ('_right_stack', List(int32))])
class _Queue:

    def __init__(self):

        # Hacks to make numba work
        self._left_stack = list(np.zeros(0, dtype=np.int32))
        self._right_stack = list(np.zeros(0, dtype=np.int32))

    def append(self, value):

        self._right_stack.append(value)

    def popleft(self):

        if len(self._left_stack) == 0:

            self._right_stack.reverse()
            self._left_stack = self._right_stack
            self._right_stack = list(np.zeros(0, dtype=np.int32))

        return self._left_stack.pop()

    def is_empty(self):

        return len(self._left_stack) + len(self._right_stack) == 0


def make_traverse_graph_via_bfs(callback, result_nb_type):
    """
        callback(vertex, incident_edge, result) - callback function
    """

    @jit(void(int32, planar_graph_nb_type, boolean[:], result_nb_type), nopython=True)
    def traverse_graph_via_bfs(start_vertex, graph, used_vertex_flags, result):

        queue = _Queue()
        queue.append(start_vertex)

        used_vertex_flags[start_vertex] = True

        while not queue.is_empty():

            vertex = queue.popleft()

            for incident_edge_index in graph.get_incident_edge_indices(vertex):

                adjacent_vertex = graph.edges.get_opposite_vertex(incident_edge_index, vertex)

                if not used_vertex_flags[adjacent_vertex]:

                    callback(vertex, graph.edges, incident_edge_index, result)
                    used_vertex_flags[adjacent_vertex] = True
                    queue.append(adjacent_vertex)

    return traverse_graph_via_bfs

def make_traverse_graph_via_post_order_dfs(callback, result_nb_type):

    @jit(int32[:](int32, planar_graph_nb_type, boolean[:], result_nb_type), nopython=True)
    def traverse_graph_via_post_order_dfs(start_vertex, graph, edges_mask, result):

        parent_edge_indices = common_utils.repeat_int(-1, graph.size)

        used_vertex_flags = common_utils.repeat_bool(False, graph.size)
        used_vertex_flags[start_vertex] = True

        stack = [start_vertex]

        while len(stack) != 0:

            vertex = stack.pop()
            stack.append(vertex)

            new_vertices_added_to_stack = False

            for incident_edge_index in graph.get_incident_edge_indices(vertex):
                if edges_mask[incident_edge_index]:

                    adjacent_vertex = graph.edges.get_opposite_vertex(incident_edge_index, vertex)

                    if not used_vertex_flags[adjacent_vertex]:

                        parent_edge_indices[adjacent_vertex] = incident_edge_index

                        used_vertex_flags[adjacent_vertex] = True
                        stack.append(adjacent_vertex)

                        new_vertices_added_to_stack = True

            if not new_vertices_added_to_stack:

                stack.pop()

                parent_edge_index = parent_edge_indices[vertex]

                if parent_edge_index != -1:
                    callback(vertex, graph.edges, parent_edge_index, result)

        return parent_edge_indices

    return traverse_graph_via_post_order_dfs

@jit(void(int32, planar_graph_edges_nb_type, int32, int32[:]), nopython=True)
def _color_adjacent_vertex(vertex, edges, incident_edge_index, colors):

    adjacent_vertex = edges.get_opposite_vertex(incident_edge_index, vertex)
    colors[adjacent_vertex] = colors[vertex]

_mark_connected_component = make_traverse_graph_via_bfs(_color_adjacent_vertex,
        int32[:])

@jit(int32[:](planar_graph_nb_type), nopython=True)
def color_connected_components(graph):

    colors = common_utils.repeat_int(-1, graph.size)
    current_color = -1

    used_vertex_flags = common_utils.repeat_bool(False, graph.size)

    for vertex in range(graph.size):
        if not used_vertex_flags[vertex]:

            current_color += 1
            colors[vertex] = current_color
            _mark_connected_component(vertex, graph, used_vertex_flags, colors)

    return colors

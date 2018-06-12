import numpy as np
from numba import jit
from numba import void, int32, boolean, float32
from lipton_tarjan import planar_graph_nb_type


@jit(int32[:](int32, int32), nopython=True)
def repeat_int(value, count):

    array = np.zeros(count, dtype=np.int32)
    array[:] = value

    return array

@jit(boolean[:](boolean, int32), nopython=True)
def repeat_bool(value, count):

    array = np.zeros(count, dtype=np.bool_)
    array[:] = value

    return array

@jit(float32[:](float32, int32), nopython=True)
def repeat_float(value, count): 

    array = np.zeros(count, dtype=np.float32)
    array[:] = value

    return array

def make_traverse_graph_via_post_order_dfs(callback, result_nb_type):

    @jit(int32[:](int32, planar_graph_nb_type, boolean[:], result_nb_type), nopython=True)
    def traverse_graph_via_post_order_dfs(start_vertex, graph, edges_mask, result):

        parent_edge_indices = repeat_int(-1, graph.size)

        used_vertex_flags = repeat_bool(False, graph.size)
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

@jit(int32[:](int32[:], int32), nopython=True)
def get_inverse_sub_mapping(sub_mapping, sub_elements_count):

    sub_element_exists_mask = (sub_mapping != -1)

    inverse_mapping = np.zeros(sub_elements_count, dtype=np.int32)
    inverse_mapping[sub_mapping[sub_element_exists_mask]] = \
            np.where(sub_element_exists_mask)[0].astype(np.int32)

    return inverse_mapping

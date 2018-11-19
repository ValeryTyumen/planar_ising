import numpy as np
from numba import jit
from numba.types import void, int32, boolean
from ..planar_graph import planar_graph_nb_type


@jit(void(planar_graph_nb_type, boolean[:], boolean[:], int32, int32), nopython=True)
def iterate_subgraph_incidence_indices(graph, subgraph_edges_mask, possible_incidences_mask,
        start_vertex_in_subgraph, start_edge_index_in_subgraph):

    current_vertex = start_vertex_in_subgraph
    current_edge_index = graph.edges.get_next_edge_index(start_edge_index_in_subgraph,
            start_vertex_in_subgraph)

    current_opposite_vertex = graph.edges.get_opposite_vertex(current_edge_index, current_vertex)

    while current_edge_index != start_edge_index_in_subgraph or \
            current_opposite_vertex != start_vertex_in_subgraph:

        if subgraph_edges_mask[current_edge_index]:
            current_vertex, current_opposite_vertex = current_opposite_vertex, \
                    current_vertex
        elif possible_incidences_mask[current_edge_index]:
            yield current_edge_index

        current_edge_index = graph.edges.get_next_edge_index(current_edge_index, current_vertex)
        current_opposite_vertex = graph.edges.get_opposite_vertex(current_edge_index,
                current_vertex)

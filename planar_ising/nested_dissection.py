import numpy as np
from numba import jit
from numba.types import void, Tuple, int32, boolean
from lipton_tarjan import PlanarGraph, planar_graph_nb_type, planar_graph_constructor, \
        planar_separator, separation_class
from .nested_dissection_map import NestedDissectionMap, nested_dissection_map_nb_type
from . import utils


@jit(nested_dissection_map_nb_type(planar_graph_nb_type, nested_dissection_map_nb_type),
        nopython=True)
def add_neighbours_to_top_level_separator(graph, nd_map):

    top_level_separator_in_order_index = np.where(nd_map.in_order_pre_order_mapping == 0)[0][0]

    if not np.any(nd_map.in_order_map == top_level_separator_in_order_index):
        return nd_map

    in_order_map = nd_map.in_order_map.copy() 

    for vertex, in_order_index in enumerate(nd_map.in_order_map):

        if in_order_index == top_level_separator_in_order_index:
            for adjacent_vertex in graph.get_adjacent_vertices(vertex):
                in_order_map[adjacent_vertex] = top_level_separator_in_order_index

    return NestedDissectionMap(in_order_map, nd_map.in_order_pre_order_mapping)

@jit(Tuple((int32[:], int32))(planar_graph_nb_type, nested_dissection_map_nb_type, boolean[:]),
        nopython=True)
def get_nested_dissection_permutation_and_top_level_separator_size(graph, nd_map,
        perfect_matching_mask):

    perfect_matching_edge_vertices1 = graph.edges.vertex1[perfect_matching_mask]
    perfect_matching_edge_vertices2 = graph.edges.vertex2[perfect_matching_mask]

    pre_order_map = nd_map.in_order_pre_order_mapping[nd_map.in_order_map]

    perfect_matching_edges_pre_order_map = \
            np.minimum(pre_order_map[perfect_matching_edge_vertices1],
            pre_order_map[perfect_matching_edge_vertices2])

    top_level_separator_edges_count = (perfect_matching_edges_pre_order_map == 0).sum()

    perfect_matching_edges_permutation = np.argsort(perfect_matching_edges_pre_order_map)[::-1]

    permutation = np.zeros(graph.size, dtype=np.int32)
    permutation[::2] = perfect_matching_edge_vertices1[perfect_matching_edges_permutation]
    permutation[1::2] = perfect_matching_edge_vertices2[perfect_matching_edges_permutation]

    return permutation, 2*top_level_separator_edges_count

@jit(planar_graph_nb_type(planar_graph_nb_type), nopython=True)
def _normalize_vertex_costs(graph):

    vertex_costs = graph.vertex_costs.copy()

    vertex_costs /= vertex_costs.sum()

    return PlanarGraph(vertex_costs, graph.incident_edge_example_indices, graph.edges)

@jit(nested_dissection_map_nb_type(planar_graph_nb_type, boolean[:]), nopython=True)
def _get_nested_dissection_submap(graph, vertices_for_map_mask):

    separation = planar_separator.mark_separation(graph)

    first_part_mask = (separation == separation_class.FIRST_PART)
    second_part_mask = (separation == separation_class.SECOND_PART)

    in_order_map = utils.repeat_int(-1, graph.size)
    in_order_map[vertices_for_map_mask] = 0

    if not np.any(first_part_mask) or not np.any(second_part_mask):
        return NestedDissectionMap(in_order_map, np.zeros(1, dtype=np.int32))

    separator_mask = (separation == separation_class.SEPARATOR)

    subgraph_vertices_masks = (np.logical_or(first_part_mask, separator_mask),
            np.logical_or(second_part_mask, separator_mask))
    non_separator_edges_mask = \
            np.logical_or(separation[graph.edges.vertex1] != separation_class.SEPARATOR,
            separation[graph.edges.vertex2] != separation_class.SEPARATOR)

    on_first_subgraph = True
    in_order_offset = 0
    pre_order_offset = 1

    in_order_pre_order_mapping = np.zeros(0, dtype=np.int32)

    for subgraph_vertices_mask in subgraph_vertices_masks:

        subgraph_vertices_mapping, subgraph_edge_indices_mapping, subgraph = \
                planar_graph_constructor.construct_subgraph(graph, subgraph_vertices_mask,
                non_separator_edges_mask)

        subgraph = _normalize_vertex_costs(subgraph)

        graph_vertices_mapping = utils.get_inverse_sub_mapping(subgraph_vertices_mapping,
                subgraph.size)

        subgraph_vertices_for_map_mask = \
                np.logical_and(vertices_for_map_mask[graph_vertices_mapping],
                separation[graph_vertices_mapping] != separation_class.SEPARATOR)

        subgraph_map = _get_nested_dissection_submap(subgraph, subgraph_vertices_for_map_mask)

        in_order_pre_order_mapping = np.concatenate((in_order_pre_order_mapping,
                subgraph_map.in_order_pre_order_mapping + pre_order_offset)).astype(np.int32)

        in_order_map[graph_vertices_mapping[subgraph_vertices_for_map_mask]] = \
                subgraph_map.in_order_map[subgraph_vertices_for_map_mask] + in_order_offset

        if on_first_subgraph:

            separator_in_order_index = subgraph_map.in_order_pre_order_mapping.shape[0]

            in_order_map[np.logical_and(vertices_for_map_mask,
                    separation == separation_class.SEPARATOR)] = separator_in_order_index

            in_order_offset = 1 + separator_in_order_index
            pre_order_offset = subgraph_map.in_order_pre_order_mapping.max() + 2

            in_order_pre_order_mapping = np.concatenate((in_order_pre_order_mapping,
                    np.zeros(1, dtype=np.int32)))

        on_first_subgraph = False

    return NestedDissectionMap(in_order_map, in_order_pre_order_mapping)

@jit(nested_dissection_map_nb_type(planar_graph_nb_type), nopython=True)
def get_nested_dissection_map(graph):

    return _get_nested_dissection_submap(graph, utils.repeat_bool(True, graph.size))

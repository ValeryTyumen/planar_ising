import numpy as np
from numba import jit
from numba.types import int32, boolean
from . import utils
from .. import common_utils
from ..planar_graph import planar_graph_nb_type, PlanarGraph, planar_graph_constructor
from ..lipton_tarjan import planar_separator, separation_class


@jit(int32[:](planar_graph_nb_type, boolean[:], int32[:]), nopython=True)
def _permute_subgraph_vertices(graph, vertices_to_permute_mask, separation):

    if separation[0] == separation_class.UNDEFINED:
        is_top_level = False
        separation = planar_separator.mark_separation(graph)
    else:
        is_top_level = True

    first_part_mask = (separation == separation_class.FIRST_PART)
    second_part_mask = (separation == separation_class.SECOND_PART)
 
    if not np.any(first_part_mask) or not np.any(second_part_mask):

        if is_top_level:
            return np.concatenate((
                    np.where(separation != separation_class.SEPARATOR)[0].astype(np.int32),
                    np.where(separation == separation_class.SEPARATOR)[0].astype(np.int32)))

        return np.where(vertices_to_permute_mask)[0].astype(np.int32)

    separator_mask = (separation == separation_class.SEPARATOR)

    subgraph_vertices_masks = (np.logical_or(first_part_mask, separator_mask),
            np.logical_or(second_part_mask, separator_mask))
    non_separator_edges_mask = \
            np.logical_or(separation[graph.edges.vertex1] != separation_class.SEPARATOR,
            separation[graph.edges.vertex2] != separation_class.SEPARATOR)

    vertices_permutation = np.zeros(np.where(vertices_to_permute_mask)[0].shape[0], dtype=np.int32)

    vertices_permutation_offset = 0

    for subgraph_vertices_mask in subgraph_vertices_masks:

        subgraph_vertices_mapping, subgraph_edge_indices_mapping, subgraph = \
                planar_graph_constructor.construct_subgraph(graph, subgraph_vertices_mask,
                non_separator_edges_mask)

        subgraph = utils.normalize_vertex_costs(subgraph)

        graph_vertices_mapping = np.zeros(subgraph.size, dtype=np.int32)

        graph_vertices_mapping[subgraph_vertices_mapping[subgraph_vertices_mask]] = \
                np.where(subgraph_vertices_mask)[0].astype(np.int32)

        subgraph_vertices_to_permute_mask = \
                np.logical_and(vertices_to_permute_mask[graph_vertices_mapping],
                separation[graph_vertices_mapping] != separation_class.SEPARATOR)

        no_separation = np.array([separation_class.UNDEFINED], dtype=np.int32)

        subgraph_vertices_permutation = _permute_subgraph_vertices(subgraph,
                subgraph_vertices_to_permute_mask, no_separation)

        vertices_permutation[vertices_permutation_offset:vertices_permutation_offset + \
                subgraph_vertices_permutation.shape[0]] = \
                graph_vertices_mapping[subgraph_vertices_permutation]

        vertices_permutation_offset += subgraph_vertices_permutation.shape[0]

    vertices_permutation[vertices_permutation_offset:] = \
            np.where(np.logical_and(separation == separation_class.SEPARATOR,
            vertices_to_permute_mask))[0].astype(np.int32)

    return vertices_permutation

@jit(int32[:](planar_graph_nb_type, int32[:]), nopython=True)
def permute_vertices(graph, top_level_separation):

    vertices_to_permute_mask = common_utils.repeat_bool(True, graph.size)

    return _permute_subgraph_vertices(graph, vertices_to_permute_mask, top_level_separation)

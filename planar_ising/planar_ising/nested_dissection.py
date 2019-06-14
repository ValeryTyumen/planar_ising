import numpy as np
from . import utils
from .. import common_utils
from ..planar_graph import PlanarGraph, PlanarGraphConstructor
from ..lipton_tarjan import PlanarSeparator, separation_class


class NestedDissection:

    @staticmethod
    def permute_vertices(graph, top_level_separation):

        vertices_to_permute_mask = common_utils.repeat_bool(True, graph.size)

        return NestedDissection._permute_subgraph_vertices(graph, vertices_to_permute_mask,
                top_level_separation)

    @staticmethod
    def _permute_subgraph_vertices(graph, vertices_to_permute_mask, separation):

        if separation[0] == separation_class.UNDEFINED:
            is_top_level = False
            separation = PlanarSeparator.mark_separation(graph)
        else:
            is_top_level = True

        first_part_mask = (separation == separation_class.FIRST_PART)
        second_part_mask = (separation == separation_class.SECOND_PART)
     
        if not np.any(first_part_mask) or not np.any(second_part_mask):

            if is_top_level:
                return np.concatenate((np.where(separation != separation_class.SEPARATOR)[0],
                        np.where(separation == separation_class.SEPARATOR)[0]))

            return np.where(vertices_to_permute_mask)[0]

        separator_mask = (separation == separation_class.SEPARATOR)

        subgraph_vertices_masks = (first_part_mask | separator_mask,
                second_part_mask | separator_mask)
        non_separator_edges_mask = \
                ((separation[graph.edges.vertex1] != separation_class.SEPARATOR) | \
                (separation[graph.edges.vertex2] != separation_class.SEPARATOR))

        vertices_permutation = np.zeros(vertices_to_permute_mask.sum(), dtype=int)

        vertices_permutation_offset = 0

        for subgraph_vertices_mask in subgraph_vertices_masks:

            subgraph_vertices_mapping, subgraph_edge_indices_mapping, subgraph = \
                    PlanarGraphConstructor.construct_subgraph(graph, subgraph_vertices_mask,
                    non_separator_edges_mask)

            subgraph.vertex_costs /= subgraph.vertex_costs.sum()

            graph_vertices_mapping = np.zeros(subgraph.size, dtype=int)

            graph_vertices_mapping[subgraph_vertices_mapping[subgraph_vertices_mask]] = \
                    np.where(subgraph_vertices_mask)[0]

            subgraph_vertices_to_permute_mask = \
                    (vertices_to_permute_mask[graph_vertices_mapping] & \
                    (separation[graph_vertices_mapping] != separation_class.SEPARATOR))

            no_separation = np.array([separation_class.UNDEFINED])

            subgraph_vertices_permutation = NestedDissection._permute_subgraph_vertices(subgraph,
                    subgraph_vertices_to_permute_mask, no_separation)

            vertices_permutation[vertices_permutation_offset:vertices_permutation_offset + \
                    subgraph_vertices_permutation.shape[0]] = \
                    graph_vertices_mapping[subgraph_vertices_permutation]

            vertices_permutation_offset += subgraph_vertices_permutation.shape[0]

        vertices_permutation[vertices_permutation_offset:] = \
                np.where(np.logical_and(separation == separation_class.SEPARATOR,
                vertices_to_permute_mask))[0]

        return vertices_permutation

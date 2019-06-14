import numpy as np
from ..planar_graph import PlanarGraph, PlanarGraphEdges
from .. import common_utils


class PerfectMatchingCollapser:

    @staticmethod
    def collapse_perfect_matching(expanded_dual_subgraph, perfect_matching_mask):

        size = expanded_dual_subgraph.size//2

        vertex_costs = np.ones(size)/size

        new_vertices_mapping = np.zeros(expanded_dual_subgraph.size, dtype=int)

        new_vertices_mapping[expanded_dual_subgraph.edges.vertex1[perfect_matching_mask]] = \
                np.arange(size)
        new_vertices_mapping[expanded_dual_subgraph.edges.vertex2[perfect_matching_mask]] = \
                np.arange(size)

        edges = PlanarGraphEdges(expanded_dual_subgraph.edges_count)

        new_edge_indices_mapping = common_utils.repeat_int(-1, expanded_dual_subgraph.edges_count)

        for edge_index, is_new_vertex in enumerate(perfect_matching_mask):

            if is_new_vertex:

                new_vertex = new_vertices_mapping[expanded_dual_subgraph.edges.vertex1[edge_index]]

                previous_adjacent_vertex = -1
                previous_new_incident_edge_index = -1
                first_new_incident_edge_index = -1
                first_adjacent_vertex = -1

                single_new_incident_edge = True

                for incident_edge_index in PerfectMatchingCollapser._iterate_edge_incidences(
                        expanded_dual_subgraph, edge_index):

                    adjacent_vertex = expanded_dual_subgraph.edges.vertex1[incident_edge_index]

                    if new_vertices_mapping[adjacent_vertex] == new_vertex:
                        adjacent_vertex = expanded_dual_subgraph.edges.vertex2[incident_edge_index]

                    new_adjacent_vertex = new_vertices_mapping[adjacent_vertex]

                    new_incident_edge_index = -1

                    if new_edge_indices_mapping[incident_edge_index] == -1:

                        if previous_adjacent_vertex == -1 or \
                                new_vertices_mapping[adjacent_vertex] != \
                                new_vertices_mapping[previous_adjacent_vertex]:

                            if first_adjacent_vertex == -1 or \
                                    new_vertices_mapping[adjacent_vertex] != \
                                    new_vertices_mapping[first_adjacent_vertex]:

                                new_incident_edge_index = edges.size
                                edges.append(new_vertex, new_adjacent_vertex)

                                new_edge_indices_mapping[incident_edge_index] = \
                                        new_incident_edge_index

                            else:

                                new_incident_edge_index = first_new_incident_edge_index

                                new_edge_indices_mapping[incident_edge_index] = \
                                    first_new_incident_edge_index

                            if previous_new_incident_edge_index != -1:
                                edges.set_previous_edge(new_incident_edge_index, new_vertex,
                                        previous_new_incident_edge_index)

                        else:

                            new_incident_edge_index = previous_new_incident_edge_index

                            new_edge_indices_mapping[incident_edge_index] = \
                                    previous_new_incident_edge_index

                    else:

                        new_incident_edge_index = new_edge_indices_mapping[incident_edge_index]

                        if previous_new_incident_edge_index != -1 and new_incident_edge_index != \
                                previous_new_incident_edge_index:
                            edges.set_previous_edge(new_incident_edge_index, new_vertex,
                                    previous_new_incident_edge_index)

                    if first_new_incident_edge_index == -1:
                        first_new_incident_edge_index = new_incident_edge_index

                    if new_incident_edge_index != first_new_incident_edge_index:
                        single_new_incident_edge = False

                    previous_new_incident_edge_index = new_incident_edge_index

                    if first_adjacent_vertex == -1:
                        first_adjacent_vertex = adjacent_vertex

                    previous_adjacent_vertex = adjacent_vertex

                if single_new_incident_edge or (first_new_incident_edge_index != \
                        previous_new_incident_edge_index):
                    # no vertices of degree 0, when at least one pm is present
                    edges.set_previous_edge(first_new_incident_edge_index, new_vertex, \
                            previous_new_incident_edge_index)

        incident_edge_example_indices = common_utils.repeat_int(-1, size)

        for new_edge_index in range(edges.size):
            incident_edge_example_indices[edges.vertex1[new_edge_index]] = new_edge_index
            incident_edge_example_indices[edges.vertex2[new_edge_index]] = new_edge_index

        return new_vertices_mapping, new_edge_indices_mapping, PlanarGraph(vertex_costs,
                incident_edge_example_indices, edges)

    @staticmethod
    def _iterate_edge_incidences(graph, edge_index):

        start_vertex = graph.edges.vertex1[edge_index]

        current_vertex = start_vertex
        current_edge_index = graph.edges.get_next_edge_index(edge_index, current_vertex)

        current_opposite_vertex = graph.edges.get_opposite_vertex(current_edge_index,
                current_vertex)

        while current_edge_index != edge_index or current_opposite_vertex != start_vertex:

            if current_edge_index == edge_index:
                current_vertex, current_opposite_vertex = current_opposite_vertex, current_vertex
            else:
                yield current_edge_index

            current_edge_index = graph.edges.get_next_edge_index(current_edge_index, current_vertex)
            current_opposite_vertex = graph.edges.get_opposite_vertex(current_edge_index,
                    current_vertex)

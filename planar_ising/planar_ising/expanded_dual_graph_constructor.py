import numpy as np
from .graph_edges_mapping import GraphEdgesMapping
from .. import common_utils
from ..planar_graph import PlanarGraphEdges, PlanarGraph, search_utils


class ExpandedDualGraphConstructor:

    @staticmethod
    def construct(graph):
        """
            graph is assumed to be connected and triangulated
        """

        expanded_dual_graph_vertices_count = 2*graph.edges_count
        dual_edges_count = graph.edges_count
        intercity_edges_count = 2*graph.edges_count

        expanded_dual_graph_edges = PlanarGraphEdges(dual_edges_count + intercity_edges_count)
        expanded_dual_graph_incident_edge_example_indices = common_utils.repeat_int(-1,
                expanded_dual_graph_vertices_count)

        first_dual_edges_mapping = common_utils.repeat_int(-1, dual_edges_count + \
                intercity_edges_count)
        second_dual_edges_mapping = common_utils.repeat_int(-1, dual_edges_count + \
                intercity_edges_count)

        for edge_index in range(graph.edges_count):

            dual_edge_vertex1 = 2*edge_index
            dual_edge_vertex2 = 2*edge_index + 1

            expanded_dual_graph_edges.append(dual_edge_vertex1, dual_edge_vertex2)
            expanded_dual_graph_incident_edge_example_indices[dual_edge_vertex1] = edge_index
            expanded_dual_graph_incident_edge_example_indices[dual_edge_vertex2] = edge_index

            first_dual_edges_mapping[edge_index] = edge_index

        for vertex in range(graph.size):
            for edge_index in graph.get_incident_edge_indices(np.int32(vertex)):

                next_edge_index = graph.edges.get_next_edge_index(edge_index, vertex)
     
                dual_vertex = ExpandedDualGraphConstructor._get_corresponding_dual_vertex(
                        graph.edges, edge_index, vertex, expanded_dual_graph_edges, edge_index)
                next_dual_vertex = ExpandedDualGraphConstructor._get_corresponding_dual_vertex(
                        graph.edges, next_edge_index, vertex, expanded_dual_graph_edges,
                        next_edge_index)
                next_non_dual_vertex = expanded_dual_graph_edges.get_opposite_vertex(
                        next_edge_index, next_dual_vertex)

                intercity_edge_index = expanded_dual_graph_edges.size

                expanded_dual_graph_edges.append(dual_vertex, next_non_dual_vertex)

                expanded_dual_graph_edges.set_next_edge(intercity_edge_index, dual_vertex,
                        edge_index)
                expanded_dual_graph_edges.set_next_edge(next_edge_index, next_non_dual_vertex,
                        intercity_edge_index)

                first_dual_edges_mapping[intercity_edge_index] = edge_index
                second_dual_edges_mapping[intercity_edge_index] = next_edge_index

        # `expanded_dual_graph_incident_edge_example_indices` doesn't contain intercity edge indices
        for vertex in range(expanded_dual_graph_vertices_count):

            incident_edge_index = expanded_dual_graph_incident_edge_example_indices[vertex]

            next_edge_index = expanded_dual_graph_edges.get_next_edge_index(incident_edge_index,
                    vertex)

            previous_edge_index = expanded_dual_graph_edges.get_previous_edge_index(
                    incident_edge_index, vertex)

            expanded_dual_graph_edges.set_next_edge(next_edge_index, vertex, previous_edge_index)

        # never used
        expanded_dual_graph_vertex_costs = np.ones(expanded_dual_graph_vertices_count)/\
                expanded_dual_graph_vertices_count
 
        return GraphEdgesMapping(first_dual_edges_mapping, second_dual_edges_mapping), \
                PlanarGraph(expanded_dual_graph_vertex_costs,
                expanded_dual_graph_incident_edge_example_indices, expanded_dual_graph_edges)
 
    @staticmethod
    def _get_corresponding_dual_vertex(edges, edge_index, vertex, expanded_dual_graph_edges,
            dual_edge_index):

        if vertex == edges.vertex1[edge_index]:
            return expanded_dual_graph_edges.vertex1[edge_index]

        return expanded_dual_graph_edges.vertex2[edge_index]

    @staticmethod
    def get_expanded_dual_graph_log_weights(interaction_values, graph_edges_mapping):

        log_weights = np.zeros(graph_edges_mapping.size)

        dual_edges_mask = (graph_edges_mapping.second == -1)

        log_weights[dual_edges_mask] = \
                interaction_values[graph_edges_mapping.first[dual_edges_mask]]*2

        return log_weights

    @staticmethod
    def get_kasteleyn_orientation(graph, graph_edges_mapping, expanded_dual_graph):

        dual_edges_mapping = np.zeros(graph.edges_count, dtype=int)

        dual_edges_mask = (graph_edges_mapping.second == -1)

        dual_edges_mapping[graph_edges_mapping.first[dual_edges_mask]] = \
                np.where(dual_edges_mask)[0]

        odd_orientation = ExpandedDualGraphConstructor._get_odd_orientation(graph)
        kasteleyn_orientation = common_utils.repeat_int(-1, expanded_dual_graph.edges_count)

        for vertex in range(graph.size):
            for edge_index in graph.get_incident_edge_indices(vertex):

                dual_edge_index = dual_edges_mapping[edge_index]
                dual_vertex = ExpandedDualGraphConstructor._get_corresponding_dual_vertex(
                        graph.edges, edge_index, vertex, expanded_dual_graph.edges, dual_edge_index)

                if odd_orientation[edge_index] != vertex:
                    kasteleyn_orientation[dual_edge_index] = dual_vertex

                previous_intercity_edge_index = \
                        expanded_dual_graph.edges.get_previous_edge_index(dual_edge_index,
                        dual_vertex)
                kasteleyn_orientation[previous_intercity_edge_index] = \
                        expanded_dual_graph.edges.get_opposite_vertex(previous_intercity_edge_index,
                        dual_vertex)

        return kasteleyn_orientation
 
    @staticmethod
    def _get_odd_orientation(graph):
        """
            The graph is assumed to be connected here
        """

        orientation = graph.edges.vertex1.copy()
        vertex_in_degrees = np.zeros(graph.size, dtype=int)

        for vertex in range(graph.size):
            for incident_edge_index in graph.get_incident_edge_indices(vertex):
                if orientation[incident_edge_index] == vertex:
                    vertex_in_degrees[vertex] += 1

        edges_mask = common_utils.repeat_bool(True, graph.edges_count)

        search_utils.traverse_graph_via_post_order_dfs(0, graph, edges_mask,
                (orientation, vertex_in_degrees),
                ExpandedDualGraphConstructor._set_parent_edge_odd_orientation)

        return orientation

    @staticmethod
    def _set_parent_edge_odd_orientation(vertex, edges, parent_edge_index,
            orientation_and_vertex_in_degrees):

        orientation, vertex_in_degrees = orientation_and_vertex_in_degrees

        parent_vertex = edges.get_opposite_vertex(parent_edge_index, vertex)

        if vertex_in_degrees[vertex]%2 == 0:

            if orientation[parent_edge_index] == vertex:

                vertex_in_degrees[vertex] -= 1
                vertex_in_degrees[parent_vertex] += 1
                orientation[parent_edge_index] = parent_vertex

            else:

                vertex_in_degrees[vertex] += 1
                vertex_in_degrees[parent_vertex] -= 1
                orientation[parent_edge_index] = vertex

    @staticmethod
    def get_expanded_dual_subgraph_perfect_matching(graph, graph_edges_mapping, spin_values):

        dual_edges_mask = (graph_edges_mapping.second == -1)

        dual_edge_vertices1 = graph.edges.vertex1[graph_edges_mapping.first[dual_edges_mask]]
        dual_edge_vertices2 = graph.edges.vertex2[graph_edges_mapping.first[dual_edges_mask]]

        dual_edge_spin_values1 = spin_values[dual_edge_vertices1]
        dual_edge_spin_values2 = spin_values[dual_edge_vertices2]

        intercity_edges_mask = (~dual_edges_mask)

        intercity_edge_first_vertices1 = \
                graph.edges.vertex1[graph_edges_mapping.first[intercity_edges_mask]]
        intercity_edge_first_vertices2 = \
                graph.edges.vertex2[graph_edges_mapping.first[intercity_edges_mask]]
        intercity_edge_second_vertices1 = \
                graph.edges.vertex1[graph_edges_mapping.second[intercity_edges_mask]]
        intercity_edge_second_vertices2 = \
                graph.edges.vertex2[graph_edges_mapping.second[intercity_edges_mask]]

        intercity_edge_first_spin_values1 = spin_values[intercity_edge_first_vertices1]
        intercity_edge_first_spin_values2 = spin_values[intercity_edge_first_vertices2]
        intercity_edge_second_spin_values1 = spin_values[intercity_edge_second_vertices1]
        intercity_edge_second_spin_values2 = spin_values[intercity_edge_second_vertices2]

        for array in (dual_edge_spin_values1, dual_edge_spin_values2, intercity_edge_first_spin_values1,
                intercity_edge_first_spin_values2, intercity_edge_second_spin_values1,
                intercity_edge_second_spin_values2):

            array[array == 0] = 1

        perfect_matching_mask = common_utils.repeat_bool(False, graph_edges_mapping.size)

        perfect_matching_mask[dual_edges_mask] = (dual_edge_spin_values1 == dual_edge_spin_values2)

        perfect_matching_mask[intercity_edges_mask] = \
                ((intercity_edge_first_spin_values1 != intercity_edge_first_spin_values2) & \
                (intercity_edge_second_spin_values1 != intercity_edge_second_spin_values2))

        return perfect_matching_mask

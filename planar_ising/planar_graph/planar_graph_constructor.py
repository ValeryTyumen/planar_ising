import numpy as np
from .planar_graph import PlanarGraph
from .planar_graph_edges import PlanarGraphEdges
from .. import common_utils


class PlanarGraphConstructor:
    """
    A static class with different planar graph construction methods.
    """

    @staticmethod
    def construct_subgraph(graph, subgraph_vertices_mask, subgraph_edges_mask):
        """
        Linear algorithm for subgraph construction.

        Parameters
        ----------
        graph : PlanarGraph
        subgraph_vertices_mask : array_like, boolean
            Boolean mask of vertices to leave in subgraph.
        subgraph_edges_mask : array_like, boolean
            Boolean mask of edges to leave in subgraph.

        Returns
        -------
        new_vertices_mapping : array_like, int32
            Mapping from `graph` vertices to corresponding `subgraph` vertices. If `graph` vertex is
            deleted, `-1` is substituted.
        new_edge_indices_mapping : array_like, int32
            Mapping from `graph` edge indices to corresponding `subgraph` edge indices. If `graph`
            edge index is deleted, `-1` is substituted.
        subgraph : PlanarGraph
            Result subgraph
        """

        vertex_costs = graph.vertex_costs[subgraph_vertices_mask]

        new_vertices_mapping = -np.ones(graph.size, dtype=int)
        current_new_vertex = 0

        for vertex, is_in_subgraph in enumerate(subgraph_vertices_mask):
            if is_in_subgraph:
                new_vertices_mapping[vertex] = current_new_vertex
                current_new_vertex += 1

        edges_count = 0
        new_edge_indices_mapping = -np.ones(graph.edges_count, dtype=int)

        for edge_index in range(graph.edges_count):

            edge_vertex1 = graph.edges.vertex1[edge_index]
            edge_vertex2 = graph.edges.vertex2[edge_index]

            if subgraph_vertices_mask[edge_vertex1] and subgraph_vertices_mask[edge_vertex2] and \
                    subgraph_edges_mask[edge_index]:
                new_edge_indices_mapping[edge_index] = edges_count
                edges_count += 1

        edges = PlanarGraphEdges(edges_count)

        for edge_index in range(graph.edges_count):

            edge_vertex1 = graph.edges.vertex1[edge_index]
            edge_vertex2 = graph.edges.vertex2[edge_index]

            if subgraph_vertices_mask[edge_vertex1] and subgraph_vertices_mask[edge_vertex2] and \
                    subgraph_edges_mask[edge_index]:
                edges.append(new_vertices_mapping[edge_vertex1], new_vertices_mapping[edge_vertex2])

        incident_edge_example_indices = -np.ones(len(vertex_costs), dtype=int)

        for vertex, is_in_subgraph in enumerate(subgraph_vertices_mask):
            if is_in_subgraph:

                new_vertex = new_vertices_mapping[vertex]

                first_new_edge_index = -1
                previous_new_edge_index = -1

                for edge_index in graph.get_incident_edge_indices(vertex):

                    new_edge_index = new_edge_indices_mapping[edge_index]

                    if new_edge_index != -1:

                        if previous_new_edge_index == -1:
                            incident_edge_example_indices[new_vertex] = new_edge_index
                            first_new_edge_index = new_edge_index
                        else:
                            edges.set_previous_edge(new_edge_index, new_vertex, previous_new_edge_index)

                        previous_new_edge_index = new_edge_index

                if first_new_edge_index != -1:
                    edges.set_previous_edge(first_new_edge_index, new_vertex, previous_new_edge_index)

        return new_vertices_mapping, new_edge_indices_mapping, PlanarGraph(vertex_costs,
                incident_edge_example_indices, edges)

    @staticmethod
    def clone_graph(graph):
        """
        Graph cloning.

        Parameters
        ----------
        graph : PlanarGraph

        Returns
        -------
        PlanarGraph
            The same graph up to different incident edge examples.
        """

        _, _, graph = PlanarGraphConstructor.construct_subgraph(graph,
                np.ones(graph.size, dtype=bool), np.ones(graph.edges_count, dtype=bool))

        return graph

    @staticmethod
    def _create_edges_and_map_adjacencies(adjacent_vertices):

        edges = PlanarGraphEdges(sum(len(vertices) for vertices in adjacent_vertices)//2)

        edge_indices_by_adjacencies = {}

        for vertex, vertex_adjacent_vertices in enumerate(adjacent_vertices):
            for adjacent_vertex in vertex_adjacent_vertices:

                if (vertex, adjacent_vertex) not in edge_indices_by_adjacencies:

                    edge_indices_by_adjacencies[(vertex, adjacent_vertex)] = edges.size
                    edge_indices_by_adjacencies[(adjacent_vertex, vertex)] = edges.size
                    edges.append(vertex, adjacent_vertex)

        return edges, edge_indices_by_adjacencies

    @staticmethod
    def construct_from_ordered_adjacencies(ordered_adjacencies):
        """
        Convenient method for constructing planar graph.

        Parameters
        ----------
        ordered_adjacencies : list of list of int
            The list, where for each vertex the list of its adjacent vertices is provided in the
            order of ccw traversal (or cw traversal, it's just a convention). For instance,
            `[[1, 2, 3, 4], [0], [0], [0], [0]]` would encode a "star" graph with 4 edges.

        Returns
        -------
        PlanarGraph

        Notes
        -----
        Only normal graphs are supported, i.e. no multiple edges or loops.
        """

            
        vertices_count = len(ordered_adjacencies)

        vertex_costs = np.ones(vertices_count, dtype=float)/vertices_count

        edges, edge_indices_by_adjacencies = \
                PlanarGraphConstructor._create_edges_and_map_adjacencies(ordered_adjacencies)

        incident_edge_example_indices = -np.ones(vertices_count, dtype=int)

        for vertex, vertex_ordered_adjacencies in enumerate(ordered_adjacencies):

            adjacent_vertices_count = len(vertex_ordered_adjacencies)

            if adjacent_vertices_count != 0:

                first_adjacent_vertex = vertex_ordered_adjacencies[0]
                first_incident_edge_index = edge_indices_by_adjacencies[(vertex,
                        first_adjacent_vertex)]
                incident_edge_example_indices[vertex] = first_incident_edge_index

                for adjacent_vertex_index, adjacent_vertex in \
                        enumerate(vertex_ordered_adjacencies):

                    incident_edge_index = edge_indices_by_adjacencies[(vertex, adjacent_vertex)]

                    next_adjacent_vertex_index = (adjacent_vertex_index + 1)%adjacent_vertices_count
                    next_adjacent_vertex = \
                            vertex_ordered_adjacencies[next_adjacent_vertex_index]
                    next_incident_edge_index = \
                            edge_indices_by_adjacencies[(vertex, next_adjacent_vertex)]

                    edges.set_next_edge(incident_edge_index, vertex, next_incident_edge_index)

        return PlanarGraph(vertex_costs, incident_edge_example_indices, edges)

    @staticmethod
    def remove_double_edges(graph):

        connecting_edge_indices = common_utils.repeat_int(-1, graph.size)
        new_edge_indices_mapping1 = np.arange(graph.edges_count, dtype=int)

        edge_indices_mask = common_utils.repeat_bool(True, graph.edges_count)

        for vertex in range(graph.size):

            for edge_index in graph.get_incident_edge_indices(vertex):

                adjacent_vertex = graph.edges.get_opposite_vertex(edge_index, vertex)

                if connecting_edge_indices[adjacent_vertex] != -1:
                    edge_indices_mask[edge_index] = False
                    new_edge_indices_mapping1[edge_index] = connecting_edge_indices[adjacent_vertex]
                else:
                    if edge_indices_mask[edge_index]:
                        connecting_edge_indices[adjacent_vertex] = edge_index

            for adjacent_vertex in graph.get_adjacent_vertices(vertex):
                connecting_edge_indices[adjacent_vertex] = -1

        new_vertices_mapping, new_edge_indices_mapping2, graph = \
                PlanarGraphConstructor.construct_subgraph(graph, common_utils.repeat_bool(True,
                graph.size), edge_indices_mask)

        new_edge_indices_mapping = new_edge_indices_mapping2[new_edge_indices_mapping1]

        return new_vertices_mapping, new_edge_indices_mapping, graph

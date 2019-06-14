import numpy as np
from .planar_graph_constructor import PlanarGraphConstructor
from .triangulator import Triangulator
from .planar_graph import PlanarGraph
from .planar_graph_edges import PlanarGraphEdges
from .. import common_utils


class PlanarGraphGenerator:
    """
    A static class for planar graph generation.
    """

    @staticmethod
    def generate_random_tree(size, random_vertex_costs=False):
        """
        O(size) algorithm for random tree generation.

        Parameters
        ----------
        size : int
            Size of the generated tree, >= 2.
        random_vertex_costs : boolean, optional
            If set to `True`, vertex costs are generated uniformly from [0, 1] and then normalized.
            If set to `False`, vertex costs are set to 1/size.
            Default is `False`.

        Returns
        -------
        PlanarGraph
            Result tree embedded into a plane.

        Notes
        -----
        To generate a random tree, the algorithm starts with two vertices connected by edge. Then on
        each step a random vertex is chosen and a new leaf is attached to it next to the edge
        indicated by current `incident_edge_example_indices` array value. This is repeated until a
        required tree size is reached.
        """

        if size < 2:
            raise RuntimeError('The minimum size of 2 is allowed.')

        if random_vertex_costs:
            # Hacks to make numba work
            vertex_costs = np.random.uniform(size=size)
        else:
            vertex_costs = np.ones(size)

        vertex_costs /= vertex_costs.sum()

        edges = PlanarGraphEdges(size - 1)

        edges.append(0, 1)
        edges.set_next_edge(0, 0, 0)
        edges.set_next_edge(0, 1, 0)

        incident_edge_example_indices = np.zeros(size, dtype=int)

        for index in range(size - 2):

            vertex = np.random.choice(index + 2)
            new_vertex = index + 2

            incident_edge_index = incident_edge_example_indices[vertex]

            new_edge_index = edges.size

            edges.append(vertex, new_vertex)

            edges.set_next_edge(new_edge_index, new_vertex, new_edge_index)

            incident_edge_example_indices[new_vertex] = new_edge_index

            incident_edge_next_edge_index = edges.get_next_edge_index(incident_edge_index, vertex)

            edges.set_next_edge(incident_edge_index, vertex, new_edge_index)
            edges.set_next_edge(new_edge_index, vertex, incident_edge_next_edge_index)

        return PlanarGraph(vertex_costs, incident_edge_example_indices, edges)


    @staticmethod
    def generate_random_graph(size, density, random_vertex_costs=False):
        """
        O(size) algorithm for random normal planar graph generation.

        Parameters
        ----------
        size : int
            Size of the generated graph, >= 2.
        density : float
            A value from [0, 1]. The result number of edges in the generated graph will be
            approximately density*(3*size - 6).
        random_vertex_costs : boolean, optional
            If set to `True`, vertex costs are generated uniformly from [0, 1] and then normalized.
            If set to `False`, vertex costs are set to 1/size.
            Default is `False`.

        Returns
        -------
        PlanarGraph
            Result planar graph.

        Notes
        -----
        To generate a random planar graph, a random tree of the same size is generated first. It is
        triangulated then. Then a random (1 - density) portion of edges is deleted from graph.
        
        Since the triangulation algorithm can produce multiple edges, some edges are deleted fulfill
        normality of the graph. Expiments show that there are not so many of such edges, so density
        is assumed to be preserved.
        """

        tree = PlanarGraphGenerator.generate_random_tree(size, random_vertex_costs)

        _, triangulated_tree = Triangulator.triangulate(tree)

        edges_to_leave_count = int(density*triangulated_tree.edges_count)
        edges_to_delete_count = triangulated_tree.edges_count - edges_to_leave_count

        random_edges_mask = np.concatenate((common_utils.repeat_bool(True, edges_to_leave_count),
                common_utils.repeat_bool(False, edges_to_delete_count)))

        np.random.shuffle(random_edges_mask)

        _, _, graph = PlanarGraphConstructor.construct_subgraph(triangulated_tree,
                common_utils.repeat_bool(True, triangulated_tree.size), random_edges_mask)

        _, _, graph = PlanarGraphConstructor.remove_double_edges(graph)

        graph.vertex_costs /= graph.vertex_costs.sum()

        return graph

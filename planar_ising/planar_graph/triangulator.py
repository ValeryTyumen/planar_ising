import numpy as np
from numba import jit
from numba.types import void, Tuple, int32, boolean
from . import planar_graph_constructor, search_utils
from .planar_graph import PlanarGraph, planar_graph_nb_type
from .planar_graph_edges import planar_graph_edges_nb_type, PlanarGraphEdges
from .. import common_utils


class Triangulator:
    """
    A static class for planar graph triangulation.
    """

    @staticmethod
    def triangulate(graph):
        """
        Linear algorithm for planar graph triangulation.

        Parameters
        ----------
        graph : PlanarGraph
            A graph to triangulated. It is assumed that graph.size >= 3 and there are no
            two-edge faces in it.

        Returns
        -------
        new_edge_indices_mapping : array_like, int32
            Mapping from `graph` edge indices to corresponding triangulated edge indices.
        triangulated_graph : PlanarGraph
            Triangulated graph. The input graph is unaffected.

        Notes
        -----
        The algorithm is borrowed from N. N. Schraudolph and D. Kamenetsky
        "Efficient exact inference in Planar Ising Models", Advances in Neural Information
        Processing Systems 21, pp. 1417-1424, Curran Associates, Inc., 2009

        The algorithm can violate normality of the graph.
        """

        return triangulate(graph)


@jit(void(planar_graph_nb_type), nopython=True)
def _add_edges_to_connect_graph_components(graph):

    component_indices = search_utils.color_connected_components(graph)

    components_count = component_indices.max() + 1
    component_example_vertices = common_utils.repeat_int(-1, components_count)

    for vertex, component_index in enumerate(component_indices):
        component_example_vertices[component_index] = vertex

    for component_index in range(components_count - 1):

        vertex1 = component_example_vertices[component_index]
        vertex2 = component_example_vertices[component_index + 1]

        edge_index = graph.edges_count
        graph.edges.append(vertex1, vertex2)

        for vertex in [vertex1, vertex2]:

            vertex_edge_index = graph.incident_edge_example_indices[vertex]

            if vertex_edge_index == -1:
                graph.incident_edge_example_indices[vertex] = edge_index
                graph.edges.set_next_edge(edge_index, vertex, edge_index)
            else:
                vertex_next_edge_index = graph.edges.get_next_edge_index(vertex_edge_index, vertex)

                graph.edges.set_next_edge(vertex_edge_index, vertex, edge_index)
                graph.edges.set_next_edge(edge_index, vertex, vertex_next_edge_index)

@jit(Tuple((int32[:], int32[:]))(planar_graph_nb_type, int32, int32, int32), nopython=True)
def _get_consequtive_face_vertices_and_edge_indices(graph, start_vertex, start_edge_index,
        vertices_in_sequence):

    vertices = [start_vertex, graph.edges.get_opposite_vertex(start_edge_index, start_vertex)]
    edge_indices = [start_edge_index]

    for _ in range(vertices_in_sequence - 2):

        edge_index_to_add = graph.edges.get_next_edge_index(edge_indices[-1], vertices[-1])
        edge_indices.append(edge_index_to_add)
        vertices.append(graph.edges.get_opposite_vertex(edge_indices[-1], vertices[-1]))

    return np.array(vertices), np.array(edge_indices)

@jit(void(planar_graph_nb_type, int32, int32, int32, int32, int32), nopython=True)
def _insert_edge(graph, triangle_vertex1, triangle_vertex2, triangle_vertex3, triangle_edge_index1,
        triangle_edge_index2):

    new_edge_index = graph.edges_count
    graph.edges.append(triangle_vertex1, triangle_vertex3)

    triangle_edge2_next_edge_index = graph.edges.get_next_edge_index(triangle_edge_index2,
            triangle_vertex3)
    graph.edges.set_next_edge(new_edge_index, triangle_vertex3, triangle_edge2_next_edge_index)

    graph.edges.set_next_edge(triangle_edge_index2, triangle_vertex3, new_edge_index)

    triangle_edge1_previous_edge_index = \
            graph.edges.get_previous_edge_index(triangle_edge_index1, triangle_vertex1)
    graph.edges.set_next_edge(triangle_edge1_previous_edge_index, triangle_vertex1, new_edge_index)

    graph.edges.set_next_edge(new_edge_index, triangle_vertex1, triangle_edge_index1)

@jit(Tuple((int32[:], planar_graph_nb_type))(planar_graph_nb_type), nopython=True)
def triangulate(graph):
    """
    Linear algorithm for planar graph triangulation.

    Parameters
    ----------
    graph : PlanarGraph
        A graph to triangulated. It is assumed that graph.size >= 3 and there are no
        two-edge faces in it.

    Returns
    -------
    new_edge_indices_mapping : array_like, int32
        Mapping from `graph` edge indices to corresponding triangulated edge indices.
    triangulated_graph : PlanarGraph
        Triangulated graph. The input graph is unaffected.

    Notes
    -----
    The algorithm is borrowed from N. N. Schraudolph and D. Kamenetsky
    "Efficient exact inference in Planar Ising Models", Advances in Neural Information
    Processing Systems 21, pp. 1417-1424, Curran Associates, Inc., 2009

    The algorithm can violate normality of the graph.
    """

    graph = planar_graph_constructor.clone_graph(graph)

    initial_graph_edges_count = graph.edges_count

    graph.edges.increase_capacity(graph.size*3 - 6)

    _add_edges_to_connect_graph_components(graph)

    for vertex in range(graph.size):
        for edge_index in graph.get_incident_edge_indices(vertex):

            face_vertices, face_edge_indices = \
                    _get_consequtive_face_vertices_and_edge_indices(graph, vertex, edge_index, 5)

            while face_vertices[0] != face_vertices[3] or face_vertices[1] != face_vertices[4]:

                if face_vertices[0] == face_vertices[2]:

                    face_vertices, face_edge_indices = \
                            _get_consequtive_face_vertices_and_edge_indices(graph,
                            face_vertices[1], face_edge_indices[1], 5)

                _insert_edge(graph, face_vertices[0], face_vertices[1], face_vertices[2],
                        face_edge_indices[0], face_edge_indices[1])

                face_vertices, face_edge_indices = \
                        _get_consequtive_face_vertices_and_edge_indices(graph, face_vertices[2],
                        face_edge_indices[2], 5)

    new_edge_indices_mapping = np.arange(initial_graph_edges_count)

    return new_edge_indices_mapping, graph

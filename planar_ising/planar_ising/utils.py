import numpy as np
from numba import jit
from numba.types import void, Tuple, int32, boolean
from .planar_ising_model import planar_ising_model_nb_type, PlanarIsingModel
from ..planar_graph import planar_graph_nb_type, PlanarGraph, triangulator


@jit(planar_graph_nb_type(planar_graph_nb_type), nopython=True)
def normalize_vertex_costs(graph):

    vertex_costs = graph.vertex_costs.copy()

    vertex_costs /= vertex_costs.sum()

    return PlanarGraph(vertex_costs, graph.incident_edge_example_indices, graph.edges)

@jit(int32[:](int32[:], int32), nopython=True)
def get_inverse_sub_mapping(sub_mapping, sub_elements_count):

    sub_element_exists_mask = (sub_mapping != -1)

    inverse_mapping = np.zeros(sub_elements_count, dtype=np.int32)
    inverse_mapping[sub_mapping[sub_element_exists_mask]] = \
            np.where(sub_element_exists_mask)[0].astype(np.int32)

    return inverse_mapping

@jit(Tuple((int32[:], planar_ising_model_nb_type))(planar_ising_model_nb_type), nopython=True)
def triangulate_ising_model(ising_model):

    graph = ising_model.graph
    interaction_values = ising_model.interaction_values

    new_edge_indices_mapping, new_graph = triangulator.triangulate(graph)

    new_interaction_values = np.zeros(new_graph.edges_count, dtype=np.float64)
    new_interaction_values[new_edge_indices_mapping] = interaction_values

    return new_edge_indices_mapping, PlanarIsingModel(new_graph, new_interaction_values)


import numpy as np
from .planar_ising_model import PlanarIsingModel
from ..planar_graph import PlanarGraph, Triangulator


def get_inverse_sub_mapping(sub_mapping, sub_elements_count):

    sub_element_exists_mask = (sub_mapping != -1)

    inverse_mapping = -np.ones(sub_elements_count, dtype=int)
    inverse_mapping[sub_mapping[sub_element_exists_mask]] = np.where(sub_element_exists_mask)[0]

    return inverse_mapping

def triangulate_ising_model(ising_model):

    graph = ising_model.graph
    interaction_values = ising_model.interaction_values

    new_edge_indices_mapping, new_graph = Triangulator.triangulate(graph)

    new_interaction_values = np.zeros(new_graph.edges_count)
    new_interaction_values[new_edge_indices_mapping] = interaction_values

    return new_edge_indices_mapping, PlanarIsingModel(new_graph, new_interaction_values)

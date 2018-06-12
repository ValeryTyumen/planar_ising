import numpy as np
from lipton_tarjan import PlanarGraphGenerator
from .planar_ising_model import PlanarIsingModel


class PlanarIsingModelGenerator:

    @staticmethod
    def generate_random_model(size, graph_density, interaction_values_std):

        graph = PlanarGraphGenerator.generate_random_graph(size, graph_density)

        interaction_values = np.random.normal(scale=interaction_values_std, size=graph.edges_count)

        return PlanarIsingModel(graph, interaction_values)

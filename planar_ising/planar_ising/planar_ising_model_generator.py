import numpy as np
from .planar_ising_model import PlanarIsingModel
from ..planar_graph import PlanarGraphGenerator


class PlanarIsingModelGenerator:
    """
    Static class for Planar Zero-Field Ising Model random generation. 
    """

    @staticmethod
    def generate_random_model(size, graph_density, interaction_values_std):
        """
        Random model generation. Topology is generated in `lipton_tarjan.PlanarGraphGenerator`
        class, interaction values are drawn from zero-mean Gaussian distribution with a given std.

        Parameters
        ----------
        size : int
            Size of the generated model, size >= 2.
        graph_density : float
            A value from [0, 1]. The result number of edges in the generated topology will be
            approximately density*(3*size - 6).
        interaction_values_std : float 
            Std of interaction values.

        Notes
        -----
        See `lipton_tarjan.PlanarGraphGenerator` docstrings for details on how planar topology is
        generated.
        """

        graph = PlanarGraphGenerator.generate_random_graph(size, graph_density)

        interaction_values = np.random.normal(scale=interaction_values_std, size=graph.edges_count)

        return PlanarIsingModel(graph, interaction_values)

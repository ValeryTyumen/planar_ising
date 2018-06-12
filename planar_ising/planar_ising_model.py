from numba import jitclass
from numba.types import float64
from lipton_tarjan import planar_graph_nb_type


@jitclass([('_graph', planar_graph_nb_type), ('_interaction_values', float64[:])])
class PlanarIsingModel:

    def __init__(self, graph, interaction_values):

        if graph.edges_count != len(interaction_values):
            raise RuntimeError('Each interaction corresponds to graph\'s edge.')

        self._graph = graph
        self._interaction_values = interaction_values

    @property
    def graph(self):

        return self._graph

    @property
    def interaction_values(self):

        return self._interaction_values

    def get_minus_energy(self, spin_values):

        return (self._interaction_values*spin_values[self._graph.edges.vertex1]*\
                spin_values[self._graph.edges.vertex2]).sum()

planar_ising_model_nb_type = PlanarIsingModel.class_type.instance_type
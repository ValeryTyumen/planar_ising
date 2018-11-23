from numba import jitclass
from numba.types import int32, float64
from .. import common_utils
from ..planar_graph import planar_graph_nb_type


@jitclass([('_graph', planar_graph_nb_type), ('_interaction_values', float64[:])])
class PlanarIsingModel:
    """
    A data structure representing planar zero-field Ising model.
    """

    def __init__(self, graph, interaction_values):
        """
        Model initialization.

        Parameters
        ----------
        graph : PlanarGraph
            Topology of the model.
        interaction_values : array_like, float64
            Interaction values assigned to each edge of `graph`.

        Notes
        -----
        See `lipton_tarjan` module docstrings and presentation notebook for details on `PlanarGraph`
        class.
        """

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
        """
        Get log-weight of spin configuration.

        Parameters
        ----------
        spin_values : array_like, int32
            Spin values of the configuration in { -1, +1 }.

        Returns
        -------
        float
            Log-weight of configuration (minus-energy).
        """

        return (self._interaction_values*spin_values[self._graph.edges.vertex1]*\
                spin_values[self._graph.edges.vertex2]).sum()


planar_ising_model_nb_type = common_utils.get_numba_type(PlanarIsingModel)

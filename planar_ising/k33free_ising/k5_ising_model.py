import numpy as np


class K5IsingModel:
    """
    A data structure representing K5 zero-field Ising model.
    """

    EDGE_LIST = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4]
    ])

    def __init__(self, interaction_values):
        """
        Model initialization.

        Parameters
        ----------
        interaction_values : array_like, float64
            10 interaction values assigned to each edge (see `K5IsingModel.EDGE_LIST`).
        """

        self._interaction_values = interaction_values

    @property
    def interaction_values(self):

        return self._interaction_values

    def get_minus_energy(self, spin_values):
        """
        Get log-weight of spin configuration.

        Parameters
        ----------
        spin_values : array_like, int32
            5 spin values of the configuration in { -1, +1 }.

        Returns
        -------
        float
            Log-weight of configuration (minus-energy).
        """

        return (self._interaction_values*spin_values[self.EDGE_LIST[:, 0]]*\
                spin_values[self.EDGE_LIST[:, 1]]).sum()

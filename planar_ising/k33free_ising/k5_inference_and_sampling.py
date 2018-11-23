import numpy as np
from .k5_ising_model import K5IsingModel


class K5InferenceAndSampling:

    def __init__(self, ising_model):

        self._ising_model = ising_model
        self._is_prepared_for_sampling = False

    def _get_configuration(self, configuration_index):

        configuration = np.zeros(5, dtype=int)

        for spin_index in range(5):
            configuration[spin_index] = 2*((configuration_index >> spin_index) & 1) - 1

        return configuration

    def _iterate_configurations_and_weights(self):

        for configuration_index in range(32):

            configuration = self._get_configuration(configuration_index)
            minus_energy = self._ising_model.get_minus_energy(configuration)

            yield (configuration, np.exp(minus_energy))

    def _compute_constrained_log_partition_function(self, edge_index):

        partition_function = 0.0

        if edge_index != -1:
            vertex1, vertex2 = K5IsingModel.EDGE_LIST[edge_index]

        for configuration, weight in self._iterate_configurations_and_weights():
            if edge_index == -1 or configuration[vertex1] != configuration[vertex2]:
                partition_function += weight

        return np.log(partition_function)

    def compute_log_partition_function(self):

        return self._compute_constrained_log_partition_function(-1)

    def compute_constrained_log_partition_function(self, edge_index):

        return self._compute_constrained_log_partition_function(edge_index)

    def _prepare_for_constrained_sampling(self, edge_index, are_equal):

        probabilities = []

        vertex1, vertex2 = K5IsingModel.EDGE_LIST[edge_index]

        for configuration, weight in self._iterate_configurations_and_weights():

            if edge_index == -1 or (are_equal and configuration[vertex1] == \
                    configuration[vertex2]) or ((not are_equal) and configuration[vertex1] != \
                    configuration[vertex2]):
                probabilities.append(weight)
            else:
                probabilities.append(0)

        self._probabilities = np.array(probabilities)
        self._probabilities /= self._probabilities.sum()

    def prepare_for_sampling(self):

        self._prepare_for_constrained_sampling(-1, None)
        self._is_prepared_for_sampling = True

    def prepare_for_constrained_sampling(self, edge_index, are_equal):

        self._prepare_for_constrained_sampling(edge_index, are_equal)
        self._is_prepared_for_sampling = True

    def sample_spin_configurations(self, sample_size):

        if not self._is_prepared_for_sampling:
            raise RuntimeError('Not prepared for sampling.')

        configuration_indices = np.random.choice(32, p=self._probabilities, size=sample_size)

        return np.array([self._get_configuration(index) for index in configuration_indices])

import numpy as np
from .k5_ising_model import K5IsingModel
from .k5_inference_and_sampling import K5InferenceAndSampling
from ..planar_ising import PlanarIsingModel, WilsonInferenceAndSampling
from . import utils


class K33freeInferenceAndSampling:
    """
    A class performing inference/sampling in K33-free zero-field Ising models.
    """

    def __init__(self, ising_model):
        """
        Initialization.

        Parameters
        ----------
        ising_model : K33freeIsingModel
            Input model.
        """

        self._ising_model = ising_model
        self._is_prepared_for_sampling = False

    def _process_inference_in_node(self, node_index, parent_index):

        aggregated_model_interaction_values = \
                np.copy(self._ising_model.nodes[node_index].interaction_values)

        result_addition = 0.0

        parent_virtual_edge_index = -1

        for virtual_edge_index, adjacent_node_index in \
                zip(self._ising_model.virtual_edge_indices[node_index],
                self._ising_model.adjacent_node_indices[node_index]):

            if adjacent_node_index != parent_index:

                child_log_pi_function = self._process_inference_in_node(adjacent_node_index,
                        node_index)

                result_addition += child_log_pi_function.mean()

                interaction_value_addition = (child_log_pi_function[0, 0] - \
                        child_log_pi_function[0, 1])/2

                aggregated_model_interaction_values[virtual_edge_index] += \
                        interaction_value_addition

            else:
                parent_virtual_edge_index = virtual_edge_index

        if self._ising_model.node_types[node_index] == 'k5':
            aggregated_model = K5IsingModel(aggregated_model_interaction_values)
            aggregated_inference = K5InferenceAndSampling(aggregated_model)
        else:

            aggregated_model = PlanarIsingModel(self._ising_model.nodes[node_index].graph, \
                    aggregated_model_interaction_values)
            aggregated_inference = WilsonInferenceAndSampling(aggregated_model)

        self._aggregated_inferences[node_index] = aggregated_inference

        log_partition_function = aggregated_inference.compute_log_partition_function()

        if parent_index == -1:
            return log_partition_function + result_addition

        log_pi_function = np.zeros((2, 2), dtype=float) + result_addition

        constrained_log_partition_function = \
                aggregated_inference.compute_constrained_log_partition_function(
                parent_virtual_edge_index)

        antidiagonal_addition = constrained_log_partition_function - np.log(2)

        log_pi_function[0, 1] += antidiagonal_addition
        log_pi_function[1, 0] += antidiagonal_addition

        diagonal_addition = np.log(np.exp(log_partition_function) - \
                np.exp(constrained_log_partition_function)) - np.log(2)

        log_pi_function[0, 0] += diagonal_addition
        log_pi_function[1, 1] += diagonal_addition

        return log_pi_function

    def compute_log_partition_function(self):
        """
        Log-partition function computation.

        Returns
        -------
        float
            Log partition function.
        """

        self._root = 0
        self._aggregated_inferences = [None]*self._ising_model.nodes_count

        self._is_prepared_for_sampling = True

        return self._process_inference_in_node(self._root, -1)

    def prepare_for_sampling(self):
        """
        Precompute data structures required for sampling. It is required
        to run this method before the first `sample_spin_configurations` method call.
        """

        self.compute_log_partition_function()
        self._is_prepared_for_sampling = True

    def _process_sampling_in_node(self, node_index, parent_virtual_edge_index, spin_values):

        inference_and_sampling = self._aggregated_inferences[node_index]
        model_vertices = self._ising_model.model_vertices[node_index]

        if parent_virtual_edge_index == -1:
            inference_and_sampling.prepare_for_sampling()
            node_spin_values = inference_and_sampling.sample_spin_configurations(1)[0]
        else:

            parent_virtual_edge_vertex1, parent_virtual_edge_vertex2 = \
                    utils.get_edge_vertices(self._ising_model, node_index,
                    parent_virtual_edge_index)
            parent_virtual_edge_spin1_value = \
                    spin_values[model_vertices[parent_virtual_edge_vertex1]]
            parent_virtual_edge_spin2_value = \
                    spin_values[model_vertices[parent_virtual_edge_vertex2]]

            spins_are_equal = (parent_virtual_edge_spin1_value == parent_virtual_edge_spin2_value)

            inference_and_sampling.prepare_for_constrained_sampling(parent_virtual_edge_index,
                    spins_are_equal)

            node_spin_values = inference_and_sampling.sample_spin_configurations(1)[0]

            if parent_virtual_edge_spin1_value != node_spin_values[parent_virtual_edge_vertex1]:
                node_spin_values = -node_spin_values

        spin_values[model_vertices] = node_spin_values

        for virtual_edge_index, adjacent_node_index, adjacent_node_virtual_edge_index in \
                zip(self._ising_model.virtual_edge_indices[node_index],
                self._ising_model.adjacent_node_indices[node_index],
                self._ising_model.adjacent_node_virtual_edge_indices[node_index]):

            if virtual_edge_index != parent_virtual_edge_index:
                self._process_sampling_in_node(adjacent_node_index, adjacent_node_virtual_edge_index,
                        spin_values)

    def sample_spin_configurations(self, sample_size):
        """
        Draw a sample of spin configurations.

        Parameters
        ----------
        sample_size : int
            The size of a sample to be drawn.

        Returns
        -------
        array_like
            Array of shape `(sample_size, spins_count)` with spin configurations as rows.
        """

        if not self._is_prepared_for_sampling:
            raise RuntimeError('Not prepared for sampling.')

        spin_configurations = []

        for sample_index in range(sample_size):

            spin_values = np.zeros(self._ising_model.size, dtype=int)
            self._process_sampling_in_node(self._root, -1, spin_values)

            spin_configurations.append(spin_values)

        return np.asarray(spin_configurations)

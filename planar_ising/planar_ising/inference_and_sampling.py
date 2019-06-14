import numpy as np
from .planar_ising_model import PlanarIsingModel
from .partial_spin_configuration import PartialSpinConfiguration
from .graph_edges_mapping import GraphEdgesMapping
from .expanded_dual_graph_constructor import ExpandedDualGraphConstructor
from .nested_dissection import NestedDissection
from .kasteleyn_matrix_constructor import KasteleynMatrixConstructor
from .perfect_matching_collapser import PerfectMatchingCollapser
from . import utils
from .. import common_utils
from ..planar_graph import PlanarGraphEdges, PlanarGraph, PlanarGraphConstructor, Triangulator
from ..lipton_tarjan import PlanarSeparator, separation_class
from ..sparse_lu import CSRMatrix, SparseLU


class InferenceAndSampling:
    """
    A class performing Wilson inference/sampling in planar zero-field Ising models.
    """

    def __init__(self, ising_model):
        """
        Initialization.

        Parameters
        ----------
        ising_model : PlanarIsingModel
            Input model.
        """

        self._ising_model = ising_model

    def prepare(self, sampling=False):
        """
        Precompute data structures.
        """

        self._new_edge_indices_mapping, self._ising_model = \
                utils.triangulate_ising_model(self._ising_model)

        graph = self._ising_model.graph
        interaction_values = self._ising_model.interaction_values

        self._graph_edges_mapping, self._expanded_dual_graph = \
                ExpandedDualGraphConstructor.construct(graph)

        self._log_weights = ExpandedDualGraphConstructor.get_expanded_dual_graph_log_weights(
                interaction_values, self._graph_edges_mapping)

        self._kasteleyn_orientation = ExpandedDualGraphConstructor.get_kasteleyn_orientation(graph,
                self._graph_edges_mapping, self._expanded_dual_graph)

        spin_values = np.zeros(graph.size, dtype=int)

        perfect_matching_mask = \
                ExpandedDualGraphConstructor.get_expanded_dual_subgraph_perfect_matching(graph,
                self._graph_edges_mapping, spin_values)

        new_vertices_mapping, _, collapsed_graph = \
                PerfectMatchingCollapser.collapse_perfect_matching(self._expanded_dual_graph,
                perfect_matching_mask)

        pm_edges_permutation = NestedDissection.permute_vertices(collapsed_graph,
                np.array([separation_class.UNDEFINED]))

        vertices_permutation = InferenceAndSampling._get_vertices_permutation(
                self._expanded_dual_graph, perfect_matching_mask, pm_edges_permutation)

        self._k_weight_indices, self._k_weight_signs, self._k_column_indices, \
                self._k_row_first_element_indices = \
                KasteleynMatrixConstructor.construct_symbolically(self._expanded_dual_graph,
                self._kasteleyn_orientation, vertices_permutation)

        self._on_sampling = sampling

        if sampling:

            self._top_level_separation, self._dilated_top_level_separator_vertices, \
                    self._separator_vertices_in_dilated_separator_mask, \
                    self._inverse_lower_right_kasteleyn_submatrix = \
                    InferenceAndSampling._prepare_data_for_sampling_in_subgraph(graph,
                    self._graph_edges_mapping, self._expanded_dual_graph, self._log_weights,
                    self._kasteleyn_orientation, spin_values)

    def register_new_interactions(self, interaction_values):
        """
        Update interaction values without a lot of recomputations.

        Only supports inference.
        """

        triangulated_interaction_values = np.zeros(self._ising_model.graph.edges_count)
        triangulated_interaction_values[self._new_edge_indices_mapping] = interaction_values

        self._ising_model.interaction_values = triangulated_interaction_values

        self._log_weights = ExpandedDualGraphConstructor.get_expanded_dual_graph_log_weights(
                triangulated_interaction_values, self._graph_edges_mapping)

        if self._on_sampling:

            graph = self._ising_model.graph

            self._top_level_separation, self._dilated_top_level_separator_vertices, \
                    self._separator_vertices_in_dilated_separator_mask, \
                    self._inverse_lower_right_kasteleyn_submatrix = \
                    InferenceAndSampling._prepare_data_for_sampling_in_subgraph(graph,
                    self._graph_edges_mapping, self._expanded_dual_graph, self._log_weights,
                    self._kasteleyn_orientation, np.zeros(graph.size, dtype=int))

    def compute_logpf(self, with_marginals=False):
        """
        Log-partition function computation.

        Returns
        -------
        float
            Log partition function.
        """

        k_matrix = CSRMatrix(self._k_weight_signs, self._log_weights[self._k_weight_indices],
                self._k_column_indices, self._k_row_first_element_indices)

        l_matrix, u_matrix = SparseLU.factorize(k_matrix)

        logdet = u_matrix.logs[u_matrix.row_first_element_indices[:-1]].sum()

        logpf = logdet/2 + np.log(2) - self._ising_model.interaction_values.sum()

        if not with_marginals:
            return logpf

        logdet_grad_signs, logdet_grad_logs = SparseLU.get_logdet_grad(k_matrix, l_matrix, u_matrix)

        old_edge_indices_mapping = utils.get_inverse_sub_mapping(self._new_edge_indices_mapping,
                self._expanded_dual_graph.edges_count)

        k_weight_old_indices = old_edge_indices_mapping[self._k_weight_indices]

        logdet_grad_signs = logdet_grad_signs[k_weight_old_indices != -1]
        logdet_grad_logs = logdet_grad_logs[k_weight_old_indices != -1]
        k_weight_old_indices = k_weight_old_indices[k_weight_old_indices != -1]

        marginal_signs = np.zeros(self._new_edge_indices_mapping.shape[0])
        marginal_logs = -np.ones_like(marginal_signs)

        for index, sign, log in zip(k_weight_old_indices, logdet_grad_signs, logdet_grad_logs):

            if marginal_logs[index] == -1:
                marginal_signs[index] = sign
                marginal_logs[index] = log
            else:
                max_log = max(marginal_logs[index], log)
                result = marginal_signs[index]*np.exp(marginal_logs[index] - max_log) + \
                        sign*np.exp(log - max_log)

                if result < 1e-300:
                    marginal_signs[index] = 0
                    marginal_logs[index] = 0
                else:

                    marginal_signs[index] = np.sign(result)
                    marginal_logs[index] = np.log(np.absolute(result)) + max_log

        marginals = marginal_signs*np.exp(marginal_logs) - 1
        marginals = np.clip(marginals, -1, 1)

        return logpf, marginals

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

        graph = self._ising_model.graph

        result = np.zeros((sample_size, graph.size), dtype=int)

        for sample_index in range(sample_size):

            partial_spin_configuration = PartialSpinConfiguration(graph)

            InferenceAndSampling._sample_perfect_matching_in_subgraph(graph,
                    self._graph_edges_mapping, self._expanded_dual_graph, self._log_weights,
                    self._kasteleyn_orientation, partial_spin_configuration,
                    self._top_level_separation, self._dilated_top_level_separator_vertices,
                    self._separator_vertices_in_dilated_separator_mask,
                    self._inverse_lower_right_kasteleyn_submatrix)

            result[sample_index] = partial_spin_configuration.spin_values

            if np.random.rand() > 0.5:
                result[sample_index] *= -1

        return result

    @staticmethod
    def _prepare_data_for_sampling_in_subgraph(graph, graph_edges_mapping, expanded_dual_subgraph,
            log_weights, kasteleyn_orientation, spin_values):

        perfect_matching_mask = \
                ExpandedDualGraphConstructor.get_expanded_dual_subgraph_perfect_matching(graph,
                graph_edges_mapping, spin_values)

        new_vertices_mapping, _, collapsed_graph = \
                PerfectMatchingCollapser.collapse_perfect_matching(expanded_dual_subgraph,
                perfect_matching_mask)

        separation = PlanarSeparator.mark_separation(expanded_dual_subgraph)

        separator_touching_edges_mask = np.logical_or(
                separation[expanded_dual_subgraph.edges.vertex1] == separation_class.SEPARATOR,
                separation[expanded_dual_subgraph.edges.vertex2] == separation_class.SEPARATOR)
     
        dilated_separation = np.copy(separation)

        dilated_separation[expanded_dual_subgraph.edges.vertex1[separator_touching_edges_mask]] = \
                separation_class.SEPARATOR
        dilated_separation[expanded_dual_subgraph.edges.vertex2[separator_touching_edges_mask]] = \
                separation_class.SEPARATOR

        separator_touching_pm_edges_mask = np.logical_or(
                dilated_separation[expanded_dual_subgraph.edges.vertex1[perfect_matching_mask]] == \
                separation_class.SEPARATOR,
                dilated_separation[expanded_dual_subgraph.edges.vertex2[perfect_matching_mask]] == \
                separation_class.SEPARATOR)

        collapsed_graph_separation = \
                dilated_separation[expanded_dual_subgraph.edges.vertex1[perfect_matching_mask]]
        collapsed_graph_separation[separator_touching_pm_edges_mask] = separation_class.SEPARATOR

        dilated_top_level_separator_size = 2*(collapsed_graph_separation == \
                separation_class.SEPARATOR).sum()

        pm_edges_permutation = NestedDissection.permute_vertices(collapsed_graph,
                collapsed_graph_separation)

        vertices_permutation = InferenceAndSampling._get_vertices_permutation(
                expanded_dual_subgraph, perfect_matching_mask, pm_edges_permutation)

        k_weight_indices, k_weight_signs, k_column_indices, k_row_first_element_indices = \
                KasteleynMatrixConstructor.construct_symbolically(expanded_dual_subgraph,
                kasteleyn_orientation, vertices_permutation)

        k_matrix = CSRMatrix(k_weight_signs, log_weights[k_weight_indices], k_column_indices,
                k_row_first_element_indices)

        dilated_separator_start = expanded_dual_subgraph.size - dilated_top_level_separator_size

        dilated_top_level_separator_vertices = vertices_permutation[dilated_separator_start:]

        separator_vertices_in_dilated_separator_mask = (separation[vertices_permutation] == \
                separation_class.SEPARATOR)[dilated_separator_start:]

        inverse_lower_right_kasteleyn_submatrix = \
                KasteleynMatrixConstructor.get_inverse_lower_right_kasteleyn_submatrix(
                k_matrix, dilated_top_level_separator_size)

        return separation, dilated_top_level_separator_vertices, \
                separator_vertices_in_dilated_separator_mask, \
                inverse_lower_right_kasteleyn_submatrix

    @staticmethod
    def _get_vertices_permutation(expanded_dual_subgraph, perfect_matching_mask,
            pm_edges_permutation):

        vertices_permutation = np.zeros(expanded_dual_subgraph.size, dtype=int)
        vertices_permutation[::2] = expanded_dual_subgraph.edges.vertex1[perfect_matching_mask]\
                [pm_edges_permutation]
        vertices_permutation[1::2] = expanded_dual_subgraph.edges.vertex2[perfect_matching_mask]\
                [pm_edges_permutation]

        return vertices_permutation

    @staticmethod
    def _sample_perfect_matching_in_subgraph(graph, graph_edges_mapping, expanded_dual_subgraph,
            log_weights, kasteleyn_orientation, partial_spin_configuration, top_level_separation,
            dilated_top_level_separator_vertices, separator_vertices_in_dilated_separator_mask,
            inverse_lower_right_kasteleyn_submatrix):

        separator_matching_edge_indices = \
                KasteleynMatrixConstructor.draw_separator_matching_edge_indices(
                expanded_dual_subgraph, np.exp(log_weights),
                inverse_lower_right_kasteleyn_submatrix, dilated_top_level_separator_vertices,
                separator_vertices_in_dilated_separator_mask)

        for edge_index in separator_matching_edge_indices:

            if graph_edges_mapping.second[edge_index] == -1:
                partial_spin_configuration.set_spin_pair(graph_edges_mapping.first[edge_index], True)
            else:
                partial_spin_configuration.set_spin_pair(graph_edges_mapping.first[edge_index], False)
                partial_spin_configuration.set_spin_pair(graph_edges_mapping.second[edge_index], False)
     
        unsaturated_vertices_mask = common_utils.repeat_bool(True, expanded_dual_subgraph.size)

        unsaturated_vertices_mask[\
                expanded_dual_subgraph.edges.vertex1[separator_matching_edge_indices]] = False
        unsaturated_vertices_mask[\
                expanded_dual_subgraph.edges.vertex2[separator_matching_edge_indices]] = False

        subsubgraph_vertices_masks = (unsaturated_vertices_mask & \
                (top_level_separation == separation_class.FIRST_PART),
                unsaturated_vertices_mask & (top_level_separation == separation_class.SECOND_PART))

        subsubgraphs_edges_mask = common_utils.repeat_bool(True, expanded_dual_subgraph.edges_count)

        on_first_subsubgraph = True

        for subsubgraph_vertices_mask in subsubgraph_vertices_masks:

            if subsubgraph_vertices_mask.sum() == 0:
                on_first_subsubgraph = False
                continue

            new_vertices_mapping, new_edge_indices_mapping, subsubgraph = \
                    PlanarGraphConstructor.construct_subgraph(expanded_dual_subgraph,
                    subsubgraph_vertices_mask, subsubgraphs_edges_mask)

            subsubgraph.vertex_costs /= subsubgraph.vertex_costs.sum()

            old_vertices_mapping = utils.get_inverse_sub_mapping(new_vertices_mapping,
                    subsubgraph.size)

            old_edge_indices_mapping = utils.get_inverse_sub_mapping(new_edge_indices_mapping,
                    subsubgraph.edges_count)

            subsubgraph_graph_edges_mapping = \
                    GraphEdgesMapping(graph_edges_mapping.first[old_edge_indices_mapping],
                    graph_edges_mapping.second[old_edge_indices_mapping])

            subsubgraph_log_weights = log_weights[old_edge_indices_mapping]

            subsubgraph_kasteleyn_orientation = \
                    new_vertices_mapping[kasteleyn_orientation[old_edge_indices_mapping]]

            subsubgraph_top_level_separation, subsubgraph_dilated_top_level_separator_vertices, \
                    subsubgraph_separator_vertices_in_dilated_separator_mask, \
                    subsubgraph_inverse_lower_right_kasteleyn_submatrix = \
                    InferenceAndSampling._prepare_data_for_sampling_in_subgraph(graph,
                    subsubgraph_graph_edges_mapping, subsubgraph, subsubgraph_log_weights,
                    subsubgraph_kasteleyn_orientation, partial_spin_configuration.spin_values)

            InferenceAndSampling._sample_perfect_matching_in_subgraph(graph,
                    subsubgraph_graph_edges_mapping, subsubgraph, subsubgraph_log_weights,
                    subsubgraph_kasteleyn_orientation, partial_spin_configuration,
                    subsubgraph_top_level_separation,
                    subsubgraph_dilated_top_level_separator_vertices,
                    subsubgraph_separator_vertices_in_dilated_separator_mask,
                    subsubgraph_inverse_lower_right_kasteleyn_submatrix)

            on_first_subsubgraph = False

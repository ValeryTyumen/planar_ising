import numpy as np
from numba import jit
from numba.types import void, Tuple, int32, float64, boolean
from lipton_tarjan import planar_graph_nb_type, triangulator, planar_graph_constructor
from sparse_lu import sparse_lu
from .planar_ising_model import PlanarIsingModel, planar_ising_model_nb_type
from .nested_dissection_map import NestedDissectionMap, nested_dissection_map_nb_type
from .partial_spin_configuration import PartialSpinConfiguration, partial_spin_configuration_nb_type
from .graph_edges_mapping import GraphEdgesMapping, graph_edges_mapping_nb_type
from . import expanded_dual_graph_constructor, nested_dissection, kasteleyn_matrix_constructor, \
        utils


class IsingInferenceAndSampling:
    """
    A class performing inference/sampling in Planar Zero-Field Ising Models.
    """

    def __init__(self, ising_model):
        """
        Initialization.

        Parameters
        ----------
        ising_model : PlanarIsingModel
            Input model.
        """

        self._initial_model = ising_model
        self._is_prepared_for_sampling = False

    def prepare_for_sampling(self):
        """
        Precompute data structures (nested dissection, etc.) required for sampling. It is required
        to run this method before the first `sample_spin_configurations` method call.
        """

        self._new_edge_indices_mapping, self._ising_model, self._graph_edges_mapping, \
                self._expanded_dual_graph, self._weights, self._kasteleyn_orientation, \
                self._nested_dissection_map = _prepare_data(self._initial_model)

        spin_values = np.zeros(self._ising_model.graph.size, dtype=np.int32)

        self._dilated_top_level_separator_vertices, \
                self._separator_vertices_in_dilated_separator_mask, \
                self._inverse_lower_right_kasteleyn_submatrix = \
                _prepare_data_for_sampling_in_subgraph(self._ising_model.graph,
                self._graph_edges_mapping, self._expanded_dual_graph,
                self._weights, self._kasteleyn_orientation, self._nested_dissection_map,
                spin_values)

        self._is_prepared_for_sampling = True

    def compute_log_partition_function(self):
        """
        Log-partition function computation.
        """

        new_edge_indices_mapping, ising_model, graph_edges_mapping, expanded_dual_graph, weights, \
                kasteleyn_orientation, nested_dissection_map = _prepare_data(self._initial_model)

        return _compute_log_partition_function(ising_model, graph_edges_mapping,
                expanded_dual_graph, weights, kasteleyn_orientation, nested_dissection_map)

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
            Array of shape `(sample_size, spins_count)` with spin configuration as rows.
        """

        if not self._is_prepared_for_sampling:
            raise RuntimeError('Not prepared for sampling.')

        return _sample_spin_configurations(sample_size, self._ising_model.graph,
                self._graph_edges_mapping, self._expanded_dual_graph, self._weights,
                self._kasteleyn_orientation, self._nested_dissection_map,
                self._dilated_top_level_separator_vertices,
                self._separator_vertices_in_dilated_separator_mask,
                self._inverse_lower_right_kasteleyn_submatrix)


@jit(Tuple((int32[:], planar_ising_model_nb_type))(planar_ising_model_nb_type), nopython=True)
def _triangulate_ising_model(ising_model):

    graph = ising_model.graph
    interaction_values = ising_model.interaction_values

    new_edge_indices_mapping, new_graph = triangulator.triangulate(graph)

    new_interaction_values = np.zeros(new_graph.edges_count, dtype=np.float64)
    new_interaction_values[new_edge_indices_mapping] = interaction_values

    return new_edge_indices_mapping, PlanarIsingModel(new_graph, new_interaction_values)

@jit(Tuple((int32[:], planar_ising_model_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type,
        float64[:], int32[:], nested_dissection_map_nb_type))(planar_ising_model_nb_type),
        nopython=True)
def _prepare_data(ising_model):

    new_edge_indices_mapping, ising_model = _triangulate_ising_model(ising_model)

    graph = ising_model.graph
    interaction_values = ising_model.interaction_values

    graph_edges_mapping, expanded_dual_graph = expanded_dual_graph_constructor.construct(graph)

    weights = expanded_dual_graph_constructor.get_expanded_dual_graph_weights(interaction_values,
            graph_edges_mapping)

    kasteleyn_orientation = expanded_dual_graph_constructor.get_kasteleyn_orientation(graph,
            graph_edges_mapping, expanded_dual_graph)

    nested_dissection_map = nested_dissection.get_nested_dissection_map(expanded_dual_graph)

    return new_edge_indices_mapping, ising_model, graph_edges_mapping, expanded_dual_graph, \
            weights, kasteleyn_orientation, nested_dissection_map

@jit(float64(planar_ising_model_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type,
        float64[:], int32[:], nested_dissection_map_nb_type), nopython=True)
def _compute_log_partition_function(ising_model, graph_edges_mapping, expanded_dual_graph, weights,
        kasteleyn_orientation, nested_dissection_map):

    spin_values = np.zeros(ising_model.graph.size, dtype=np.int32)

    perfect_matching_mask = \
            expanded_dual_graph_constructor.get_expanded_dual_subgraph_perfect_matching(
            ising_model.graph, graph_edges_mapping, spin_values)

    vertices_permutation, _ = \
            nested_dissection.get_nested_dissection_permutation_and_top_level_separator_size(
            expanded_dual_graph, nested_dissection_map, perfect_matching_mask)

    kasteleyn_matrix = kasteleyn_matrix_constructor.construct(expanded_dual_graph, weights,
            kasteleyn_orientation, vertices_permutation)

    _, u_matrix = sparse_lu.factorize(kasteleyn_matrix)

    combinatorial_log_partition_function = \
            np.log(np.absolute(u_matrix.element_values[u_matrix.row_first_element_indices[:-1]]\
            )).sum()/2

    return combinatorial_log_partition_function + np.log(2) - ising_model.interaction_values.sum()

@jit(Tuple((int32[:], boolean[:], float64[:, :]))(planar_graph_nb_type, graph_edges_mapping_nb_type,
        planar_graph_nb_type, float64[:], int32[:], nested_dissection_map_nb_type, int32[:]),
        nopython=True)
def _prepare_data_for_sampling_in_subgraph(graph, graph_edges_mapping, expanded_dual_subgraph,
        weights, kasteleyn_orientation, nd_map, spin_values):

    perfect_matching_mask = \
            expanded_dual_graph_constructor.get_expanded_dual_subgraph_perfect_matching(graph,
            graph_edges_mapping, spin_values)

    dilated_nd_map = nested_dissection.add_neighbours_to_top_level_separator(expanded_dual_subgraph,
            nd_map)

    vertices_permutation, dilated_top_level_separator_size = \
            nested_dissection.get_nested_dissection_permutation_and_top_level_separator_size(
            expanded_dual_subgraph, dilated_nd_map, perfect_matching_mask)

    kasteleyn_matrix = kasteleyn_matrix_constructor.construct(expanded_dual_subgraph, weights,
            kasteleyn_orientation, vertices_permutation)

    dilated_top_level_separator_vertices = vertices_permutation[expanded_dual_subgraph.size - \
            dilated_top_level_separator_size:]

    top_level_separator_in_order_index = np.where(nd_map.in_order_pre_order_mapping == 0)[0][0]

    separator_vertices_in_dilated_separator_mask = \
            (nd_map.in_order_map[dilated_top_level_separator_vertices] == \
            top_level_separator_in_order_index)

    inverse_lower_right_kasteleyn_submatrix = \
            kasteleyn_matrix_constructor.get_inverse_lower_right_kasteleyn_submatrix(
            kasteleyn_matrix, dilated_top_level_separator_size)

    return dilated_top_level_separator_vertices, separator_vertices_in_dilated_separator_mask, \
            inverse_lower_right_kasteleyn_submatrix

@jit(void(planar_graph_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type, float64[:],
        int32[:], nested_dissection_map_nb_type, partial_spin_configuration_nb_type, int32[:],
        boolean[:], float64[:, :]), nopython=True)
def _sample_perfect_matching_in_subgraph(graph, graph_edges_mapping, expanded_dual_subgraph,
        weights, kasteleyn_orientation, nested_dissection_map, partial_spin_configuration,
        dilated_top_level_separator_vertices, separator_vertices_in_dilated_separator_mask,
        inverse_lower_right_kasteleyn_submatrix):

    separator_matching_edge_indices = \
            kasteleyn_matrix_constructor.draw_separator_matching_edge_indices(
            expanded_dual_subgraph, weights, inverse_lower_right_kasteleyn_submatrix,
            dilated_top_level_separator_vertices, separator_vertices_in_dilated_separator_mask)

    for edge_index in separator_matching_edge_indices:

        if graph_edges_mapping.second[edge_index] == -1:
            partial_spin_configuration.set_spin_pair(graph_edges_mapping.first[edge_index], True)
        else:
            partial_spin_configuration.set_spin_pair(graph_edges_mapping.first[edge_index], False)
            partial_spin_configuration.set_spin_pair(graph_edges_mapping.second[edge_index], False)
 
    unsaturated_vertices_mask = utils.repeat_bool(True, expanded_dual_subgraph.size)

    unsaturated_vertices_mask[\
            expanded_dual_subgraph.edges.vertex1[separator_matching_edge_indices]] = False
    unsaturated_vertices_mask[\
            expanded_dual_subgraph.edges.vertex2[separator_matching_edge_indices]] = False

    top_level_separator_in_order_index = \
            np.where(nested_dissection_map.in_order_pre_order_mapping == 0)[0][0]

    subsubgraph_vertices_masks = (np.logical_and(unsaturated_vertices_mask,
            nested_dissection_map.in_order_map < top_level_separator_in_order_index),
            np.logical_and(unsaturated_vertices_mask,
            nested_dissection_map.in_order_map > top_level_separator_in_order_index))

    subsubgraphs_edges_mask = utils.repeat_bool(True, expanded_dual_subgraph.edges_count)

    on_first_subsubgraph = True

    for subsubgraph_vertices_mask in subsubgraph_vertices_masks:

        if subsubgraph_vertices_mask.sum() == 0:
            on_first_subsubgraph = False
            continue

        new_vertices_mapping, new_edge_indices_mapping, subsubgraph = \
                planar_graph_constructor.construct_subgraph(expanded_dual_subgraph,
                subsubgraph_vertices_mask, subsubgraphs_edges_mask)

        old_vertices_mapping = utils.get_inverse_sub_mapping(new_vertices_mapping, subsubgraph.size)
        old_edge_indices_mapping = utils.get_inverse_sub_mapping(new_edge_indices_mapping,
                subsubgraph.edges_count)

        subsubgraph_graph_edges_mapping = \
                GraphEdgesMapping(graph_edges_mapping.first[old_edge_indices_mapping],
                graph_edges_mapping.second[old_edge_indices_mapping])

        subsubgraph_weights = weights[old_edge_indices_mapping]

        subsubgraph_kasteleyn_orientation = \
                new_vertices_mapping[kasteleyn_orientation[old_edge_indices_mapping]]

        subsubgraph_in_order_map = \
                nested_dissection_map.in_order_map[old_vertices_mapping].copy()

        if on_first_subsubgraph:

            subsubgraph_in_order_pre_order_mapping = \
                    nested_dissection_map.in_order_pre_order_mapping[:\
                    top_level_separator_in_order_index].copy()

        else:

            subsubgraph_in_order_map -= top_level_separator_in_order_index + 1

            subsubgraph_in_order_pre_order_mapping = \
                    nested_dissection_map.in_order_pre_order_mapping[\
                    top_level_separator_in_order_index + 1:].copy()

        pre_order_shift = subsubgraph_in_order_pre_order_mapping.min()

        subsubgraph_in_order_pre_order_mapping -= pre_order_shift

        subsubgraph_nested_dissection_map = NestedDissectionMap(subsubgraph_in_order_map,
                subsubgraph_in_order_pre_order_mapping)

        subsubgraph_dilated_top_level_separator_vertices, \
                subsubgraph_separator_vertices_in_dilated_separator_mask, \
                subsubgraph_inverse_lower_right_kasteleyn_submatrix = \
                _prepare_data_for_sampling_in_subgraph(graph, subsubgraph_graph_edges_mapping,
                subsubgraph, subsubgraph_weights, subsubgraph_kasteleyn_orientation,
                subsubgraph_nested_dissection_map, partial_spin_configuration.spin_values)

        _sample_perfect_matching_in_subgraph(graph, subsubgraph_graph_edges_mapping, subsubgraph,
                subsubgraph_weights, subsubgraph_kasteleyn_orientation,
                subsubgraph_nested_dissection_map, partial_spin_configuration,
                subsubgraph_dilated_top_level_separator_vertices,
                subsubgraph_separator_vertices_in_dilated_separator_mask,
                subsubgraph_inverse_lower_right_kasteleyn_submatrix)

        on_first_subsubgraph = False

@jit(int32[:, :](int32, planar_graph_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type,
        float64[:], int32[:], nested_dissection_map_nb_type, int32[:], boolean[:], float64[:, :]),
        nopython=True)
def _sample_spin_configurations(sample_size, graph, graph_edges_mapping, expanded_dual_graph,
        weights, kasteleyn_orientation, nested_dissection_map, dilated_top_level_separator_vertices,
        separator_vertices_in_dilated_separator_mask, inverse_lower_right_kasteleyn_submatrix):

    result = np.zeros((np.int64(sample_size), np.int64(graph.size)), dtype=np.int32)
 
    #TODO: parallelize

    for sample_index in range(sample_size):

        partial_spin_configuration = PartialSpinConfiguration(graph)

        _sample_perfect_matching_in_subgraph(graph, graph_edges_mapping, expanded_dual_graph,
                weights, kasteleyn_orientation, nested_dissection_map, partial_spin_configuration,
                dilated_top_level_separator_vertices, separator_vertices_in_dilated_separator_mask,
                inverse_lower_right_kasteleyn_submatrix)

        result[sample_index] = partial_spin_configuration.spin_values

        if np.random.rand() > 0.5:
            result[sample_index] *= -1

    return result

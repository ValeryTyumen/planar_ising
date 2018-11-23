import numpy as np
from numba import jit
from numba.types import void, Tuple, int32, float64, boolean
from .planar_ising_model import PlanarIsingModel, planar_ising_model_nb_type
from .partial_spin_configuration import PartialSpinConfiguration, partial_spin_configuration_nb_type
from .graph_edges_mapping import GraphEdgesMapping, graph_edges_mapping_nb_type
from . import expanded_dual_graph_constructor, nested_dissection_permutation, \
        kasteleyn_matrix_constructor, utils
from .. import common_utils
from ..planar_graph import PlanarGraphEdges, PlanarGraph, planar_graph_nb_type, \
        planar_graph_constructor, triangulator
from ..lipton_tarjan import planar_separator, separation_class
from ..sparse_lu import sparse_lu


class WilsonInferenceAndSampling:
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

        self._initial_model = ising_model
        self._is_prepared_for_sampling = False
 
    def _find_dual_edge_index(self, graph_edges_mapping, edge_index):

        return np.where((graph_edges_mapping.first == edge_index) & \
                (graph_edges_mapping.second == -1))[0]

    def _compute_constrained_log_partition_function(self, edge_index):

        new_edge_indices_mapping, ising_model, graph_edges_mapping, expanded_dual_graph, weights, \
                kasteleyn_orientation = _prepare_data(self._initial_model)
 
        spin_values = np.zeros(ising_model.graph.size, dtype=np.int32)

        if edge_index != -1:

            edge_index = new_edge_indices_mapping[edge_index]

            dual_edge_index = self._find_dual_edge_index(graph_edges_mapping, edge_index)
            weights[dual_edge_index] = 0
 
            vertex1 = ising_model.graph.edges.vertex1[edge_index]
            vertex2 = ising_model.graph.edges.vertex2[edge_index]

            spin_values[vertex1] = 1
            spin_values[vertex2] = -1

        return _compute_log_partition_function(ising_model, graph_edges_mapping,
                expanded_dual_graph, weights, kasteleyn_orientation, spin_values)

    def compute_log_partition_function(self):
        """
        Log-partition function computation.

        Returns
        -------
        float
            Log partition function.
        """

        return self._compute_constrained_log_partition_function(-1)

    def compute_constrained_log_partition_function(self, edge_index):
        """
        Find the weight of the event "spins incident to edge are equal/unequal".

        Parameters
        ----------
        edge_index : int
            Edge between spins defining the event.

        Returns
        -------
        float
            Log weight of the event.
        """

        return self._compute_constrained_log_partition_function(edge_index)

    def _prepare_for_constrained_sampling(self, edge_index, are_equal):

        self._condition_edge_index = edge_index
        self._are_equal_condition = are_equal

        new_edge_indices_mapping, self._ising_model, self._graph_edges_mapping, \
                self._expanded_dual_graph, self._weights, self._kasteleyn_orientation = \
                _prepare_data(self._initial_model)

        spin_values = np.zeros(self._ising_model.graph.size, dtype=np.int32)

        if edge_index != -1:

            edge_index = new_edge_indices_mapping[edge_index]

            dual_edge_index = self._find_dual_edge_index(self._graph_edges_mapping, edge_index)

            if are_equal:

                dual_edges = self._expanded_dual_graph.edges

                dual_vertex1 = dual_edges.vertex1[dual_edge_index]
                dual_vertex2 = dual_edges.vertex2[dual_edge_index]

                zero_out_mask = ((dual_edges.vertex1 == dual_vertex1) | \
                        (dual_edges.vertex1 == dual_vertex2) | \
                        (dual_edges.vertex2 == dual_vertex1) | \
                        (dual_edges.vertex2 == dual_vertex2))

                zero_out_mask[dual_edge_index] = False

                self._weights[zero_out_mask] = 0

            else:
                self._weights[dual_edge_index] = 0

            vertex1 = self._ising_model.graph.edges.vertex1[edge_index]
            vertex2 = self._ising_model.graph.edges.vertex2[edge_index]

            spin_values[vertex1] = 1

            if are_equal:
                spin_values[vertex2] = 1
            else:
                spin_values[vertex2] = -1

        self._top_level_separation, self._dilated_top_level_separator_vertices, \
                self._separator_vertices_in_dilated_separator_mask, \
                self._inverse_lower_right_kasteleyn_submatrix = \
                _prepare_data_for_sampling_in_subgraph(self._ising_model.graph,
                self._graph_edges_mapping, self._expanded_dual_graph,
                self._weights, self._kasteleyn_orientation, spin_values)

        self._is_prepared_for_sampling = True

    def prepare_for_sampling(self):
        """
        Precompute data structures required for sampling. It is required
        to run this method before the first `sample_spin_configurations` method call.
        """

        return self._prepare_for_constrained_sampling(-1, True)

    def prepare_for_constrained_sampling(self, edge_index, are_equal):
        """
        Precompute data structures required for sampling conditional on the event "spins incident to
        edge are equal/unequal". It is required to run this method before the first
        `sample_spin_configurations` method call.

        Parameters
        ----------
        edge_index : int
            Edge between spins defining the event.
        are_equal : boolean
            Whether the event is for equal or unequal spins.
        """

        return self._prepare_for_constrained_sampling(edge_index, are_equal)

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

        return _sample_spin_configurations(sample_size, self._ising_model.graph,
                self._graph_edges_mapping, self._expanded_dual_graph, self._weights,
                self._kasteleyn_orientation, self._top_level_separation,
                self._dilated_top_level_separator_vertices,
                self._separator_vertices_in_dilated_separator_mask,
                self._inverse_lower_right_kasteleyn_submatrix, self._condition_edge_index,
                self._are_equal_condition)


@jit(Tuple((int32[:], planar_ising_model_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type,
        float64[:], int32[:]))(planar_ising_model_nb_type),
        nopython=True)
def _prepare_data(ising_model):

    new_edge_indices_mapping, ising_model = utils.triangulate_ising_model(ising_model)

    graph = ising_model.graph
    interaction_values = ising_model.interaction_values

    graph_edges_mapping, expanded_dual_graph = expanded_dual_graph_constructor.construct(graph)

    weights = expanded_dual_graph_constructor.get_expanded_dual_graph_weights(interaction_values,
            graph_edges_mapping)

    kasteleyn_orientation = \
            expanded_dual_graph_constructor.get_kasteleyn_orientation(graph,
            graph_edges_mapping, expanded_dual_graph)

    return new_edge_indices_mapping, ising_model, graph_edges_mapping, expanded_dual_graph, \
            weights, kasteleyn_orientation

@jit(void(planar_graph_nb_type, int32), nopython=True)
def _iterate_edge_incidences(graph, edge_index):

    start_vertex = graph.edges.vertex1[edge_index]

    current_vertex = start_vertex
    current_edge_index = graph.edges.get_next_edge_index(edge_index, current_vertex)

    current_opposite_vertex = graph.edges.get_opposite_vertex(current_edge_index, current_vertex)

    while current_edge_index != edge_index or current_opposite_vertex != start_vertex:

        if current_edge_index == edge_index:
            current_vertex, current_opposite_vertex = current_opposite_vertex, current_vertex
        else:
            yield current_edge_index

        current_edge_index = graph.edges.get_next_edge_index(current_edge_index, current_vertex)
        current_opposite_vertex = graph.edges.get_opposite_vertex(current_edge_index,
                current_vertex)

@jit(Tuple((int32[:], int32[:], planar_graph_nb_type))(planar_graph_nb_type, boolean[:]),
        nopython=True)
def _collapse_perfect_matching(expanded_dual_subgraph, perfect_matching_mask):

    size = expanded_dual_subgraph.size//2

    vertex_costs = np.array([1/size]*size, dtype=np.float32)

    new_vertices_mapping = np.zeros(np.int64(expanded_dual_subgraph.size), dtype=np.int32)

    new_vertices_mapping[expanded_dual_subgraph.edges.vertex1[perfect_matching_mask]] = \
            np.arange(size).astype(np.int32)
    new_vertices_mapping[expanded_dual_subgraph.edges.vertex2[perfect_matching_mask]] = \
            np.arange(size).astype(np.int32)

    edges = PlanarGraphEdges(expanded_dual_subgraph.edges_count)

    new_edge_indices_mapping = np.array([-1]*expanded_dual_subgraph.edges_count, dtype=np.int32)

    for edge_index, is_new_vertex in enumerate(perfect_matching_mask):

        if is_new_vertex:

            new_vertex = new_vertices_mapping[expanded_dual_subgraph.edges.vertex1[edge_index]]

            previous_adjacent_vertex = -1
            previous_new_incident_edge_index = -1
            first_new_incident_edge_index = -1
            first_adjacent_vertex = -1

            single_new_incident_edge = True

            for incident_edge_index in _iterate_edge_incidences(expanded_dual_subgraph, edge_index):

                adjacent_vertex = expanded_dual_subgraph.edges.vertex1[incident_edge_index]

                if new_vertices_mapping[adjacent_vertex] == new_vertex:
                    adjacent_vertex = expanded_dual_subgraph.edges.vertex2[incident_edge_index]

                new_adjacent_vertex = new_vertices_mapping[adjacent_vertex]

                new_incident_edge_index = -1

                if new_edge_indices_mapping[incident_edge_index] == -1:

                    if previous_adjacent_vertex == -1 or new_vertices_mapping[adjacent_vertex] != \
                            new_vertices_mapping[previous_adjacent_vertex]:

                        if first_adjacent_vertex == -1 or new_vertices_mapping[adjacent_vertex] != \
                                new_vertices_mapping[first_adjacent_vertex]:

                            new_incident_edge_index = edges.size
                            edges.append(new_vertex, new_adjacent_vertex)

                            new_edge_indices_mapping[incident_edge_index] = new_incident_edge_index

                        else:

                            new_incident_edge_index = first_new_incident_edge_index

                            new_edge_indices_mapping[incident_edge_index] = \
                                first_new_incident_edge_index

                        if previous_new_incident_edge_index != -1:
                            edges.set_previous_edge(new_incident_edge_index, new_vertex,
                                    previous_new_incident_edge_index)

                    else:

                        new_incident_edge_index = previous_new_incident_edge_index

                        new_edge_indices_mapping[incident_edge_index] = \
                                previous_new_incident_edge_index

                else:

                    new_incident_edge_index = new_edge_indices_mapping[incident_edge_index]

                    if previous_new_incident_edge_index != -1 and new_incident_edge_index != \
                            previous_new_incident_edge_index:
                        edges.set_previous_edge(new_incident_edge_index, new_vertex,
                                previous_new_incident_edge_index)

                if first_new_incident_edge_index == -1:
                    first_new_incident_edge_index = new_incident_edge_index

                if new_incident_edge_index != first_new_incident_edge_index:
                    single_new_incident_edge = False

                previous_new_incident_edge_index = new_incident_edge_index

                if first_adjacent_vertex == -1:
                    first_adjacent_vertex = adjacent_vertex

                previous_adjacent_vertex = adjacent_vertex

            if single_new_incident_edge or (first_new_incident_edge_index != \
                    previous_new_incident_edge_index):
                # no vertices of degree 0, when at least one pm is present
                edges.set_previous_edge(first_new_incident_edge_index, new_vertex, \
                        previous_new_incident_edge_index)

    incident_edge_example_indices = np.array([-1]*size, dtype=np.int32)

    for new_edge_index in range(edges.size):
        incident_edge_example_indices[edges.vertex1[new_edge_index]] = new_edge_index
        incident_edge_example_indices[edges.vertex2[new_edge_index]] = new_edge_index

    return new_vertices_mapping, new_edge_indices_mapping, PlanarGraph(vertex_costs,
            incident_edge_example_indices, edges)

@jit(int32[:](planar_graph_nb_type, boolean[:], int32[:]), nopython=True)
def _get_vertices_permutation(expanded_dual_subgraph, perfect_matching_mask, pm_edges_permutation):

    vertices_permutation = np.zeros(expanded_dual_subgraph.size, dtype=np.int32)
    vertices_permutation[::2] = expanded_dual_subgraph.edges.vertex1[perfect_matching_mask]\
            [pm_edges_permutation]
    vertices_permutation[1::2] = expanded_dual_subgraph.edges.vertex2[perfect_matching_mask]\
            [pm_edges_permutation]

    return vertices_permutation

@jit(float64(planar_ising_model_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type,
        float64[:], int32[:], int32[:]), nopython=True)
def _compute_log_partition_function(ising_model, graph_edges_mapping, expanded_dual_graph, weights,
        kasteleyn_orientation, spin_values):

    perfect_matching_mask = \
            expanded_dual_graph_constructor.get_expanded_dual_subgraph_perfect_matching(
            ising_model.graph, graph_edges_mapping, spin_values)

    new_vertices_mapping, _, collapsed_graph = _collapse_perfect_matching(expanded_dual_graph,
            perfect_matching_mask)

    pm_edges_permutation = nested_dissection_permutation.permute_vertices(collapsed_graph,
            np.array([separation_class.UNDEFINED], dtype=np.int32))

    vertices_permutation = _get_vertices_permutation(expanded_dual_graph, perfect_matching_mask, \
            pm_edges_permutation)

    kasteleyn_matrix = kasteleyn_matrix_constructor.construct(expanded_dual_graph, weights,
            kasteleyn_orientation, vertices_permutation)

    _, u_matrix = sparse_lu.factorize(kasteleyn_matrix)

    combinatorial_log_partition_function = \
            np.log(np.absolute(u_matrix.element_values[u_matrix.row_first_element_indices[:-1]]\
            )).sum()/2

    return combinatorial_log_partition_function + np.log(2) - ising_model.interaction_values.sum()

@jit(Tuple((int32[:], int32[:], boolean[:], float64[:, :]))(planar_graph_nb_type,
        graph_edges_mapping_nb_type, planar_graph_nb_type, float64[:], int32[:], int32[:]),
        nopython=True)
def _prepare_data_for_sampling_in_subgraph(graph, graph_edges_mapping, expanded_dual_subgraph,
        weights, kasteleyn_orientation, spin_values):

    perfect_matching_mask = \
            expanded_dual_graph_constructor.get_expanded_dual_subgraph_perfect_matching(graph,
            graph_edges_mapping, spin_values)

    new_vertices_mapping, _, collapsed_graph = _collapse_perfect_matching(expanded_dual_subgraph,
            perfect_matching_mask)

    separation = planar_separator.mark_separation(expanded_dual_subgraph)

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

    pm_edges_permutation = nested_dissection_permutation.permute_vertices(collapsed_graph,
            collapsed_graph_separation)

    vertices_permutation = _get_vertices_permutation(expanded_dual_subgraph, \
            perfect_matching_mask, pm_edges_permutation)

    kasteleyn_matrix = kasteleyn_matrix_constructor.construct(expanded_dual_subgraph, weights,
            kasteleyn_orientation, vertices_permutation)

    dilated_separator_start = expanded_dual_subgraph.size - dilated_top_level_separator_size

    dilated_top_level_separator_vertices = vertices_permutation[dilated_separator_start:]

    separator_vertices_in_dilated_separator_mask = (separation[vertices_permutation] == \
            separation_class.SEPARATOR)[dilated_separator_start:]

    inverse_lower_right_kasteleyn_submatrix = \
            kasteleyn_matrix_constructor.get_inverse_lower_right_kasteleyn_submatrix(
            kasteleyn_matrix, dilated_top_level_separator_size)

    return separation, dilated_top_level_separator_vertices, \
            separator_vertices_in_dilated_separator_mask, inverse_lower_right_kasteleyn_submatrix

@jit(void(planar_graph_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type, float64[:],
        int32[:], partial_spin_configuration_nb_type, int32[:], int32[:], boolean[:], float64[:, :]),
        nopython=True)
def _sample_perfect_matching_in_subgraph(graph, graph_edges_mapping, expanded_dual_subgraph,
        weights, kasteleyn_orientation, partial_spin_configuration,
        top_level_separation, dilated_top_level_separator_vertices,
        separator_vertices_in_dilated_separator_mask, inverse_lower_right_kasteleyn_submatrix):

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
 
    unsaturated_vertices_mask = common_utils.repeat_bool(True, expanded_dual_subgraph.size)

    unsaturated_vertices_mask[\
            expanded_dual_subgraph.edges.vertex1[separator_matching_edge_indices]] = False
    unsaturated_vertices_mask[\
            expanded_dual_subgraph.edges.vertex2[separator_matching_edge_indices]] = False

    subsubgraph_vertices_masks = (np.logical_and(unsaturated_vertices_mask,
            top_level_separation == separation_class.FIRST_PART),
            np.logical_and(unsaturated_vertices_mask,
            top_level_separation == separation_class.SECOND_PART))

    subsubgraphs_edges_mask = common_utils.repeat_bool(True, expanded_dual_subgraph.edges_count)

    on_first_subsubgraph = True

    for subsubgraph_vertices_mask in subsubgraph_vertices_masks:

        if subsubgraph_vertices_mask.sum() == 0:
            on_first_subsubgraph = False
            continue

        new_vertices_mapping, new_edge_indices_mapping, subsubgraph = \
                planar_graph_constructor.construct_subgraph(expanded_dual_subgraph,
                subsubgraph_vertices_mask, subsubgraphs_edges_mask)

        subsubgraph = utils.normalize_vertex_costs(subsubgraph)

        old_vertices_mapping = utils.get_inverse_sub_mapping(new_vertices_mapping, subsubgraph.size)
        old_edge_indices_mapping = utils.get_inverse_sub_mapping(new_edge_indices_mapping,
                subsubgraph.edges_count)

        subsubgraph_graph_edges_mapping = \
                GraphEdgesMapping(graph_edges_mapping.first[old_edge_indices_mapping],
                graph_edges_mapping.second[old_edge_indices_mapping])

        subsubgraph_weights = weights[old_edge_indices_mapping]

        subsubgraph_kasteleyn_orientation = \
                new_vertices_mapping[kasteleyn_orientation[old_edge_indices_mapping]]

        subsubgraph_top_level_separation, subsubgraph_dilated_top_level_separator_vertices, \
                subsubgraph_separator_vertices_in_dilated_separator_mask, \
                subsubgraph_inverse_lower_right_kasteleyn_submatrix = \
                _prepare_data_for_sampling_in_subgraph(graph, subsubgraph_graph_edges_mapping,
                subsubgraph, subsubgraph_weights, subsubgraph_kasteleyn_orientation,
                partial_spin_configuration.spin_values)

        _sample_perfect_matching_in_subgraph(graph, subsubgraph_graph_edges_mapping, subsubgraph,
                subsubgraph_weights, subsubgraph_kasteleyn_orientation,
                partial_spin_configuration, subsubgraph_top_level_separation,
                subsubgraph_dilated_top_level_separator_vertices,
                subsubgraph_separator_vertices_in_dilated_separator_mask,
                subsubgraph_inverse_lower_right_kasteleyn_submatrix)

        on_first_subsubgraph = False

@jit(int32[:, :](int32, planar_graph_nb_type, graph_edges_mapping_nb_type, planar_graph_nb_type,
        float64[:], int32[:], int32[:], int32[:], boolean[:], float64[:, :], int32, boolean),
        nopython=True)
def _sample_spin_configurations(sample_size, graph, graph_edges_mapping, expanded_dual_graph,
        weights, kasteleyn_orientation, top_level_separation, dilated_top_level_separator_vertices,
        separator_vertices_in_dilated_separator_mask, inverse_lower_right_kasteleyn_submatrix,
        conditional_edge_index, are_equal_condition):

    result = np.zeros((np.int64(sample_size), np.int64(graph.size)), dtype=np.int32)

    for sample_index in range(sample_size):

        partial_spin_configuration = PartialSpinConfiguration(graph)

        if conditional_edge_index != -1:
            partial_spin_configuration.set_spin_pair(conditional_edge_index,
                    are_equal_condition)

        _sample_perfect_matching_in_subgraph(graph, graph_edges_mapping, expanded_dual_graph,
                weights, kasteleyn_orientation, partial_spin_configuration,
                top_level_separation, dilated_top_level_separator_vertices,
                separator_vertices_in_dilated_separator_mask,
                inverse_lower_right_kasteleyn_submatrix)

        result[sample_index] = partial_spin_configuration.spin_values

        if np.random.rand() > 0.5:
            result[sample_index] *= -1

    return result

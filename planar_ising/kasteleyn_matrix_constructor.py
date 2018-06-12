import numpy as np
from numba import jit
from numba.types import Tuple, int32, float64, boolean
from lipton_tarjan import planar_graph_nb_type
from sparse_lu import CSRMatrix, csr_matrix_nb_type, sparse_lu
from . import utils


@jit(int32[:](int32[:]), nopython=True)
def _get_column_vertex_indices(vertex_indices):

    return (vertex_indices + 1 - 2*(vertex_indices%2)).astype(np.int32)

@jit(int32(int32), nopython=True)
def _get_column_vertex_index(vertex_index):

    return np.int32(vertex_index + 1 - 2*(vertex_index%2))

@jit(csr_matrix_nb_type(planar_graph_nb_type, float64[:], int32[:], int32[:]), nopython=True)
def construct(expanded_dual_subgraph, weights, kasteleyn_orientation, vertices_permutation):

    vertex_positions = utils.get_inverse_sub_mapping(vertices_permutation,
            expanded_dual_subgraph.size)

    element_values = list(np.zeros(0, dtype=np.float64))
    element_column_indices = list(np.zeros(0, dtype=np.int32))
    row_first_element_indices = np.zeros(expanded_dual_subgraph.size + 1, dtype=np.int32)

    for vertex_index in range(expanded_dual_subgraph.size):

        vertex = vertices_permutation[vertex_index]

        incident_edge_indices_list = []
        adjacent_vertex_indices_list = []

        for edge_index in expanded_dual_subgraph.get_incident_edge_indices(vertex):
            incident_edge_indices_list.append(edge_index)

        for adjacent_vertex in expanded_dual_subgraph.get_adjacent_vertices(vertex):
            adjacent_vertex_indices_list.append(vertex_positions[adjacent_vertex])

        incident_edge_indices = np.array(incident_edge_indices_list)

        adjacent_vertex_indices = np.array(adjacent_vertex_indices_list)
        column_indices = _get_column_vertex_indices(adjacent_vertex_indices)

        # Since each vertex of the expanded dual subgraph is of degree <= 3,
        # sorting takes O(1) time

        column_indices_permutation = column_indices.argsort()
        column_indices = column_indices[column_indices_permutation]
        incident_edge_indices = incident_edge_indices[column_indices_permutation]

        row_element_values = weights[incident_edge_indices]
        row_element_values[kasteleyn_orientation[incident_edge_indices] == vertex] *= -1

        element_values += [element_value for element_value in row_element_values]
        element_column_indices += [column_index for column_index in column_indices]
        row_first_element_indices[vertex_index + 1] = len(element_values)

    return CSRMatrix(np.array(element_values), np.array(element_column_indices),
            row_first_element_indices)

@jit(float64[:, :](csr_matrix_nb_type, int32), nopython=True)
def get_inverse_lower_right_kasteleyn_submatrix(kasteleyn_matrix, submatrix_size):

    l_matrix, u_matrix = sparse_lu.factorize(kasteleyn_matrix)

    l_submatrix = sparse_lu.get_lower_right_submatrix(l_matrix, submatrix_size)
    u_submatrix = sparse_lu.get_lower_right_submatrix(u_matrix, submatrix_size)

    submatrix_size_int64 = np.int64(submatrix_size)

    inverse_kasteleyn_submatrix = np.zeros((submatrix_size_int64, submatrix_size_int64),
            dtype=np.float64)

    unit_vector = np.zeros(submatrix_size, dtype=np.float64)

    for index in range(submatrix_size):

        unit_vector[index] = 1

        inverse_kasteleyn_submatrix[:, index] = sparse_lu.solve(l_submatrix, u_submatrix,
                unit_vector)

        unit_vector[index] = 0

    return inverse_kasteleyn_submatrix

@jit(float64[:](float64[:, :], float64[:]), nopython=True)
def _solve_lower_triangular(matrix, right_hand_side):

    solution = np.zeros_like(right_hand_side)

    for row_index in range(matrix.shape[0]):

        solution[row_index] = (right_hand_side[row_index] - \
                (solution[:row_index]*matrix[row_index, :row_index]).sum())\
                /matrix[row_index, row_index]

    return solution

@jit(Tuple((float64[:], float64[:]))(float64[:, :], float64[:, :], float64[:, :], int32[:], int32),
        nopython=True)
def _get_next_l_matrix_row_and_u_matrix_column(current_l_matrix, current_u_matrix,
        inverse_lower_right_kasteleyn_submatrix, separator_matching_vertex_indices, vertex_index):

    column_vertex_index = _get_column_vertex_index(vertex_index)

    separator_matching_swapped_vertex_indices = np.zeros_like(separator_matching_vertex_indices)
    separator_matching_swapped_vertex_indices[::2] = separator_matching_vertex_indices[1::2]
    separator_matching_swapped_vertex_indices[1::2] = separator_matching_vertex_indices[::2]

    separator_matching_column_vertex_indices = \
            _get_column_vertex_indices(separator_matching_vertex_indices)

    next_l_matrix_row = _solve_lower_triangular(current_u_matrix.transpose(),
            inverse_lower_right_kasteleyn_submatrix[column_vertex_index]\
            [separator_matching_swapped_vertex_indices])

    next_u_matrix_column = _solve_lower_triangular(current_l_matrix,
            inverse_lower_right_kasteleyn_submatrix.transpose()[vertex_index]\
            [separator_matching_column_vertex_indices])

    return next_l_matrix_row, next_u_matrix_column

@jit(int32[:](planar_graph_nb_type, float64[:], float64[:, :], int32[:], boolean[:]), nopython=True)
def draw_separator_matching_edge_indices(expanded_dual_subgraph, weights,
        inverse_lower_right_kasteleyn_submatrix, submatrix_vertices,
        separator_vertices_in_submatrix_mask):

    separator_vertices_count = separator_vertices_in_submatrix_mask.shape[0]

    l_matrix = np.identity(2*separator_vertices_count).astype(np.float64)
    u_matrix = np.zeros_like(l_matrix)

    unsaturated_separator_vertices_mask = separator_vertices_in_submatrix_mask.copy()

    separator_matching_edge_indices_buffer = np.zeros(2*separator_vertices_count, dtype=np.int32)
    separator_matching_vertex_indices_buffer = np.zeros(separator_vertices_count, dtype=np.int32)
    separator_matching_edges_count = 0

    while np.any(unsaturated_separator_vertices_mask):

        separator_matching_vertices_count = 2*separator_matching_edges_count

        current_l_matrix = l_matrix[:separator_matching_vertices_count,
                :separator_matching_vertices_count]

        current_u_matrix = u_matrix[:separator_matching_vertices_count,
                :separator_matching_vertices_count]

        separator_matching_vertex_indices = \
                separator_matching_vertex_indices_buffer[:separator_matching_vertices_count]
 
        vertex_index = np.where(unsaturated_separator_vertices_mask)[0][0]
        column_vertex_index = _get_column_vertex_index(vertex_index)
        unsaturated_separator_vertices_mask[vertex_index] = False
        vertex = submatrix_vertices[vertex_index]

        next_l_matrix_row, next_next_u_matrix_column = \
                _get_next_l_matrix_row_and_u_matrix_column(current_l_matrix, current_u_matrix,
                inverse_lower_right_kasteleyn_submatrix, separator_matching_vertex_indices,
                vertex_index)

        random_number = np.random.uniform(0, 1)

        aggregated_probability_to_choose_adjacent_vertex = 0.0

        chosen_edge_index = -1
        chosen_adjacent_vertex_index = -1
        chosen_next_next_l_matrix_row = np.zeros(0, dtype=np.float64)
        chosen_next_u_matrix_column = np.zeros(0, dtype=np.float64)

        for edge_index in expanded_dual_subgraph.get_incident_edge_indices(vertex):

            adjacent_vertex = expanded_dual_subgraph.edges.get_opposite_vertex(edge_index, vertex)

            # adjacent_vertex must be in submatrix_vertices
            adjacent_vertex_index = np.where(submatrix_vertices == adjacent_vertex)[0][0]
 
            next_next_l_matrix_row, next_u_matrix_column = \
                    _get_next_l_matrix_row_and_u_matrix_column(current_l_matrix,
                    current_u_matrix, inverse_lower_right_kasteleyn_submatrix,
                    separator_matching_vertex_indices, adjacent_vertex_index)

            probability_to_choose_adjacent_vertex = np.absolute(weights[edge_index]*\
                    (inverse_lower_right_kasteleyn_submatrix[column_vertex_index,
                    adjacent_vertex_index] - (next_l_matrix_row*next_u_matrix_column).sum()))

            aggregated_probability_to_choose_adjacent_vertex += \
                    probability_to_choose_adjacent_vertex

            if aggregated_probability_to_choose_adjacent_vertex > random_number:

                chosen_edge_index = edge_index
                chosen_adjacent_vertex_index = adjacent_vertex_index
                chosen_next_next_l_matrix_row = next_next_l_matrix_row
                chosen_next_u_matrix_column = next_u_matrix_column

                break

        unsaturated_separator_vertices_mask[chosen_adjacent_vertex_index] = False

        l_matrix[separator_matching_vertices_count,
                :separator_matching_vertices_count] = next_l_matrix_row
        l_matrix[separator_matching_vertices_count + 1,
                :separator_matching_vertices_count] = chosen_next_next_l_matrix_row
        u_matrix[:separator_matching_vertices_count,
                separator_matching_vertices_count] = chosen_next_u_matrix_column
        u_matrix[:separator_matching_vertices_count,
                separator_matching_vertices_count + 1] = next_next_u_matrix_column

        chosen_column_adjacent_vertex_index = _get_column_vertex_index(chosen_adjacent_vertex_index)

        u_matrix[separator_matching_vertices_count, separator_matching_vertices_count] = \
                inverse_lower_right_kasteleyn_submatrix[column_vertex_index,
                chosen_adjacent_vertex_index] - \
                (next_l_matrix_row*chosen_next_u_matrix_column).sum()

        u_matrix[separator_matching_vertices_count + 1, separator_matching_vertices_count + 1] = \
                inverse_lower_right_kasteleyn_submatrix[chosen_column_adjacent_vertex_index,
                vertex_index] - (chosen_next_next_l_matrix_row*next_next_u_matrix_column).sum()

        separator_matching_edge_indices_buffer[separator_matching_edges_count] = chosen_edge_index

        separator_matching_vertex_indices_buffer[separator_matching_vertices_count] = vertex_index
        separator_matching_vertex_indices_buffer[separator_matching_vertices_count + 1] = \
                chosen_adjacent_vertex_index

        separator_matching_edges_count += 1

    return separator_matching_edge_indices_buffer[:separator_matching_edges_count]

import numpy as np
from . import utils
from ..sparse_lu import SparseLU


class KasteleynMatrixConstructor:

    @staticmethod
    def construct_symbolically(expanded_dual_graph, kasteleyn_orientation, vertices_permutation):

        vertex_positions = utils.get_inverse_sub_mapping(vertices_permutation,
                expanded_dual_graph.size)

        weight_indices = []
        weight_signs = []
        column_indices = []
        row_first_element_indices = np.zeros(expanded_dual_graph.size + 1, dtype=int)
        elements_count = 0

        for vertex_index in range(expanded_dual_graph.size):

            vertex = vertices_permutation[vertex_index]

            incident_edge_indices_list = []
            adjacent_vertex_indices_list = []

            for edge_index in expanded_dual_graph.get_incident_edge_indices(vertex):
                incident_edge_indices_list.append(edge_index)

            for adjacent_vertex in expanded_dual_graph.get_adjacent_vertices(vertex):
                adjacent_vertex_indices_list.append(vertex_positions[adjacent_vertex])

            incident_edge_indices = np.array(incident_edge_indices_list)

            adjacent_vertex_indices = np.array(adjacent_vertex_indices_list)
            row_column_indices = KasteleynMatrixConstructor._get_column_vertex_indices(
                    adjacent_vertex_indices)

            # Since each vertex of the expanded dual graph is of degree <= 3,
            # sorting takes O(1) time

            row_column_indices_permutation = row_column_indices.argsort()
            row_column_indices = row_column_indices[row_column_indices_permutation]
            incident_edge_indices = incident_edge_indices[row_column_indices_permutation]

            row_weight_indices = incident_edge_indices
            row_weight_signs = np.ones_like(incident_edge_indices)
            row_weight_signs[kasteleyn_orientation[incident_edge_indices] == vertex] = -1

            weight_indices.append(row_weight_indices)
            weight_signs.append(row_weight_signs)
            elements_count += row_weight_indices.shape[0]
            column_indices.append(row_column_indices)
            row_first_element_indices[vertex_index + 1] = elements_count

        if len(weight_indices) == 0:
            return np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=int), \
                    row_first_element_indices

        weight_indices = np.concatenate(weight_indices)
        weight_signs = np.concatenate(weight_signs)
        column_indices = np.concatenate(column_indices)

        return weight_indices, weight_signs, column_indices, row_first_element_indices

    @staticmethod
    def _get_column_vertex_indices(vertex_indices):

        return vertex_indices + 1 - 2*(vertex_indices%2)

    @staticmethod
    def get_inverse_lower_right_kasteleyn_submatrix(kasteleyn_matrix, submatrix_size):

        l_matrix, u_matrix = SparseLU.factorize(kasteleyn_matrix)

        l_submatrix = SparseLU.get_lower_right_submatrix(l_matrix, submatrix_size)
        u_submatrix = SparseLU.get_lower_right_submatrix(u_matrix, submatrix_size)

        inverse_kasteleyn_submatrix = np.zeros((submatrix_size, submatrix_size))

        unit_vector = np.zeros(submatrix_size)

        for index in range(submatrix_size):

            unit_vector[index] = 1

            inverse_kasteleyn_submatrix[:, index] = SparseLU.solve(l_submatrix, u_submatrix,
                    unit_vector)

            unit_vector[index] = 0

        return inverse_kasteleyn_submatrix

    @staticmethod
    def draw_separator_matching_edge_indices(expanded_dual_subgraph, weights,
            inverse_lower_right_kasteleyn_submatrix, submatrix_vertices,
            separator_vertices_in_submatrix_mask):

        separator_vertices_count = separator_vertices_in_submatrix_mask.shape[0]

        l_matrix = np.identity(2*separator_vertices_count)
        u_matrix = np.zeros_like(l_matrix)

        unsaturated_separator_vertices_mask = separator_vertices_in_submatrix_mask.copy()

        separator_matching_edge_indices_buffer = np.zeros(2*separator_vertices_count, dtype=int)
        separator_matching_vertex_indices_buffer = np.zeros(separator_vertices_count, dtype=int)
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
            column_vertex_index = KasteleynMatrixConstructor._get_column_vertex_index(vertex_index)
            unsaturated_separator_vertices_mask[vertex_index] = False
            vertex = submatrix_vertices[vertex_index]

            next_l_matrix_row, next_next_u_matrix_column = \
                    KasteleynMatrixConstructor._get_next_l_matrix_row_and_u_matrix_column(
                    current_l_matrix, current_u_matrix, inverse_lower_right_kasteleyn_submatrix,
                    separator_matching_vertex_indices, vertex_index)

            random_number = np.random.uniform(0, 1)

            aggregated_probability_to_choose_adjacent_vertex = 0.0

            chosen_edge_index = -1
            chosen_adjacent_vertex_index = -1
            chosen_next_next_l_matrix_row = np.zeros(0)
            chosen_next_u_matrix_column = np.zeros(0)

            for edge_index in expanded_dual_subgraph.get_incident_edge_indices(vertex):

                adjacent_vertex = expanded_dual_subgraph.edges.get_opposite_vertex(edge_index,
                        vertex)

                # adjacent_vertex must be in submatrix_vertices
                adjacent_vertex_index = np.where(submatrix_vertices == adjacent_vertex)[0][0]
     
                next_next_l_matrix_row, next_u_matrix_column = \
                        KasteleynMatrixConstructor._get_next_l_matrix_row_and_u_matrix_column(
                        current_l_matrix, current_u_matrix, inverse_lower_right_kasteleyn_submatrix,
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

            chosen_column_adjacent_vertex_index = \
                    KasteleynMatrixConstructor._get_column_vertex_index(
                    chosen_adjacent_vertex_index)

            u_matrix[separator_matching_vertices_count, separator_matching_vertices_count] = \
                    inverse_lower_right_kasteleyn_submatrix[column_vertex_index,
                    chosen_adjacent_vertex_index] - \
                    (next_l_matrix_row*chosen_next_u_matrix_column).sum()

            u_matrix[separator_matching_vertices_count + 1,
                    separator_matching_vertices_count + 1] = \
                    inverse_lower_right_kasteleyn_submatrix[chosen_column_adjacent_vertex_index,
                    vertex_index] - (chosen_next_next_l_matrix_row*next_next_u_matrix_column).sum()

            separator_matching_edge_indices_buffer[separator_matching_edges_count] = \
                    chosen_edge_index

            separator_matching_vertex_indices_buffer[separator_matching_vertices_count] = \
                    vertex_index
            separator_matching_vertex_indices_buffer[separator_matching_vertices_count + 1] = \
                    chosen_adjacent_vertex_index

            separator_matching_edges_count += 1

        return separator_matching_edge_indices_buffer[:separator_matching_edges_count]

    @staticmethod
    def _get_next_l_matrix_row_and_u_matrix_column(current_l_matrix, current_u_matrix,
            inverse_lower_right_kasteleyn_submatrix, separator_matching_vertex_indices,
            vertex_index):

        column_vertex_index = KasteleynMatrixConstructor._get_column_vertex_index(vertex_index)

        separator_matching_swapped_vertex_indices = np.zeros_like(separator_matching_vertex_indices)
        separator_matching_swapped_vertex_indices[::2] = separator_matching_vertex_indices[1::2]
        separator_matching_swapped_vertex_indices[1::2] = separator_matching_vertex_indices[::2]

        separator_matching_column_vertex_indices = \
                KasteleynMatrixConstructor._get_column_vertex_indices(
                separator_matching_vertex_indices)

        next_l_matrix_row = KasteleynMatrixConstructor._solve_lower_triangular(current_u_matrix.T,
                inverse_lower_right_kasteleyn_submatrix[column_vertex_index]\
                [separator_matching_swapped_vertex_indices])

        next_u_matrix_column = KasteleynMatrixConstructor._solve_lower_triangular(current_l_matrix,
                inverse_lower_right_kasteleyn_submatrix.transpose()[vertex_index]\
                [separator_matching_column_vertex_indices])

        return next_l_matrix_row, next_u_matrix_column
 
    @staticmethod
    def _solve_lower_triangular(matrix, right_hand_side):

        solution = np.zeros_like(right_hand_side)

        for row_index in range(matrix.shape[0]):

            solution[row_index] = (right_hand_side[row_index] - \
                    (solution[:row_index]*matrix[row_index, :row_index]).sum())\
                    /matrix[row_index, row_index]

        return solution

    @staticmethod
    def _get_column_vertex_index(vertex_index):

        return vertex_index + 1 - 2*(vertex_index%2)

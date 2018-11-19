import numpy as np
from numba import jit
from numba.types import Tuple, int32, float64, boolean
from .csr_matrix import CSRMatrix, csr_matrix_nb_type


class SparseLU:

    @staticmethod
    def factorize(matrix):

        return factorize(matrix)

    @staticmethod
    def get_lower_right_submatrix(matrix, lower_right_submatrix_size):

        return get_lower_right_submatrix(matrix, lower_right_submatrix_size)

    @staticmethod
    def solve(l_matrix, u_matrix, right_hand_side):

        return solve(l_matrix, u_matrix, right_hand_side)


@jit(Tuple((csr_matrix_nb_type, csr_matrix_nb_type))(csr_matrix_nb_type), nopython=True)
def factorize(matrix):

    # reference to an algorithm?

    l_element_values = list(np.zeros(0, dtype=np.float64))
    l_element_column_indices = list(np.zeros(0, dtype=np.int32))
    l_row_first_element_indices = np.zeros(matrix.size + 1, dtype=np.int32)

    u_element_values = list(np.zeros(0, dtype=np.float64))
    u_element_column_indices = list(np.zeros(0, dtype=np.int32))
    u_row_first_element_indices = np.zeros(matrix.size + 1, dtype=np.int32)

    new_fill_in_values = np.zeros(matrix.size, dtype=np.float64)
    new_fill_in_values_mask = np.array([False for i in range(matrix.size)])

    l_new_fill_in_column_indices = np.zeros(matrix.size, dtype=np.int32)
    u_new_fill_in_column_indices = np.zeros(matrix.size, dtype=np.int32)

    for row_index in range(matrix.size):

        l_new_fill_in_column_indices_count = 0
        u_new_fill_in_column_indices_count = 0

        for element_index in range(matrix.row_first_element_indices[row_index],
                matrix.row_first_element_indices[row_index + 1]):

            element_value = matrix.element_values[element_index]
            element_column_index = matrix.element_column_indices[element_index]

            # matrix should have non-zero diagonal

            new_fill_in_values[element_column_index] = element_value
            new_fill_in_values_mask[element_column_index] = True

            if element_column_index < row_index:

                l_new_fill_in_column_indices[l_new_fill_in_column_indices_count] = \
                        element_column_index
                l_new_fill_in_column_indices_count += 1

            else:

                u_new_fill_in_column_indices[u_new_fill_in_column_indices_count] = \
                        element_column_index
                u_new_fill_in_column_indices_count += 1

        while l_new_fill_in_column_indices_count > 0:

            l_element_column_index_position = \
                    l_new_fill_in_column_indices[:l_new_fill_in_column_indices_count].argmin()

            l_element_column_index = l_new_fill_in_column_indices[l_element_column_index_position]

            l_new_fill_in_column_indices[l_element_column_index_position:\
                    l_new_fill_in_column_indices_count - 1] = \
                    l_new_fill_in_column_indices[l_element_column_index_position + 1:\
                    l_new_fill_in_column_indices_count]

            l_new_fill_in_column_indices_count -= 1

            l_element_column_indices.append(l_element_column_index)

            u_element_value = \
                    u_element_values[u_row_first_element_indices[l_element_column_index]]
            l_element_value = \
                    new_fill_in_values[l_element_column_index]/u_element_value

            new_fill_in_values_mask[l_element_column_index] = False

            l_element_values.append(l_element_value)

            for u_element_index in range(u_row_first_element_indices[l_element_column_index] + \
                    1, u_row_first_element_indices[l_element_column_index + 1]):

                u_element_value = u_element_values[u_element_index]
                u_element_column_index = u_element_column_indices[u_element_index]

                value_to_substract = l_element_value*u_element_value

                if not new_fill_in_values_mask[u_element_column_index]:

                    new_fill_in_values[u_element_column_index] = -value_to_substract
                    new_fill_in_values_mask[u_element_column_index] = True

                    if u_element_column_index < row_index:

                        l_new_fill_in_column_indices[l_new_fill_in_column_indices_count] = \
                                u_element_column_index
                        l_new_fill_in_column_indices_count += 1

                    else:

                        u_new_fill_in_column_indices[u_new_fill_in_column_indices_count] = \
                                u_element_column_index
                        u_new_fill_in_column_indices_count += 1

                else:
                    new_fill_in_values[u_element_column_index] = \
                            new_fill_in_values[u_element_column_index] - value_to_substract

        l_element_values.append(1)
        l_element_column_indices.append(np.int32(row_index))
        l_row_first_element_indices[row_index + 1] = len(l_element_values)

        u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count] = \
                np.sort(u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count])

        u_element_column_indices += [column_index for column_index in \
                u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count]]
        u_element_values += [new_fill_in_values[column_index] for column_index in \
                u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count]]
        u_row_first_element_indices[row_index + 1] = len(u_element_values)

        for column_index in u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count]:
            new_fill_in_values_mask[column_index] = False

    l_matrix = CSRMatrix(np.array(l_element_values), np.array(l_element_column_indices),
            l_row_first_element_indices)
    u_matrix = CSRMatrix(np.array(u_element_values), np.array(u_element_column_indices),
            u_row_first_element_indices)

    return l_matrix, u_matrix

@jit(csr_matrix_nb_type(csr_matrix_nb_type, int32), nopython=True)
def get_lower_right_submatrix(matrix, lower_right_submatrix_size):

    element_values = list(np.zeros(0, dtype=np.float64))
    element_column_indices = list(np.zeros(0, dtype=np.int32))
    row_first_element_indices = np.zeros(lower_right_submatrix_size + 1, dtype=np.int32)

    for row_index in range(lower_right_submatrix_size):

        matrix_row_index = matrix.size - lower_right_submatrix_size + row_index

        matrix_row_first_element_index = matrix.row_first_element_indices[matrix_row_index]
        matrix_next_row_first_element_index = matrix.row_first_element_indices[matrix_row_index + 1]

        matrix_row_element_values = matrix.element_values[matrix_row_first_element_index:\
                matrix_next_row_first_element_index]
        matrix_row_element_column_indices = \
                matrix.element_column_indices[matrix_row_first_element_index:\
                matrix_next_row_first_element_index]

        relevant_column_indices_mask = matrix_row_element_column_indices >= matrix.size - \
                lower_right_submatrix_size

        if not np.any(relevant_column_indices_mask):
            continue

        relevant_element_values = matrix_row_element_values[relevant_column_indices_mask]
        relevant_element_column_indices = \
                matrix_row_element_column_indices[relevant_column_indices_mask]

        element_values += [element_value for element_value in relevant_element_values]
        element_column_indices += [element_column_index - matrix.size + lower_right_submatrix_size \
                for element_column_index in relevant_element_column_indices]

        row_first_element_indices[row_index + 1] = len(element_values)

    return CSRMatrix(np.array(element_values), np.array(element_column_indices, dtype=np.int32),
            row_first_element_indices)

@jit(float64[:](csr_matrix_nb_type, float64[:], boolean), nopython=True)
def _solve_triangular(matrix, right_hand_side, is_upper_triangular):

    solution = np.zeros_like(right_hand_side)

    problem_size = matrix.size

    if is_upper_triangular:
        row_indices_range = range(problem_size - 1, -1, -1)
    else:
        row_indices_range = range(problem_size)

    for row_index in row_indices_range:

        row_first_element_index = matrix.row_first_element_indices[row_index]
        next_row_first_element_index = matrix.row_first_element_indices[row_index + 1]

        row_element_values = matrix.element_values[row_first_element_index:\
                next_row_first_element_index]
        row_element_column_indices = \
                matrix.element_column_indices[row_first_element_index:\
                next_row_first_element_index]

        solution_elements = solution[row_element_column_indices]

        if is_upper_triangular:
            known_solution_elements = solution_elements[1:]
            known_solution_element_coefficients = row_element_values[1:]
            unknown_solution_element_coefficient = row_element_values[0]
        else:
            known_solution_elements = solution_elements[:-1]
            known_solution_element_coefficients = row_element_values[:-1]
            unknown_solution_element_coefficient = row_element_values[-1]

        solution[row_index] = (right_hand_side[row_index] - \
                (known_solution_elements*known_solution_element_coefficients).sum())\
                /unknown_solution_element_coefficient

    return solution

@jit(float64[:](csr_matrix_nb_type, csr_matrix_nb_type, float64[:]), nopython=True)
def solve(l_matrix, u_matrix, right_hand_side):

    return _solve_triangular(u_matrix, _solve_triangular(l_matrix, right_hand_side, False), True)

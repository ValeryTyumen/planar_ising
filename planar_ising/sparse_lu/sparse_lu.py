import numpy as np
from .csr_matrix import CSRMatrix


class SparseLU:

    @staticmethod
    def factorize(matrix):

        l_column_indices, l_row_first_element_indices, u_column_indices, \
                u_row_first_element_indices = SparseLU._factorize_symbolically(
                matrix.column_indices, matrix.row_first_element_indices)

        l_signs, l_logs, u_signs, u_logs = SparseLU._factorize(matrix, l_column_indices,
                l_row_first_element_indices, u_column_indices, u_row_first_element_indices)

        l_matrix = CSRMatrix(l_signs, l_logs, l_column_indices, l_row_first_element_indices)
        u_matrix = CSRMatrix(u_signs, u_logs, u_column_indices, u_row_first_element_indices)

        return l_matrix, u_matrix

    @staticmethod
    def _factorize_symbolically(column_indices, row_first_element_indices):

        size = row_first_element_indices.shape[0] - 1

        l_column_indices = []
        l_row_first_element_indices = np.zeros(size + 1, dtype=int)

        u_column_indices = []
        u_row_first_element_indices = np.zeros(size + 1, dtype=int)

        new_fill_in_values_mask = np.zeros(size, dtype=bool)

        l_new_fill_in_column_indices = np.zeros(size, dtype=np.int64)
        u_new_fill_in_column_indices = np.zeros(size, dtype=np.int64)

        for row_index in range(size):

            l_new_fill_in_column_indices_count = 0
            u_new_fill_in_column_indices_count = 0

            for element_index in range(row_first_element_indices[row_index],
                    row_first_element_indices[row_index + 1]):

                column_index = column_indices[element_index]
                new_fill_in_values_mask[column_index] = True

                if column_index < row_index:
                    l_new_fill_in_column_indices[l_new_fill_in_column_indices_count] = column_index
                    l_new_fill_in_column_indices_count += 1
                else:
                    u_new_fill_in_column_indices[u_new_fill_in_column_indices_count] = column_index
                    u_new_fill_in_column_indices_count += 1

            while l_new_fill_in_column_indices_count > 0:

                l_column_index_position = \
                        l_new_fill_in_column_indices[:l_new_fill_in_column_indices_count].argmin()

                l_column_index = l_new_fill_in_column_indices[l_column_index_position]

                l_new_fill_in_column_indices[l_column_index_position:\
                        l_new_fill_in_column_indices_count - 1] = \
                        l_new_fill_in_column_indices[l_column_index_position + 1:\
                        l_new_fill_in_column_indices_count]

                l_new_fill_in_column_indices_count -= 1

                l_column_indices.append(l_column_index)

                new_fill_in_values_mask[l_column_index] = False

                for u_element_index in range(u_row_first_element_indices[l_column_index] + \
                        1, u_row_first_element_indices[l_column_index + 1]):

                    u_column_index = u_column_indices[u_element_index]

                    if not new_fill_in_values_mask[u_column_index]:

                        new_fill_in_values_mask[u_column_index] = True

                        if u_column_index < row_index:

                            l_new_fill_in_column_indices[l_new_fill_in_column_indices_count] = \
                                    u_column_index
                            l_new_fill_in_column_indices_count += 1

                        else:

                            u_new_fill_in_column_indices[u_new_fill_in_column_indices_count] = \
                                    u_column_index
                            u_new_fill_in_column_indices_count += 1

            l_column_indices.append(row_index)
            l_row_first_element_indices[row_index + 1] = len(l_column_indices)

            u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count] = \
                    np.sort(u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count])

            u_column_indices += [column_index for column_index in \
                    u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count]]
            u_row_first_element_indices[row_index + 1] = len(u_column_indices)

            for column_index in u_new_fill_in_column_indices[:u_new_fill_in_column_indices_count]:
                new_fill_in_values_mask[column_index] = False

        return np.array(l_column_indices), l_row_first_element_indices, \
                np.array(u_column_indices), u_row_first_element_indices

    @staticmethod
    def _factorize(matrix, l_column_indices, l_row_first_element_indices, u_column_indices,
            u_row_first_element_indices):

        signs = matrix.signs
        logs = matrix.logs
        column_indices = matrix.column_indices
        row_first_element_indices = matrix.row_first_element_indices

        size = row_first_element_indices.shape[0] - 1

        l_signs = np.zeros(l_column_indices.shape[0])
        l_logs = np.zeros_like(l_signs)
        u_signs = np.zeros(u_column_indices.shape[0])
        u_logs = np.zeros_like(u_signs)

        buffer_signs = np.zeros(size)
        buffer_logs = np.zeros_like(buffer_signs)

        for row_index in range(size):

            row_column_indices = column_indices[row_first_element_indices[row_index]:\
                    row_first_element_indices[row_index + 1]]
            row_signs = signs[row_first_element_indices[row_index]:\
                    row_first_element_indices[row_index + 1]]
            row_logs = logs[row_first_element_indices[row_index]:\
                    row_first_element_indices[row_index + 1]]

            buffer_signs[row_column_indices] = row_signs
            buffer_logs[row_column_indices] = row_logs

            l_row_first_element_index = l_row_first_element_indices[row_index]
            l_next_row_first_element_index = l_row_first_element_indices[row_index + 1]
            l_row_column_indices = l_column_indices[\
                    l_row_first_element_index:l_next_row_first_element_index]

            u_row_first_element_index = u_row_first_element_indices[row_index]
            u_next_row_first_element_index = u_row_first_element_indices[row_index + 1]
            u_row_column_indices = u_column_indices[\
                    u_row_first_element_index:u_next_row_first_element_index]

            for eliminate_index in range(l_next_row_first_element_index - \
                    l_row_first_element_index - 1):

                eliminating_row_index = l_row_column_indices[eliminate_index]

                eliminating_row_column_indices = u_column_indices[\
                        u_row_first_element_indices[eliminating_row_index]:\
                        u_row_first_element_indices[eliminating_row_index + 1]]

                eliminating_row_signs = u_signs[u_row_first_element_indices[eliminating_row_index]:\
                        u_row_first_element_indices[eliminating_row_index + 1]]
                eliminating_row_logs = u_logs[u_row_first_element_indices[eliminating_row_index]:\
                        u_row_first_element_indices[eliminating_row_index + 1]]

                l_sign = buffer_signs[eliminating_row_index]*eliminating_row_signs[0]
                l_log = buffer_logs[eliminating_row_index] - eliminating_row_logs[0]

                l_signs[l_row_first_element_index + eliminate_index] = l_sign
                l_logs[l_row_first_element_index + eliminate_index] = l_log

                result_signs, result_logs = SparseLU._logaddexp(
                        buffer_signs[eliminating_row_column_indices],
                        buffer_logs[eliminating_row_column_indices],
                        -l_sign*eliminating_row_signs,
                        l_log + eliminating_row_logs)

                buffer_signs[eliminating_row_column_indices] = result_signs
                buffer_logs[eliminating_row_column_indices] = result_logs

            l_signs[l_next_row_first_element_index - 1] = 1
            l_logs[l_next_row_first_element_index - 1] = 0

            u_signs[u_row_first_element_index:u_next_row_first_element_index] = \
                    buffer_signs[u_row_column_indices]
            u_logs[u_row_first_element_index:u_next_row_first_element_index] = \
                    buffer_logs[u_row_column_indices]

            buffer_signs[u_row_column_indices] = 0

        return l_signs, l_logs, u_signs, u_logs

    @staticmethod
    def get_logdet_grad(matrix, l_matrix, u_matrix):

        signs = matrix.signs
        logs = matrix.logs
        column_indices = matrix.column_indices
        row_first_element_indices = matrix.row_first_element_indices
        l_signs = l_matrix.signs
        l_logs = l_matrix.logs
        l_column_indices = l_matrix.column_indices
        l_row_first_element_indices = l_matrix.row_first_element_indices
        u_signs = u_matrix.signs
        u_logs = u_matrix.logs
        u_column_indices = u_matrix.column_indices
        u_row_first_element_indices = u_matrix.row_first_element_indices

        size = l_row_first_element_indices.shape[0] - 1
 
        grad_signs = np.zeros_like(signs)
        grad_logs = np.zeros_like(logs)

        u_grad_signs = np.zeros_like(u_signs)
        u_grad_signs[u_row_first_element_indices[:-1]] = u_signs[u_row_first_element_indices[:-1]]
        u_grad_logs = np.zeros_like(u_logs)
        u_grad_logs[u_row_first_element_indices[:-1]] = -u_logs[u_row_first_element_indices[:-1]]

        buffer_grad_signs = np.zeros(size)
        buffer_grad_logs = np.zeros_like(buffer_grad_signs)

        for row_index in range(size - 1, -1, -1):

            l_row_first_element_index = l_row_first_element_indices[row_index]
            l_next_row_first_element_index = l_row_first_element_indices[row_index + 1]
            l_row_column_indices = l_column_indices[\
                    l_row_first_element_index:l_next_row_first_element_index - 1]
 
            u_row_first_element_index = u_row_first_element_indices[row_index]
            u_next_row_first_element_index = u_row_first_element_indices[row_index + 1]
            u_row_column_indices = u_column_indices[\
                    u_row_first_element_index:u_next_row_first_element_index]

            buffer_grad_signs[u_row_column_indices] = u_grad_signs[\
                    u_row_first_element_index:u_next_row_first_element_index]
            buffer_grad_logs[u_row_column_indices] = u_grad_logs[\
                    u_row_first_element_index:u_next_row_first_element_index]

            for eliminate_index in range(l_next_row_first_element_index - \
                    l_row_first_element_index - 2, -1, -1):

                eliminating_row_index = l_row_column_indices[eliminate_index]

                eliminating_row_column_indices = u_column_indices[\
                        u_row_first_element_indices[eliminating_row_index] + 1:\
                        u_row_first_element_indices[eliminating_row_index + 1]]

                eliminating_row_signs = u_signs[\
                        u_row_first_element_indices[eliminating_row_index] + 1:\
                        u_row_first_element_indices[eliminating_row_index + 1]]
                eliminating_row_logs = u_logs[\
                        u_row_first_element_indices[eliminating_row_index] + 1:\
                        u_row_first_element_indices[eliminating_row_index + 1]]

                eliminating_row_grad_signs = u_grad_signs[\
                        u_row_first_element_indices[eliminating_row_index] + 1:\
                        u_row_first_element_indices[eliminating_row_index + 1]]
                eliminating_row_grad_logs = u_grad_logs[\
                        u_row_first_element_indices[eliminating_row_index] + 1:\
                        u_row_first_element_indices[eliminating_row_index + 1]]

                grad_sum_sign, grad_sum_log = SparseLU._logsumexp(eliminating_row_signs*\
                        buffer_grad_signs[eliminating_row_column_indices],
                        eliminating_row_logs + buffer_grad_logs[eliminating_row_column_indices])

                grad_sum_sign = grad_sum_sign*u_signs[\
                        u_row_first_element_indices[eliminating_row_index]]
                grad_sum_log = grad_sum_log - u_logs[\
                        u_row_first_element_indices[eliminating_row_index]]

                buffer_grad_signs[eliminating_row_index] = -grad_sum_sign
                buffer_grad_logs[eliminating_row_index] = grad_sum_log

                l_sign = l_signs[l_row_first_element_index + eliminate_index]
                l_log = l_logs[l_row_first_element_index + eliminate_index]

                result_signs, result_logs = SparseLU._logaddexp(
                        u_grad_signs[u_row_first_element_indices[eliminating_row_index]],
                        u_grad_logs[u_row_first_element_indices[eliminating_row_index]],
                        l_sign*grad_sum_sign, l_log + grad_sum_log)

                u_grad_signs[u_row_first_element_indices[eliminating_row_index]] = result_signs
                u_grad_logs[u_row_first_element_indices[eliminating_row_index]] = result_logs

                result_signs, result_logs = SparseLU._logaddexp(eliminating_row_grad_signs,
                        eliminating_row_grad_logs,
                        -l_sign*buffer_grad_signs[eliminating_row_column_indices],
                        l_log + buffer_grad_logs[eliminating_row_column_indices])
 
                eliminating_row_grad_signs[:] = result_signs
                eliminating_row_grad_logs[:] = result_logs
 
            row_column_indices = column_indices[row_first_element_indices[row_index]:\
                    row_first_element_indices[row_index + 1]]

            grad_signs[row_first_element_indices[row_index]:\
                    row_first_element_indices[row_index + 1]] = \
                    buffer_grad_signs[row_column_indices]
            grad_logs[row_first_element_indices[row_index]:\
                    row_first_element_indices[row_index + 1]] = buffer_grad_logs[row_column_indices]

            buffer_grad_signs[l_row_column_indices] = 0
            buffer_grad_signs[u_row_column_indices] = 0
            buffer_grad_logs[l_row_column_indices] = 0
            buffer_grad_logs[u_row_column_indices] = 0

        #grad = signs*grad_signs*np.exp(logs + grad_logs)
 
        return signs*grad_signs, logs + grad_logs
 
    @staticmethod
    def _logaddexp(signs1, logs1, signs2, logs2):

        max_logs = np.maximum(logs1, logs2)
        #max_logs[np.isinf(max_logs)] = 0

        values1 = signs1*np.exp(logs1 - max_logs)
        values2 = signs2*np.exp(logs2 - max_logs)

        result = values1 + values2

        signs = np.sign(result)
        logs = np.zeros_like(result)
        logs[result != 0] = np.log(np.absolute(result[result != 0]))
        logs += max_logs

        return signs, logs

    @staticmethod
    def _logsumexp(signs, logs):

        max_log = logs.max()

        #if np.isinf(max_log):
        #    max_log = 0

        result = (signs*np.exp(logs - max_log)).sum()
 
        if result == 0:
            sign = 0
            log = 0
        else:
            sign = np.sign(result)
            log = np.log(np.absolute(result))

        log += max_log

        return sign, log

    @staticmethod
    def get_lower_right_submatrix(matrix, lower_right_submatrix_size):

        signs = []
        logs = []
        column_indices = []
        row_first_element_indices = np.zeros(lower_right_submatrix_size + 1, dtype=int)

        for row_index in range(lower_right_submatrix_size):

            matrix_row_index = matrix.size - lower_right_submatrix_size + row_index

            matrix_row_first_element_index = matrix.row_first_element_indices[matrix_row_index]
            matrix_next_row_first_element_index = \
                    matrix.row_first_element_indices[matrix_row_index + 1]

            matrix_row_signs = matrix.signs[matrix_row_first_element_index:\
                    matrix_next_row_first_element_index]
            matrix_row_logs = matrix.logs[matrix_row_first_element_index:\
                    matrix_next_row_first_element_index]
            matrix_row_column_indices = matrix.column_indices[matrix_row_first_element_index:\
                    matrix_next_row_first_element_index]

            relevant_column_indices_mask = (matrix_row_column_indices >= matrix.size - \
                    lower_right_submatrix_size)

            if not np.any(relevant_column_indices_mask):
                continue

            signs.append(matrix_row_signs[relevant_column_indices_mask])
            logs.append(matrix_row_logs[relevant_column_indices_mask])
            column_indices.append(matrix_row_column_indices[relevant_column_indices_mask] - \
                    matrix.size + lower_right_submatrix_size)

            row_first_element_indices[row_index + 1] = row_first_element_indices[row_index] + \
                    relevant_column_indices_mask.sum()

        if len(signs) == 0:
            return CSRMatrix(np.zeros(0), np.zeros(0), np.zeros(0, dtype=int),
                    row_first_element_indices)

        return CSRMatrix(np.concatenate(signs), np.concatenate(logs),
                np.concatenate(column_indices), row_first_element_indices)

    @staticmethod
    def solve(l_matrix, u_matrix, right_hand_side):

        return SparseLU._solve_triangular(u_matrix, SparseLU._solve_triangular(l_matrix,
                right_hand_side, False), True)

    @staticmethod
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

            row_elements = matrix.signs[row_first_element_index:next_row_first_element_index]*\
                    np.exp(matrix.logs[row_first_element_index:next_row_first_element_index])
            row_column_indices = matrix.column_indices[row_first_element_index:\
                    next_row_first_element_index]

            row_solution = solution[row_column_indices]
 
            if is_upper_triangular:

                known_solution = row_solution[1:]
                known_solution_coefs = row_elements[1:]
                unknown_solution_coef = row_elements[0]

            else:

                known_solution = row_solution[:-1]
                known_solution_coefs = row_elements[:-1]
                unknown_solution_coef = row_elements[-1]

            solution[row_index] = (right_hand_side[row_index] - (known_solution*\
                    known_solution_coefs).sum())/unknown_solution_coef

        return solution

import numpy as np
from .small_inference_and_sampling import SmallInferenceAndSampling
from .cond_planar_inference_and_sampling import CondPlanarInferenceAndSampling
from ..planar_ising import PlanarIsingModel, InferenceAndSampling


class DecompInferenceAndSampling:
    """
    A class performing inference/sampling in decomposable zero-field Ising models.
    """

    def __init__(self, graph):

        self._graph = graph

    def prepare(self, sampling=False):

        graph = self._graph

        self._cond_inference_and_sampling = {}

        for node_index, parent_index, adjacency_index_in_parent in graph.traverse():

            if parent_index == -1:
                constraint_vertices = np.array([], dtype=int)
            else:
                constraint_vertices = graph.adjacent_node_connection_vertices[parent_index]\
                        [adjacency_index_in_parent]

            if graph.is_small_node[node_index]:
                cond_inference_and_sampling = SmallInferenceAndSampling(graph.nodes[node_index],
                        constraint_vertices)
            else:

                cond_inference_and_sampling = CondPlanarInferenceAndSampling(
                        graph.nodes[node_index], constraint_vertices)
                cond_inference_and_sampling.prepare(sampling=sampling)

            self._cond_inference_and_sampling[node_index] = cond_inference_and_sampling

    def compute_logpf(self, interaction_values, with_marginals=False):

        result = self._construct_aggregated_interactions(interaction_values, with_marginals)

        if with_marginals:
            return result[1:]
        else:
            return result[1]

    def sample_spin_configurations(self, sample_size, interaction_values):
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

        aggregated_interaction_values, _ = \
                self._construct_aggregated_interactions(interaction_values, False)

        graph = self._graph

        spins = []

        for sample_index in range(sample_size):

            node_spins = [None]*graph.nodes_count

            for node_index, parent_index, adjacency_index_in_parent in graph.traverse():

                if parent_index == -1:
                    constraint_spins = np.array([], dtype=int)
                else:

                    connection_vertices_in_parent = \
                            graph.connection_vertices[parent_index][adjacency_index_in_parent]
                    constraint_spins = node_spins[parent_index][connection_vertices_in_parent]

                node = graph.nodes[node_index]

                aggr_interaction_values = aggregated_interaction_values[node_index]

                node_spins[node_index] = \
                        self._cond_inference_and_sampling[node_index].sample_spin_configurations(1,
                        aggr_interaction_values, constraint_spins)[0]

            spins.append(np.zeros(graph.size, dtype=int))

            for node_index, _, _ in graph.traverse():
                spins[-1][graph.graph_vertices[node_index]] = node_spins[node_index]

        return np.asarray(spins)
 
    def _construct_aggregated_interactions(self, interaction_values, with_marginals):

        graph = self._graph

        aggregated_interaction_values = [None]*graph.nodes_count
        cond_logpf = [None]*graph.nodes_count

        if with_marginals:
            node_marginals = [None]*graph.nodes_count

        logpf = 0.0
 
        matrix = np.linalg.inv(np.array([
                [+1, +1, +1, +1],
                [+1, +1, -1, -1],
                [+1, -1, +1, -1],
                [+1, -1, -1, +1]
        ]))

        for node_index, parent_index, adjacency_index_in_parent in list(graph.traverse())[::-1]:

            node = graph.nodes[node_index]

            graph_edge_indices = graph.graph_edge_indices[node_index]

            aggr_interaction_values = np.zeros(graph_edge_indices.shape)
            aggr_interaction_values[graph_edge_indices != -1] = \
                    interaction_values[graph_edge_indices[graph_edge_indices != -1]]

            for child_index, connection_vertices in zip(graph.adjacent_node_indices[node_index],
                    graph.connection_vertices[node_index]):

                if child_index == parent_index:
                    continue

                if connection_vertices.shape[0] < 2:
                    logpf += cond_logpf[child_index]
                elif connection_vertices.shape[0] == 2:

                    cond_logpf_plus, cond_logpf_minus = cond_logpf[child_index]

                    logpf += (cond_logpf_plus + cond_logpf_minus)/2

                    connection_edge_index = self._find_edge_index(node_index, *connection_vertices)

                    aggr_interaction_values[connection_edge_index] += (cond_logpf_plus - \
                            cond_logpf_minus)/2

                else:

                    solution = matrix.dot(cond_logpf[child_index])

                    logpf += solution[0]

                    connection_edge_indices = [self._find_edge_index(node_index,
                            connection_vertices[i], connection_vertices[j]) for i, j in [(0, 1),
                            (0, 2), (1, 2)]]

                    for index in range(3):
                        aggr_interaction_values[connection_edge_indices[index]] += \
                                solution[index + 1]

            aggregated_interaction_values[node_index] = aggr_interaction_values

            if parent_index == -1:
                connection_vertices = np.array([], dtype=int)
            else:
                connection_vertices = graph.adjacent_node_connection_vertices[parent_index]\
                        [adjacency_index_in_parent]

            if connection_vertices.shape[0] == 0:
                constraint_spins = np.array([[]])
            elif connection_vertices.shape[0] == 1:
                constraint_spins = np.array([[1]])
            elif connection_vertices.shape[0] == 2:
                constraint_spins = np.array([[1, 1], [1, -1]])
            else:
                constraint_spins = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])

            cond_logpf[node_index] = []

            if with_marginals:
                node_marginals[node_index] = []

            for constr_spins in constraint_spins:

                result = self._cond_inference_and_sampling[node_index].compute_logpf(
                        aggr_interaction_values, constr_spins, with_marginals=with_marginals)

                if with_marginals:
                    cond_logpf[node_index].append(result[0])
                    node_marginals[node_index].append(result[1])
                else:
                    cond_logpf[node_index].append(result)

            cond_logpf[node_index] = np.asarray(cond_logpf[node_index])

            if with_marginals:
                node_marginals[node_index] = np.asarray(node_marginals[node_index])

        logpf += cond_logpf[0][0]

        if not with_marginals:
            return aggregated_interaction_values, logpf

        for node_index, parent_index, adjacency_index_in_parent in graph.traverse():
 
            if parent_index == -1:
                connection_vertices_in_parent = np.array([], dtype=int)
            else:
                connection_vertices_in_parent = \
                        graph.connection_vertices[parent_index][adjacency_index_in_parent]

            if connection_vertices_in_parent.shape[0] < 2:
                node_marginals[node_index] = node_marginals[node_index][0]
            elif connection_vertices_in_parent.shape[0] == 2:

                connection_edge_index_in_parent = self._find_edge_index(parent_index,
                        *connection_vertices_in_parent)

                marginal_prob = \
                        (node_marginals[parent_index][connection_edge_index_in_parent] + 1)/2

                node_marginals[node_index] = node_marginals[node_index][0]*marginal_prob + \
                        node_marginals[node_index][1]*(1 - marginal_prob)

            else:

                connection_edge_indices_in_parent = [self._find_edge_index(parent_index,
                        connection_vertices_in_parent[i], connection_vertices_in_parent[j]) \
                        for i, j in [(1, 2), (0, 2), (0, 1)]]

                marginal_probs = [(node_marginals[parent_index][i] + 1)/2 for i in \
                        connection_edge_indices_in_parent]

                probs = np.array([
                        marginal_probs[0] + marginal_probs[1] + marginal_probs[2] - 1,
                        marginal_probs[2] - marginal_probs[0] - marginal_probs[1] + 1,
                        marginal_probs[1] - marginal_probs[0] - marginal_probs[2] + 1,
                        marginal_probs[0] - marginal_probs[1] - marginal_probs[2] + 1
                ])/2

                node_marginals[node_index] = (node_marginals[node_index]*probs[:, None]).sum(axis=0)

        marginals = np.zeros_like(interaction_values)

        for node_marginal_values, graph_edge_indices in zip(node_marginals,
                graph.graph_edge_indices):

            marginals[graph_edge_indices[graph_edge_indices != -1]] = \
                    node_marginal_values[graph_edge_indices != -1]

        return aggregated_interaction_values, logpf, marginals

    def _find_edge_index(self, node_index, vertex1, vertex2):

        node = self._graph.nodes[node_index]

        if self._graph.is_small_node[node_index]:

            return np.where(((node[:, 0] == vertex1) & (node[:, 1] == vertex2)) | \
                    ((node[:, 0] == vertex2) & (node[:, 1] == vertex1)))[0][0]

        return np.where(((node.edges.vertex1 == vertex1) & (node.edges.vertex2 == vertex2)) | \
                ((node.edges.vertex1 == vertex2) & (node.edges.vertex2 == vertex1)))[0][0]

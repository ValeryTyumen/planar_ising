import numpy as np
from ..planar_ising import PlanarIsingModel, InferenceAndSampling
from ..planar_graph import PlanarGraph, PlanarGraphEdges, PlanarGraphConstructor


class CondPlanarInferenceAndSampling:

    def __init__(self, graph, constraint_vertices):

        self._graph = graph
        self._constraint_vertices = constraint_vertices

    def prepare(self, sampling=False):

        if len(self._constraint_vertices) > 1:

            self._collapsed_vertex, self._new_vertices_mapping, self._new_edge_indices_mapping, \
                    self._cond_graph = CondPlanarInferenceAndSampling._collapse_constraint_vertices(
                    self._graph, self._constraint_vertices)

        else:
            self._cond_graph = self._graph

        self._cond_inference = InferenceAndSampling(PlanarIsingModel(self._cond_graph,
            np.zeros(self._cond_graph.edges_count)))
        self._cond_inference.prepare(sampling=sampling)

    def compute_logpf(self, interaction_values, constraint_spins, with_marginals=False):

        if self._constraint_vertices.shape[0] < 2:

            self._cond_inference.register_new_interactions(interaction_values)
            result = self._cond_inference.compute_logpf(with_marginals=with_marginals)

            addition = 0.0

            if self._constraint_vertices.shape[0] == 1:
                addition = -np.log(2)

            if with_marginals:
                return result[0] + addition, result[1]

            return result + addition

        cond_interaction_values, constraint_vertex_spins, constraint_edge_marginals = \
                self._get_cond_interaction_values(interaction_values, constraint_spins)

        self._cond_inference.register_new_interactions(cond_interaction_values)
        result = self._cond_inference.compute_logpf(with_marginals=with_marginals)

        if with_marginals:
            logpf = result[0]
        else:
            logpf = result

        logpf = logpf - np.log(2) + sum(interaction_values[e]*m for e, m in \
                constraint_edge_marginals.items())

        if with_marginals:

            marginals = np.zeros_like(interaction_values)
            marginals[self._new_edge_indices_mapping != -1] = \
                    result[1][self._new_edge_indices_mapping[self._new_edge_indices_mapping != -1]]

            for vertex in self._constraint_vertices:
                if constraint_vertex_spins[vertex] == -1:
                    for edge_index in self._graph.get_incident_edge_indices(vertex):

                        adjacent_vertex = self._graph.edges.get_opposite_vertex(edge_index, vertex)

                        if adjacent_vertex not in self._constraint_vertices:
                            marginals[edge_index] = -marginals[edge_index]

            for edge_index, marginal in constraint_edge_marginals.items():
                marginals[edge_index] = marginal

            return logpf, marginals

        return logpf

    def sample_spin_configurations(self, sample_size, interaction_values, constraint_spins):

        if self._constraint_vertices.shape[0] < 2:

            self._cond_inference.register_new_interactions(interaction_values)
            spins = self._cond_inference.sample_spin_configurations(sample_size)

            if self._constraint_vertices.shape[0] == 1:
                spins[spins[:, self._constraint_vertices[0]] != constraint_spins[0]] *= -1

            return spins

        cond_interaction_values, _, _ = self._get_cond_interaction_values(interaction_values,
                constraint_spins)

        self._cond_inference.register_new_interactions(cond_interaction_values)

        cond_spins = self._cond_inference.sample_spin_configurations(sample_size)
        cond_spins[cond_spins[:, self._collapsed_vertex] != 1] *= -1

        spins = np.zeros((sample_size, self._graph.size), dtype=int)

        spins[:, self._new_vertices_mapping != -1] = \
                cond_spins[:, self._new_vertices_mapping[self._new_vertices_mapping != -1]]
        spins[:, self._constraint_vertices] = constraint_spins[None, :]

        return spins

    def _get_cond_interaction_values(self, interaction_values, constraint_spins):

        cond_interaction_values = np.array(interaction_values)

        constraint_vertex_spins = {v:s for v, s in zip(self._constraint_vertices, constraint_spins)}

        constraint_edge_marginals = {}

        for vertex in self._constraint_vertices:
            for edge_index in self._graph.get_incident_edge_indices(vertex):

                adjacent_vertex = self._graph.edges.get_opposite_vertex(edge_index, vertex)

                if adjacent_vertex not in self._constraint_vertices:
                    if constraint_vertex_spins[vertex] == -1:
                        cond_interaction_values[edge_index] = -cond_interaction_values[edge_index]
                elif constraint_vertex_spins[vertex] == constraint_vertex_spins[adjacent_vertex]:
                    constraint_edge_marginals[edge_index] = 1
                else:
                    constraint_edge_marginals[edge_index] = -1

        cond_interaction_values = np.bincount(
                self._new_edge_indices_mapping[self._new_edge_indices_mapping != -1],
                weights=cond_interaction_values[self._new_edge_indices_mapping != -1])

        return cond_interaction_values, constraint_vertex_spins, constraint_edge_marginals

    @staticmethod
    def _collapse_constraint_vertices(graph, constraint_vertices):

        constraint_vertices_mask = np.zeros(graph.size, dtype=bool)
        constraint_vertices_mask[constraint_vertices] = True

        collapsed_vertex = (~constraint_vertices_mask).sum()

        new_vertices_mapping1 = -np.ones(graph.size, dtype=int)
        new_vertices_mapping1[~constraint_vertices_mask] = np.arange(collapsed_vertex)
 
        constraint_edges_mask = (constraint_vertices_mask[graph.edges.vertex1] & \
                constraint_vertices_mask[graph.edges.vertex2])

        remaining_edges_count = (~constraint_edges_mask).sum()

        new_edge_indices_mapping1 = -np.ones(graph.edges_count, dtype=int)
        new_edge_indices_mapping1[~constraint_edges_mask] = np.arange(remaining_edges_count)

        new_edges = PlanarGraphEdges(remaining_edges_count)
        new_incident_edge_example_indices = -np.ones(collapsed_vertex + 1, dtype=int)

        for edge_index, (vertex1, vertex2) in enumerate(zip(graph.edges.vertex1,
                graph.edges.vertex2)):

            if constraint_edges_mask[edge_index]:
                continue

            if constraint_vertices_mask[vertex1]:
                new_vertex1 = collapsed_vertex
            else:
                new_vertex1 = new_vertices_mapping1[vertex1]

            if constraint_vertices_mask[vertex2]:
                new_vertex2 = collapsed_vertex
            else:
                new_vertex2 = new_vertices_mapping1[vertex2]

            new_edges.append(new_vertex1, new_vertex2)

        for vertex in range(graph.size):

            if constraint_vertices_mask[vertex]:
                continue

            new_vertex = new_vertices_mapping1[vertex]

            for edge_index in graph.get_incident_edge_indices(vertex):

                new_edge_index = new_edge_indices_mapping1[edge_index]

                next_edge_index = graph.edges.get_next_edge_index(edge_index, vertex)
                new_next_edge_index = new_edge_indices_mapping1[next_edge_index]

                new_edges.set_next_edge(new_edge_index, new_vertex, new_next_edge_index)

                new_incident_edge_example_indices[new_vertex] = new_edge_index

        for vertex in constraint_vertices:
            for edge_index in graph.get_incident_edge_indices(vertex):
                if constraint_edges_mask[edge_index]:

                    main_vertex = vertex
                    main_edge_index = edge_index

                    break

        current_vertex = main_vertex
        current_edge_index = main_edge_index
        on_start = True

        new_previous_edge_index = -1
        new_first_edge_index = -1

        while on_start or current_vertex != main_vertex or current_edge_index != main_edge_index:

            on_start = False

            current_edge_index = graph.edges.get_next_edge_index(current_edge_index, current_vertex)

            if current_edge_index == main_edge_index and constraint_vertices.shape[0] == 3:
                continue

            if constraint_edges_mask[current_edge_index]:
                current_vertex = graph.edges.get_opposite_vertex(current_edge_index, current_vertex)
                continue

            new_current_edge_index = new_edge_indices_mapping1[current_edge_index]

            if new_previous_edge_index == -1:
                new_first_edge_index = new_current_edge_index
            else:
                new_edges.set_next_edge(new_previous_edge_index, collapsed_vertex,
                        new_current_edge_index)

            new_incident_edge_example_indices[collapsed_vertex] = new_current_edge_index
            new_previous_edge_index = new_current_edge_index

        new_edges.set_next_edge(new_current_edge_index, collapsed_vertex, new_first_edge_index)

        new_vertex_costs = np.ones(collapsed_vertex + 1)/(collapsed_vertex + 1)
        new_graph1 = PlanarGraph(new_vertex_costs, new_incident_edge_example_indices, new_edges)

        new_vertices_mapping2, new_edge_indices_mapping2, new_graph2 = \
                PlanarGraphConstructor.remove_double_edges(new_graph1)

        new_edge_exists_mask = (new_edge_indices_mapping1 != -1)
        new_edge_indices_mapping = -np.ones_like(new_edge_indices_mapping1)
        new_edge_indices_mapping[new_edge_exists_mask] = new_edge_indices_mapping2[\
                new_edge_indices_mapping1[new_edge_exists_mask]]

        new_vertex_exists_mask = (new_vertices_mapping1 != -1)
        new_vertices_mapping = -np.ones_like(new_vertices_mapping1)
        new_vertices_mapping[new_vertex_exists_mask] = new_vertices_mapping2[\
                new_vertices_mapping1[new_vertex_exists_mask]]

        return new_vertices_mapping2[collapsed_vertex], new_vertices_mapping, \
                new_edge_indices_mapping, new_graph2

import numpy as np
from .. import common_utils
from ..planar_graph import PlanarGraph, search_utils


class TreeCycles:

    @staticmethod
    def get_tree_cycle_masks_and_next_edge_indices_in_path_to_cycle(graph, parent_edge_indices,
            non_tree_edge_index):

        cycle_vertices_mask = common_utils.repeat_bool(False, graph.size)
        cycle_edges_mask = common_utils.repeat_bool(False, graph.edges_count)

        next_edge_indices_in_path_to_cycle = common_utils.repeat_int(-1, graph.size)

        non_tree_edge_vertex1 = graph.edges.vertex1[non_tree_edge_index]
        non_tree_edge_vertex2 = graph.edges.vertex2[non_tree_edge_index]

        cycle_vertices_mask[non_tree_edge_vertex1] = True
        cycle_vertices_mask[non_tree_edge_vertex2] = True

        no_vertices_overlapped_yet = True

        for start_vertex in [non_tree_edge_vertex1, non_tree_edge_vertex2]:

            current_vertex = start_vertex
            current_edge_index = parent_edge_indices[current_vertex]

            while current_edge_index != -1:

                cycle_edges_mask[current_edge_index] = (not cycle_edges_mask[current_edge_index])
                current_vertex = graph.edges.get_opposite_vertex(current_edge_index, current_vertex)

                if cycle_vertices_mask[current_vertex]:

                    if no_vertices_overlapped_yet:
                        no_vertices_overlapped_yet = False
                    else:
                        cycle_vertices_mask[current_vertex] = False
                        next_edge_indices_in_path_to_cycle[current_vertex] = current_edge_index

                else:
                    cycle_vertices_mask[current_vertex] = True

                current_edge_index = parent_edge_indices[current_vertex]

        return cycle_vertices_mask, cycle_edges_mask, next_edge_indices_in_path_to_cycle

    @staticmethod
    def iterate_tree_adjacency_costs_on_tree_cycle_side(graph, tree_edges_mask,
            total_descendants_costs, parent_edge_indices, start_vertex_on_cycle,
            start_edge_index_on_cycle, cycle_vertices_mask, cycle_edges_mask, add_end_marker):

        total_vertices_cost = graph.vertex_costs.sum()

        for edge_index in search_utils.iterate_subgraph_incidence_indices(graph, cycle_edges_mask,
                tree_edges_mask, start_vertex_on_cycle, start_edge_index_on_cycle):

            cycle_vertex = TreeCycles._get_cycle_vertex(edge_index, graph, cycle_vertices_mask)
            vertex_on_cycle_side = graph.edges.get_opposite_vertex(edge_index, cycle_vertex)

            if parent_edge_indices[cycle_vertex] == edge_index:
                yield total_vertices_cost - total_descendants_costs[cycle_vertex]
            else:
                yield total_descendants_costs[vertex_on_cycle_side]

        if add_end_marker:
            yield -1.0

    @staticmethod
    def shrink_cycle(graph, tree_edges_mask, total_descendants_costs, parent_edge_indices,
            cycle_vertices_mask, cycle_edges_mask, next_edge_indices_in_path_to_cycle,
            cost_inside_cycle, non_tree_edge_index, non_tree_edge_primary_vertex):

        non_tree_edge_secondary_vertex = \
                graph.edges.get_opposite_vertex(non_tree_edge_index, non_tree_edge_primary_vertex)

        first_internal_edge_index = graph.edges.get_next_edge_index(non_tree_edge_index,
                non_tree_edge_primary_vertex)

        internal_vertex = graph.edges.get_opposite_vertex(first_internal_edge_index,
                non_tree_edge_primary_vertex)

        second_internal_edge_index = graph.edges.get_next_edge_index(first_internal_edge_index,
                internal_vertex)

        if tree_edges_mask[first_internal_edge_index] or \
                tree_edges_mask[second_internal_edge_index]:

            return TreeCycles._shrink_cycle_if_one_of_internal_edges_is_on_tree(graph.vertex_costs,
                    tree_edges_mask, first_internal_edge_index, second_internal_edge_index,
                    non_tree_edge_primary_vertex, non_tree_edge_secondary_vertex, internal_vertex,
                    cycle_vertices_mask, cycle_edges_mask, cost_inside_cycle)

        tree_path_cost = TreeCycles._add_tree_path_to_cycle_masks_and_return_its_cost(graph,
                parent_edge_indices, cycle_vertices_mask, cycle_edges_mask,
                next_edge_indices_in_path_to_cycle, internal_vertex)

        adjacency_costs_inside_first_internal_cycle = \
                TreeCycles.iterate_tree_adjacency_costs_on_tree_cycle_side(graph, tree_edges_mask,
                total_descendants_costs, parent_edge_indices, non_tree_edge_primary_vertex,
                first_internal_edge_index, cycle_vertices_mask, cycle_edges_mask, True)

        adjacency_costs_inside_second_internal_cycle = \
                TreeCycles.iterate_tree_adjacency_costs_on_tree_cycle_side(graph, tree_edges_mask,
                total_descendants_costs, parent_edge_indices, internal_vertex,
                second_internal_edge_index, cycle_vertices_mask, cycle_edges_mask, True)

        cost_inside_first_internal_cycle = 0.0
        cost_inside_second_internal_cycle = 0.0

        for first_adjacency_cost, second_adjacency_cost in \
                zip(adjacency_costs_inside_first_internal_cycle,
                adjacency_costs_inside_second_internal_cycle):

            if first_adjacency_cost != -1.0 and second_adjacency_cost != -1.0:
                cost_inside_first_internal_cycle += first_adjacency_cost
                cost_inside_second_internal_cycle += second_adjacency_cost
            else:

                # last iteration
                if first_adjacency_cost == -1.0:
                    cost_inside_second_internal_cycle = cost_inside_cycle - tree_path_cost - \
                            cost_inside_first_internal_cycle
                else:
                    cost_inside_first_internal_cycle = cost_inside_cycle - tree_path_cost - \
                            cost_inside_second_internal_cycle

        if cost_inside_first_internal_cycle > cost_inside_second_internal_cycle:
            return cost_inside_first_internal_cycle, first_internal_edge_index, \
                    non_tree_edge_primary_vertex

        return cost_inside_second_internal_cycle, second_internal_edge_index, internal_vertex

    @staticmethod
    def _shrink_cycle_if_one_of_internal_edges_is_on_tree(vertex_costs, tree_edges_mask,
            first_internal_edge_index, second_internal_edge_index, non_tree_edge_primary_vertex,
            non_tree_edge_secondary_vertex, internal_vertex, cycle_vertices_mask, cycle_edges_mask,
            cost_inside_cycle):

        if not cycle_vertices_mask[internal_vertex]:
            cost_inside_cycle -= vertex_costs[internal_vertex]

        cycle_vertices_mask[internal_vertex] = True

        if tree_edges_mask[first_internal_edge_index]:
            cycle_edges_mask[first_internal_edge_index] = True
            return cost_inside_cycle, second_internal_edge_index, internal_vertex
        else:
            cycle_edges_mask[second_internal_edge_index] = True
            return cost_inside_cycle, first_internal_edge_index, non_tree_edge_primary_vertex

    @staticmethod
    def _add_tree_path_to_cycle_masks_and_return_its_cost(graph, parent_edge_indices,
            cycle_vertices_mask, cycle_edges_mask, next_edge_indices_in_path_to_cycle,
            internal_vertex):

        tree_path_cost = 0.0

        current_vertex = internal_vertex

        while not cycle_vertices_mask[current_vertex]:

            cycle_vertices_mask[current_vertex] = True
            tree_path_cost += graph.vertex_costs[current_vertex]

            if next_edge_indices_in_path_to_cycle[current_vertex] == -1:
                current_edge_index = parent_edge_indices[current_vertex]
            else:
                current_edge_index = next_edge_indices_in_path_to_cycle[current_vertex]
                next_edge_indices_in_path_to_cycle[current_vertex] = -1

            cycle_edges_mask[current_edge_index] = True
            current_vertex = graph.edges.get_opposite_vertex(current_edge_index, current_vertex)

        return tree_path_cost

    @staticmethod
    def iterate_vertices_on_cycle_side(graph, tree_edges_mask, start_vertex_on_cycle,
            start_edge_index_on_cycle, cycle_vertices_mask, cycle_edges_mask):

        stack = []
        used_vertex_flags = common_utils.repeat_bool(False, graph.size)

        for edge_index in search_utils.iterate_subgraph_incidence_indices(graph, cycle_edges_mask,
                tree_edges_mask, start_vertex_on_cycle, start_edge_index_on_cycle):

            cycle_vertex = TreeCycles._get_cycle_vertex(edge_index, graph, cycle_vertices_mask)
            vertex_on_cycle_side = graph.edges.get_opposite_vertex(edge_index, cycle_vertex)

            # TODO: make unified dfs method?

            if used_vertex_flags[vertex_on_cycle_side]:
                continue

            stack.append(vertex_on_cycle_side)
            used_vertex_flags[vertex_on_cycle_side] = True

            while len(stack) != 0:

                vertex = stack.pop()

                yield vertex

                for adjacent_vertex in graph.get_adjacent_vertices(vertex):

                    if not cycle_vertices_mask[adjacent_vertex] and \
                            not used_vertex_flags[adjacent_vertex]:
                        used_vertex_flags[adjacent_vertex] = True
                        stack.append(adjacent_vertex)
 
    @staticmethod
    def _get_cycle_vertex(edge_index, graph, cycle_vertices_mask):

        edge_vertex1 = graph.edges.vertex1[edge_index]
        edge_vertex2 = graph.edges.vertex2[edge_index]

        if cycle_vertices_mask[edge_vertex1]:
            return edge_vertex1

        return edge_vertex2

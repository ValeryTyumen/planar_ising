import numpy as np
from numba import jit
from numba.types import int32, boolean, float32
from . import bfs_tree, tree_cycles, separation_class
from .separation_class import SeparationClass
from .. import common_utils
from ..planar_graph import planar_graph_nb_type, PlanarGraph, planar_graph_constructor, \
        triangulator, search_utils


class PlanarSeparator:
    """
    Static class implementing Lipton-Tarjan algorithm.
    """

    @staticmethod
    def mark_separation(graph):
        """
        Separate the graph into three parts - FIRST_PART, SEPARATOR and SECOND_PART.

        Parameters
        ----------
        graph : PlanarGraph
            A normal planar graph instance (i.e. without multiple edges and loops).

        Returns
        -------
        array_like, SeparationClass
            Result separation.
        """

        return np.array([SeparationClass(value) for value in mark_separation(graph)], dtype=object)


@jit(int32[:](int32[:]), nopython=True)
def _fill_undefined_with_second_part(separation):

    new_separation = separation.copy()
    new_separation[separation == separation_class.UNDEFINED] = separation_class.SECOND_PART

    return new_separation

@jit(int32[:](int32, int32[:], int32[:]), nopython=True)
def _map_levels(new_vertices_count, new_vertices_mapping, bfs_levels):

    new_vertex_exists_mask = (new_vertices_mapping != -1)

    new_bfs_levels = common_utils.repeat_int(-1, new_vertices_count)

    new_bfs_levels[new_vertices_mapping[new_vertex_exists_mask]] = \
            bfs_levels[new_vertex_exists_mask]

    return new_bfs_levels

@jit(int32[:](int32[:], int32[:]), nopython=True)
def _get_new_vertices_mappings_composition(new_vertices_mapping1, new_vertices_mapping2):

    new_vertex_exists_mask = (new_vertices_mapping1 != -1)

    composition = common_utils.repeat_int(-1, len(new_vertices_mapping1))

    composition[new_vertex_exists_mask] = \
            new_vertices_mapping2[new_vertices_mapping1[new_vertex_exists_mask]]

    return composition

@jit(boolean[:](int32, int32[:], boolean[:]), nopython=True)
def _map_edges_mask(new_edges_count, new_edge_indices_mapping, edges_mask):

    new_edge_exists_mask = (new_edge_indices_mapping != -1)

    new_edges_mask = common_utils.repeat_bool(False, new_edges_count)

    new_edges_mask[new_edge_indices_mapping[new_edge_exists_mask]] = \
            edges_mask[new_edge_exists_mask]

    return new_edges_mask

@jit(int32[:](int32[:], int32[:]), nopython=True)
def _get_new_parent_edge_indices(parent_edge_indices, new_edge_indices_mapping):

    parent_edge_exists_mask = (parent_edge_indices != -1)

    new_parent_edge_indices = common_utils.repeat_int(-1, len(parent_edge_indices))

    new_parent_edge_indices[parent_edge_exists_mask] = \
            new_edge_indices_mapping[parent_edge_indices[parent_edge_exists_mask]]

    return new_parent_edge_indices

@jit(int32[:](int32[:], float32[:]), nopython=True)
def _mark_separation_when_components_less_than_one_third(connected_component_indices,
        connected_component_costs):

    first_part_components_count = 0
    first_part_cost = 0.0

    while first_part_cost <= 1/3 and first_part_components_count < \
            len(connected_component_costs):
        first_part_cost += connected_component_costs[first_part_components_count]
        first_part_components_count += 1

    separation = common_utils.repeat_int(separation_class.SECOND_PART,
            len(connected_component_indices))

    separation[connected_component_indices < first_part_components_count] = \
            separation_class.FIRST_PART

    return separation

@jit(int32[:](int32[:], int32), nopython=True)
def _mark_separation_when_components_less_than_two_thirds(connected_component_indices,
        max_component_index):

    separation = common_utils.repeat_int(separation_class.SECOND_PART,
            len(connected_component_indices))

    separation[connected_component_indices == max_component_index] = separation_class.FIRST_PART

    return separation

@jit(int32(float32[:]), nopython=True)
def _find_level_one(bfs_level_costs):

    level_one = 0
    cost_up_to_level_one = bfs_level_costs[level_one]

    while cost_up_to_level_one <= 1/2:
        level_one += 1
        cost_up_to_level_one += bfs_level_costs[level_one]

    return level_one

@jit(int32(int32[:], int32, int32), nopython=True)
def _find_level_zero(bfs_level_sizes, level_one, vertices_count_up_to_level_one):

    level_zero = 0

    threshold = 2*np.sqrt(vertices_count_up_to_level_one)

    for level in range(level_one + 1):
        if bfs_level_sizes[level] + 2*(level_one - level_zero) <= threshold:
            level_zero = level

    return level_zero

@jit(int32(int32[:], int32, int32), nopython=True)
def _find_level_two(bfs_level_sizes, level_one, vertices_count_behind_level_one):

    threshold = 2*np.sqrt(vertices_count_behind_level_one)

    for level in range(level_one + 1, len(bfs_level_sizes)):
        if bfs_level_sizes[level] + 2*(level - level_one - 1) <= threshold:
            return level

    return len(bfs_level_sizes)

@jit(int32[:](int32[:], float32[:], int32, int32, float32), nopython=True)
def _mark_separation_if_cost_between_levels_is_less_than_two_thirds(bfs_levels,
        bfs_level_costs, level_zero, level_two, cost_between_levels):

    cost_below_level_zero = bfs_level_costs[:level_zero].sum()
    cost_behind_level_two = bfs_level_costs[level_two + 1:].sum()

    levels_count = len(bfs_level_costs)

    level_separation = common_utils.repeat_int(separation_class.UNDEFINED, levels_count)

    level_separation[level_zero] = separation_class.SEPARATOR

    if level_two != len(level_separation):
        level_separation[level_two] = separation_class.SEPARATOR

    if cost_between_levels >= max(cost_below_level_zero, cost_behind_level_two):
        level_separation[level_zero + 1:level_two] = separation_class.FIRST_PART
    elif cost_below_level_zero >= max(cost_between_levels, cost_behind_level_two):
        level_separation[:level_zero] = separation_class.FIRST_PART
    else:
        level_separation[level_two + 1:] = separation_class.FIRST_PART

    level_separation = _fill_undefined_with_second_part(level_separation)

    separation = common_utils.repeat_int(separation_class.UNDEFINED, len(bfs_levels))

    level_exists_mask = (bfs_levels != -1)

    separation[level_exists_mask] = level_separation[bfs_levels[level_exists_mask]]

    return separation

@jit(int32[:](float32[:], int32[:]), nopython=True)
def _swap_separation_parts_if_cost_of_first_part_is_smaller(vertex_costs, separation):

    first_part_cost = vertex_costs[separation == separation_class.FIRST_PART].sum()
    second_part_cost = vertex_costs[separation == separation_class.SECOND_PART].sum()

    if first_part_cost >= second_part_cost:
        return separation

    new_separation = separation.copy()
    new_separation[separation == separation_class.FIRST_PART] = separation_class.SECOND_PART
    new_separation[separation == separation_class.SECOND_PART] = separation_class.FIRST_PART

    return new_separation

@jit(int32[:](planar_graph_nb_type, int32), nopython=True)
def _mark_separation_of_graph_with_small_radius(graph, center_vertex):

    # triangulator only works when there are at least three vertices
    # take all component as separator

    if graph.size < 3:
        return common_utils.repeat_int(separation_class.SEPARATOR, graph.size)

    initial_graph = graph

    # Step 7

    bfs_tree_edges_mask = bfs_tree.construct_bfs_tree_edges_mask(center_vertex, graph)

    parent_edge_indices, total_descendants_costs = \
            bfs_tree.record_bfs_tree_parent_edge_indices_and_total_descendants_costs(graph,
            center_vertex, bfs_tree_edges_mask)

    new_edge_indices_mapping, graph = triangulator.triangulate(graph)

    bfs_tree_edges_mask = _map_edges_mask(graph.edges_count, new_edge_indices_mapping,
            bfs_tree_edges_mask)

    parent_edge_indices = _get_new_parent_edge_indices(parent_edge_indices,
            new_edge_indices_mapping)

    # Step 8

    # non-tree edges exist surely
    non_tree_edge_index = np.where(np.logical_not(bfs_tree_edges_mask))[0][0]

    cycle_vertices_mask, cycle_edges_mask, next_edge_indices_in_path_to_cycle = \
            tree_cycles.get_tree_cycle_masks_and_next_edge_indices_in_path_to_cycle(graph,
            parent_edge_indices, non_tree_edge_index)
    cost_on_cycle = graph.vertex_costs[cycle_vertices_mask].sum()

    non_tree_edge_primary_vertex = graph.edges.vertex1[non_tree_edge_index]

    tree_adjacency_costs_inside_cycle = \
            tree_cycles.iterate_tree_adjacency_costs_on_tree_cycle_side(graph,
            bfs_tree_edges_mask, total_descendants_costs, parent_edge_indices,
            non_tree_edge_primary_vertex, non_tree_edge_index, cycle_vertices_mask,
            cycle_edges_mask, False)

    cost_inside_cycle = 0.0

    for cost in tree_adjacency_costs_inside_cycle:
        cost_inside_cycle += cost
 
    cost_outside_cycle = graph.vertex_costs.sum() - cost_on_cycle - cost_inside_cycle

    if cost_outside_cycle > cost_inside_cycle:
        non_tree_edge_primary_vertex = graph.edges.vertex2[non_tree_edge_index]
        cost_inside_cycle = cost_outside_cycle

    # Step 9

    while cost_inside_cycle > 2/3:

        cost_inside_cycle, non_tree_edge_index, non_tree_edge_primary_vertex = \
                tree_cycles.shrink_cycle(graph, bfs_tree_edges_mask, total_descendants_costs,
                parent_edge_indices, cycle_vertices_mask, cycle_edges_mask,
                next_edge_indices_in_path_to_cycle, cost_inside_cycle, non_tree_edge_index,
                non_tree_edge_primary_vertex)

    # Current `shrink_cycle` algorithm doesn't "clean" vertices and edges outside current cycle
    cycle_vertices_mask, cycle_edges_mask, _ = \
            tree_cycles.get_tree_cycle_masks_and_next_edge_indices_in_path_to_cycle(graph,
            parent_edge_indices, non_tree_edge_index)

    separation = common_utils.repeat_int(separation_class.UNDEFINED, graph.size)

    separation[cycle_vertices_mask] = separation_class.SEPARATOR

    for vertex in tree_cycles.iterate_vertices_on_cycle_side(graph, bfs_tree_edges_mask,
            non_tree_edge_primary_vertex, non_tree_edge_index, cycle_vertices_mask,
            cycle_edges_mask):

        separation[vertex] = separation_class.FIRST_PART

    separation = _fill_undefined_with_second_part(separation)

    return separation

@jit(int32[:](planar_graph_nb_type, int32[:], int32), nopython=True)
def _mark_separation_for_one_connected_component(graph, connected_component_indices,
        component_index):

    # Step 3

    connected_component_indices = connected_component_indices

    bfs_tree_root = np.where(connected_component_indices == component_index)[0][0]

    bfs_levels = bfs_tree.construct_bfs_levels(bfs_tree_root, graph)

    bfs_level_exists_mask = (bfs_levels != -1)

    bfs_level_sizes = np.bincount(bfs_levels[bfs_level_exists_mask]).astype(np.int32)

    # Step 4

    bfs_level_costs = np.bincount(bfs_levels[bfs_level_exists_mask],
            graph.vertex_costs[bfs_level_exists_mask])

    level_one = _find_level_one(bfs_level_costs)

    vertices_count_up_to_level_one = bfs_level_sizes[:level_one + 1].sum()
    vertices_count_behind_level_one = bfs_level_sizes[level_one + 1:].sum()

    # Step 5

    level_zero = _find_level_zero(bfs_level_sizes, level_one,
            vertices_count_up_to_level_one)
    level_two = _find_level_two(bfs_level_sizes, level_one,
            vertices_count_behind_level_one)

    cost_between_levels = bfs_level_costs[level_zero + 1:level_two].sum()

    if cost_between_levels < 2/3:
        return _mark_separation_if_cost_between_levels_is_less_than_two_thirds(bfs_levels,
                bfs_level_costs, level_zero, level_two, cost_between_levels)

    # Step 6

    vertices_below_level_two_mask = np.logical_and(bfs_levels != -1, bfs_levels < level_two)

    new_vertices_mapping1, _, new_graph = planar_graph_constructor.construct_subgraph(graph,
            vertices_below_level_two_mask, common_utils.repeat_bool(True, graph.edges_count))

    bfs_tree_root = new_vertices_mapping1[bfs_tree_root]

    new_bfs_levels = _map_levels(new_graph.size, new_vertices_mapping1, bfs_levels)
    bfs_tree_edges_mask = bfs_tree.construct_bfs_tree_edges_mask(bfs_tree_root, new_graph)

    collapsed_vertex, new_vertices_mapping2, _, new_graph = \
            bfs_tree.collapse_bfs_subtree(new_graph, new_bfs_levels, level_zero,
            bfs_tree_edges_mask)

    new_vertices_mapping = _get_new_vertices_mappings_composition(new_vertices_mapping1,
            new_vertices_mapping2)

    new_graph_separation = _mark_separation_of_graph_with_small_radius(new_graph,
            collapsed_vertex)

    new_graph_separation = \
            _swap_separation_parts_if_cost_of_first_part_is_smaller(new_graph.vertex_costs,
            new_graph_separation)

    separation = common_utils.repeat_int(separation_class.UNDEFINED, graph.size)

    new_vertex_exists_mask = (new_vertices_mapping != -1)

    separation[new_vertex_exists_mask] = \
            new_graph_separation[new_vertices_mapping[new_vertex_exists_mask]]

    separation[separation == separation_class.SECOND_PART] = separation_class.UNDEFINED

    separation[np.logical_or(bfs_levels == level_zero, bfs_levels == level_two)] = \
            separation_class.SEPARATOR

    separation = _fill_undefined_with_second_part(separation)

    return separation

@jit(int32[:](planar_graph_nb_type), nopython=True)
def mark_separation(graph):
    """
    Separate the graph into three parts - FIRST_PART(0), SEPARATOR(2) and SECOND_PART(1).

    Parameters
    ----------
    graph : PlanarGraph
        A normal planar graph instance (i.e. without multiple edges and loops).

    Returns
    -------
    array_like, int32
        Result separation.
    """

    # Step 2

    connected_component_indices = search_utils.color_connected_components(graph)
    connected_component_costs = np.bincount(connected_component_indices,
            graph.vertex_costs)

    max_component_cost = connected_component_costs.max()
    max_component_index = np.where(connected_component_costs == max_component_cost)[0][0]

    if max_component_cost <= 1/3:
        return _mark_separation_when_components_less_than_one_third(
                connected_component_indices, connected_component_costs)

    if max_component_cost <= 2/3:
        return _mark_separation_when_components_less_than_two_thirds(
                connected_component_indices, max_component_index)

    separation = _mark_separation_for_one_connected_component(graph,
            connected_component_indices, max_component_index)

    separation = _swap_separation_parts_if_cost_of_first_part_is_smaller(graph.vertex_costs,
            separation)

    separation = _fill_undefined_with_second_part(separation)

    return separation

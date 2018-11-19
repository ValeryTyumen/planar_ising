import numpy as np
from numba import jit
from numba.types import void, Tuple, boolean, int32, float32
from . import utils
from .. import common_utils
from ..planar_graph import planar_graph_edges_nb_type, PlanarGraphEdges, planar_graph_nb_type, \
        PlanarGraph, planar_graph_constructor, search_utils


@jit(void(int32, planar_graph_edges_nb_type, int32, int32[:]), nopython=True)
def _set_adjacent_vertex_level(vertex, edges, incident_edge_index, levels):

    adjacent_vertex = edges.get_opposite_vertex(incident_edge_index, vertex)
    levels[adjacent_vertex] = levels[vertex] + 1

_set_levels = search_utils.make_traverse_graph_via_bfs(_set_adjacent_vertex_level, int32[:])

@jit(int32[:](int32, planar_graph_nb_type), nopython=True)
def construct_bfs_levels(root, graph):

    levels = common_utils.repeat_int(-1, graph.size)
    levels[root] = 0

    used_vertex_flags = common_utils.repeat_bool(False, graph.size)

    _set_levels(root, graph, used_vertex_flags, levels)

    return levels

@jit(void(int32, planar_graph_edges_nb_type, int32, boolean[:]), nopython=True)
def _add_edge_to_tree(vertex, edges, incident_edge_index, tree_edges_mask):

    tree_edges_mask[incident_edge_index] = True

_construct_tree_edges_mask = search_utils.make_traverse_graph_via_bfs(_add_edge_to_tree, boolean[:])

@jit(boolean[:](int32, planar_graph_nb_type), nopython=True)
def construct_bfs_tree_edges_mask(root, graph):

    tree_edges_mask = common_utils.repeat_bool(False, graph.edges_count)

    used_vertex_flags = common_utils.repeat_bool(False, graph.size)

    _construct_tree_edges_mask(root, graph, used_vertex_flags, tree_edges_mask)

    return tree_edges_mask

@jit(Tuple((int32[:], int32[:]))(planar_graph_nb_type, int32[:], int32, boolean[:]), nopython=True)
def _get_ordered_bfs_subtree_adjacencies_and_incidence_indices(graph, bfs_levels,
        subtree_max_level, bfs_tree_edges_mask):

    bfs_subtree_edges_mask = bfs_tree_edges_mask

    for edge_index in range(graph.edges_count):

        edge_vertex1 = graph.edges.vertex1[edge_index]
        edge_vertex2 = graph.edges.vertex2[edge_index]

        if max(bfs_levels[edge_vertex1], bfs_levels[edge_vertex2]) > subtree_max_level:
            bfs_subtree_edges_mask[edge_index] = False

    ordered_bfs_subtree_adjacencies_list = []
    ordered_bfs_subtree_incidence_indices_list = []

    if not np.any(bfs_subtree_edges_mask):

        bfs_tree_root = np.where(bfs_levels == 0)[0][0]

        for vertex in graph.get_adjacent_vertices(bfs_tree_root):
            ordered_bfs_subtree_adjacencies_list.append(vertex)

        for edge_index in graph.get_incident_edge_indices(bfs_tree_root):
            ordered_bfs_subtree_incidence_indices_list.append(edge_index)

    else:

        bfs_subtree_adjacencies_mask = common_utils.repeat_bool(False, graph.size)

        start_edge_index_on_subtree = np.where(bfs_subtree_edges_mask)[0][0]
        start_vertex_on_subtree = graph.edges.vertex1[start_edge_index_on_subtree]

        for edge_index in utils.iterate_subgraph_incidence_indices(graph, bfs_subtree_edges_mask,
                common_utils.repeat_bool(True, graph.edges_count), start_vertex_on_subtree,
                start_edge_index_on_subtree):

            edge_vertex1 = graph.edges.vertex1[edge_index]
            edge_vertex2 = graph.edges.vertex2[edge_index]   

            if max(bfs_levels[edge_vertex1], bfs_levels[edge_vertex2]) <= subtree_max_level:
                continue

            if bfs_levels[edge_vertex1] > subtree_max_level:
                adjacent_vertex = edge_vertex1
            else:
                adjacent_vertex = edge_vertex2

            if not bfs_subtree_adjacencies_mask[adjacent_vertex]:

                bfs_subtree_adjacencies_mask[adjacent_vertex] = True

                ordered_bfs_subtree_adjacencies_list.append(adjacent_vertex)
                ordered_bfs_subtree_incidence_indices_list.append(edge_index)

    ordered_bfs_subtree_adjacencies = np.array(ordered_bfs_subtree_adjacencies_list)
    ordered_bfs_subtree_incidence_indices = np.array(ordered_bfs_subtree_incidence_indices_list)

    return ordered_bfs_subtree_adjacencies, ordered_bfs_subtree_incidence_indices

@jit(Tuple((int32, int32[:], int32[:], planar_graph_nb_type))(planar_graph_nb_type, int32[:], int32,
        boolean[:]), nopython=True)
def collapse_bfs_subtree(graph, bfs_levels, subtree_max_level, bfs_tree_edges_mask):

    ordered_bfs_subtree_adjacencies, ordered_bfs_subtree_incidence_indices = \
            _get_ordered_bfs_subtree_adjacencies_and_incidence_indices(graph, bfs_levels,
            subtree_max_level, bfs_tree_edges_mask)

    bfs_subtree_mask = (bfs_levels <= subtree_max_level)

    new_vertices_mapping, new_edge_indices_mapping, new_graph = \
            planar_graph_constructor.construct_subgraph(graph, np.logical_not(bfs_subtree_mask),
            common_utils.repeat_bool(True, graph.edges_count))

    bfs_subtree_vertex = new_graph.size

    ordered_new_edges = PlanarGraphEdges(len(ordered_bfs_subtree_adjacencies))

    for vertex in ordered_bfs_subtree_adjacencies:
        ordered_new_edges.append(bfs_subtree_vertex, new_vertices_mapping[vertex])

    subgraph_edges_count = new_graph.edges_count
 
    new_graph.edges.extend(ordered_new_edges)

    for index, (edge_index, adjacent_vertex) in enumerate(zip(ordered_bfs_subtree_incidence_indices,
            ordered_bfs_subtree_adjacencies)):

        new_edge_index = index + subgraph_edges_count

        new_adjacent_vertex = new_vertices_mapping[adjacent_vertex]

        new_graph.incident_edge_example_indices[new_adjacent_vertex] = new_edge_index

        root_next_edge_index = (index + 1)%ordered_new_edges.size + subgraph_edges_count
        new_graph.edges.set_next_edge(new_edge_index, bfs_subtree_vertex, root_next_edge_index)

        next_edge_index = graph.edges.get_next_edge_index(edge_index, adjacent_vertex)

        while new_edge_indices_mapping[next_edge_index] == -1 and next_edge_index != edge_index:
            next_edge_index = graph.edges.get_next_edge_index(next_edge_index, adjacent_vertex)

        if next_edge_index == edge_index:
            new_adjacent_vertex_next_edge_index = new_edge_index
            new_adjacent_vertex_previous_edge_index = new_edge_index
        else:

            new_adjacent_vertex_next_edge_index = new_edge_indices_mapping[next_edge_index]
            new_adjacent_vertex_previous_edge_index = \
                    new_graph.edges.get_previous_edge_index(new_adjacent_vertex_next_edge_index,
                    new_adjacent_vertex)

        new_graph.edges.set_next_edge(new_edge_index, new_adjacent_vertex,
                new_adjacent_vertex_next_edge_index)
        new_graph.edges.set_previous_edge(new_edge_index, new_adjacent_vertex,
                new_adjacent_vertex_previous_edge_index)

    root_incident_edge_example_index = -1

    if ordered_new_edges.size != 0:
        root_incident_edge_example_index = subgraph_edges_count

    new_new_graph_vertex_costs = np.concatenate((new_graph.vertex_costs,
            np.array([graph.vertex_costs[bfs_subtree_mask].sum()])))

    new_new_graph_incident_edge_example_indices = \
            np.concatenate((new_graph.incident_edge_example_indices,
            np.array([root_incident_edge_example_index], dtype=np.int32)))

    return bfs_subtree_vertex, new_vertices_mapping, new_edge_indices_mapping, \
            PlanarGraph(new_new_graph_vertex_costs, new_new_graph_incident_edge_example_indices,
            new_graph.edges)

@jit(void(int32, planar_graph_edges_nb_type, int32, float32[:]), nopython=True)
def _update_parent_total_descendants_costs(vertex, edges, parent_edge_index,
        total_descendants_costs):

    parent_vertex = edges.get_opposite_vertex(parent_edge_index, vertex)
    total_descendants_costs[parent_vertex] += total_descendants_costs[vertex]

_set_total_descendants_costs = \
        search_utils.make_traverse_graph_via_post_order_dfs(_update_parent_total_descendants_costs,
        float32[:])

@jit(Tuple((int32[:], float32[:]))(planar_graph_nb_type, int32, boolean[:]), nopython=True)
def record_bfs_tree_parent_edge_indices_and_total_descendants_costs(graph, bfs_tree_root,
        bfs_tree_edges_mask):

    total_descendants_costs = graph.vertex_costs.copy()

    parent_edge_indices = _set_total_descendants_costs(bfs_tree_root, graph,
            bfs_tree_edges_mask, total_descendants_costs)

    return parent_edge_indices, total_descendants_costs

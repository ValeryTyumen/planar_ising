import numpy as np


class DecompGraph:
    """
    A data structure, representing a tree decomposition of graph into planar and O(1)-sized
    components (nodes), sharing <= 3 vertices.
    """

    def __init__(self):
        """
        Graph initialization.
        """

        self._nodes = []
        self._is_small_node = []
        self._connection_vertices = []
        self._adjacent_node_indices = []
        self._adjacent_node_connection_vertices = []

        self._graph_enumerated = False

    def add_component(self, is_small_node, node):

        if self._graph_enumerated:
            raise RuntimeError('Graph is already enumerated.')

        self._nodes.append(node)
        self._is_small_node.append(is_small_node)
        self._connection_vertices.append([])
        self._adjacent_node_indices.append([])
        self._adjacent_node_connection_vertices.append([])

    def add_connection(self, node1_index, node2_index, node1_connection_vertices,
            node2_connection_vertices):
 
        if self._graph_enumerated:
            raise RuntimeError('Graph is already enumerated.')

        self._connection_vertices[node1_index].append(node1_connection_vertices)
        self._connection_vertices[node2_index].append(node2_connection_vertices)

        self._adjacent_node_indices[node1_index].append(node2_index)
        self._adjacent_node_indices[node2_index].append(node1_index)

        self._adjacent_node_connection_vertices[node1_index].append(node2_connection_vertices)
        self._adjacent_node_connection_vertices[node2_index].append(node1_connection_vertices)

    def traverse(self):

        yield from self._traverse_subtree(0, -1, -1)

    def _traverse_subtree(self, node_index, parent_index, adjacency_index_in_parent):

        yield node_index, parent_index, adjacency_index_in_parent

        for adjacency_index, adjacent_node_index in \
                enumerate(self._adjacent_node_indices[node_index]):
            if adjacent_node_index != parent_index:
                yield from self._traverse_subtree(adjacent_node_index, node_index, adjacency_index)

    def enumerate(self):

        self._graph_vertices = [None]*len(self._nodes)
        self._graph_edge_indices = [None]*len(self._nodes)

        current_graph_vertex = 0
        current_graph_edge_index = 0

        for node_index, parent_index, adjacency_index_in_parent in self.traverse():

            node = self._nodes[node_index]

            if self._is_small_node[node_index]:
                size = node.max() + 1
                edges_count = node.shape[0]
            else:
                size = node.size
                edges_count = node.edges_count

            if parent_index == -1:

                self._graph_vertices[node_index] = np.arange(size)
                self._graph_edge_indices[node_index] = np.arange(edges_count)

                current_graph_vertex += size
                current_graph_edge_index += edges_count

            else:

                self._graph_vertices[node_index] = -np.ones(size, dtype=int)
                self._graph_edge_indices[node_index] = -np.ones(edges_count, dtype=int)

                graph_vertices = self._graph_vertices[node_index]
                graph_edge_indices = self._graph_edge_indices[node_index]

                connecting_to_parent_vertices = self._adjacent_node_connection_vertices\
                        [parent_index][adjacency_index_in_parent]

                graph_vertices[connecting_to_parent_vertices] = self._graph_vertices[parent_index]\
                        [self._connection_vertices[parent_index][adjacency_index_in_parent]]

                connection_size = connecting_to_parent_vertices.shape[0]

                non_connection_to_parent_mask = (graph_vertices == -1)

                graph_vertices[graph_vertices == -1] = np.arange(size - connection_size) + \
                        current_graph_vertex

                if (not self._is_small_node[node_index]) and ((size < 3 and connection_size == 0) \
                        or (size - connection_size + 1 < 3 and connection_size > 0)):

                    self._is_small_node[node_index] = True
                    self._nodes[node_index] = np.concatenate((node.edges.vertex1[:, None],
                            node.edges.vertex2[:, None]), axis=1)
                    node = self._nodes[node_index]

                current_graph_vertex += size - connection_size

                if self._is_small_node[node_index]:
                    graph_edges_mask = (non_connection_to_parent_mask[node[:, 0]] | \
                            non_connection_to_parent_mask[node[:, 1]])
                else:
                    graph_edges_mask = (non_connection_to_parent_mask[node.edges.vertex1] | \
                            non_connection_to_parent_mask[node.edges.vertex2])

                graph_edge_indices[graph_edges_mask] = np.arange(graph_edges_mask.sum()) + \
                        current_graph_edge_index

                current_graph_edge_index += graph_edges_mask.sum()

        self._size, self._edges_count = current_graph_vertex, current_graph_edge_index

        self._graph_enumerated = True
 
    def get_edges(self):

        edges = np.zeros((self._edges_count, 2), dtype=int)

        for node_index in range(len(self._nodes)):

            node = self._nodes[node_index]
            graph_vertices = self._graph_vertices[node_index]
            graph_edge_indices = self._graph_edge_indices[node_index]

            if self._is_small_node[node_index]:
                edges[graph_edge_indices[graph_edge_indices != -1]] = \
                        graph_vertices[node[graph_edge_indices != -1]]
            else:
                edges[graph_edge_indices[graph_edge_indices != -1], 0] = \
                        graph_vertices[node.edges.vertex1[graph_edge_indices != -1]]
                edges[graph_edge_indices[graph_edge_indices != -1], 1] = \
                        graph_vertices[node.edges.vertex2[graph_edge_indices != -1]]

        return edges

    @property
    def size(self):

        if not self._graph_enumerated:
            raise RuntimeError('Graph is not yet enumerated.')

        return self._size

    @property
    def edges_count(self):

        if not self._graph_enumerated:
            raise RuntimeError('Graph is not yet enumerated.')

        return self._edges_count

    @property
    def nodes_count(self):

        return len(self._nodes)

    @property
    def nodes(self):

        return self._nodes

    @property
    def is_small_node(self):

        return self._is_small_node

    @property
    def connection_vertices(self):

        return self._connection_vertices

    @property
    def adjacent_node_indices(self):

        return self._adjacent_node_indices

    @property
    def adjacent_node_connection_vertices(self):

        return self._adjacent_node_connection_vertices

    @property
    def graph_vertices(self):

        return self._graph_vertices

    @property
    def graph_edge_indices(self):

        return self._graph_edge_indices

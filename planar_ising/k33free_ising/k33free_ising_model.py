import numpy as np
from .k5_ising_model import K5IsingModel
from . import utils


class K33freeIsingModel:
    """
    A data structure, representing a tree decomposition of K33 free zero-field Ising model into
    planar and K5 components (nodes), sharing virtual edges. This decomposition is somewhat more
    general than a tree of triconnected components, but inference and sampling algorithm still
    applies to it.
    """

    def __init__(self):
        """
        Model initialization.
        """

        self._nodes = []
        self._node_types = []

        self._virtual_edge_indices = []
        self._adjacent_node_indices = []
        self._adjacent_node_virtual_edge_indices = []

        self._model_vertices_enumerated = False

    def add_component(self, ising_model):
        """
        Add a component to model. Can only be executed before `enumerate_model_vertices`.

        Parameters
        ----------
        ising_model : K5IsingModel or PlanarIsingModel
            Ising model component.
        """

        if self._model_vertices_enumerated:
            raise RuntimeError('Model vertices are already enumerated.')

        if ising_model.__class__.__name__ == 'K5IsingModel':
            ising_model_type = 'k5'
        else:
            ising_model_type = 'planar'

        self._nodes.append(ising_model)
        self._node_types.append(ising_model_type)

        self._virtual_edge_indices.append([])
        self._adjacent_node_indices.append([])
        self._adjacent_node_virtual_edge_indices.append([])

    def add_virtual_edge(self, node1_index, node2_index, node1_edge_index, node2_edge_index):
        """
        Assign a virtual edge between two components (nodes). Can only be executed before
        `enumerate_model_vertices`.

        Parameters
        ----------
        node1_index : int
            First node index.
        node2_index : int
            Second node index.
        node1_edge_index : int
            Index of edge to become virtual in the first node.
        node2_edge_index : int
            Index of edge to become virtual in the second node.
        """
 
        if self._model_vertices_enumerated:
            raise RuntimeError('Model vertices are already enumerated.')

        self._virtual_edge_indices[node1_index].append(node1_edge_index)
        self._virtual_edge_indices[node2_index].append(node2_edge_index)

        self._adjacent_node_indices[node1_index].append(node2_index)
        self._adjacent_node_indices[node2_index].append(node1_index)

        self._adjacent_node_virtual_edge_indices[node1_index].append(node2_edge_index)
        self._adjacent_node_virtual_edge_indices[node2_index].append(node1_edge_index)

    def _enumerate_model_vertices_in_subtree(self, node_index, parent_virtual_edge_index,
            parent_virtual_edge_model_vertices, current_model_vertex):

        if self._node_types[node_index] == 'k5':
            node_size = 5
        else:
            node_size = self._nodes[node_index].graph.size

        if parent_virtual_edge_index == -1:

            self._model_vertices[node_index] = np.arange(node_size) + current_model_vertex
            node_model_vertices = self._model_vertices[node_index]

            current_model_vertex += node_size

        else:

            self._model_vertices[node_index] = -np.ones(node_size, dtype=int)

            node_model_vertices = self._model_vertices[node_index]

            parent_virtual_edge_vertex1, parent_virtual_edge_vertex2 = \
                    utils.get_edge_vertices(self, node_index, parent_virtual_edge_index)

            node_model_vertices[parent_virtual_edge_vertex1] = parent_virtual_edge_model_vertices[0]
            node_model_vertices[parent_virtual_edge_vertex2] = parent_virtual_edge_model_vertices[1]

            node_model_vertices[node_model_vertices == -1] = np.arange(node_size - 2) + \
                    current_model_vertex

            current_model_vertex += node_size - 2

        for virtual_edge_index, adjacent_node_index, adjacent_node_virtual_edge_index in \
                zip(self._virtual_edge_indices[node_index],
                self._adjacent_node_indices[node_index],
                self._adjacent_node_virtual_edge_indices[node_index]):

            if virtual_edge_index != parent_virtual_edge_index:

                virtual_edge_vertex1, virtual_edge_vertex2 = utils.get_edge_vertices(self,
                        node_index, virtual_edge_index)

                virtual_edge_model_vertices = [
                        node_model_vertices[virtual_edge_vertex1],
                        node_model_vertices[virtual_edge_vertex2]
                ]

                current_model_vertex = \
                        self._enumerate_model_vertices_in_subtree(adjacent_node_index,
                        adjacent_node_virtual_edge_index, virtual_edge_model_vertices,
                        current_model_vertex)

        return current_model_vertex

    def enumerate_model_vertices(self):
        """
        Introduce a common enumeration of vertices in all components. No topology modification is
        possible after this method call.
        """

        self._model_vertices = [None]*len(self._nodes)

        self._size = self._enumerate_model_vertices_in_subtree(0, -1, None, 0)

        self._model_vertices_enumerated = True

    def get_minus_energy(self, spin_values):
        """
        Get log-weight of spin configuration.

        Parameters
        ----------
        spin_values : array_like, int32
            Spin values of the configuration in { -1, +1 }.

        Returns
        -------
        float
            Log-weight of configuration (minus-energy).
        """

        if not self._model_vertices_enumerated:
            raise RuntimeError('Model vertices are not yet enumerated.')

        minus_energy = 0.0

        for node, model_vertices in zip(self._nodes, self._model_vertices):

            node_spin_values = spin_values[model_vertices]

            minus_energy += node.get_minus_energy(node_spin_values)

        return minus_energy

    @property
    def size(self):

        if not self._model_vertices_enumerated:
            raise RuntimeError('Model vertices are not yet enumerated.')

        return self._size

    @property
    def nodes_count(self):

        return len(self._nodes)

    @property
    def nodes(self):

        return self._nodes

    @property
    def node_types(self):

        return self._node_types

    @property
    def virtual_edge_indices(self):

        return self._virtual_edge_indices

    @property
    def adjacent_node_indices(self):

        return self._adjacent_node_indices

    @property
    def adjacent_node_virtual_edge_indices(self):

        return self._adjacent_node_virtual_edge_indices

    @property
    def model_vertices(self):

        return self._model_vertices

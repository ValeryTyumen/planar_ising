from .k5_ising_model import K5IsingModel


def get_edge_vertices(k33free_model, node_index, edge_index):

    if k33free_model.node_types[node_index] == 'k5':
        return K5IsingModel.EDGE_LIST[edge_index, 0], K5IsingModel.EDGE_LIST[edge_index, 1]

    edges = k33free_model.nodes[node_index].graph.edges

    return edges.vertex1[edge_index], edges.vertex2[edge_index]

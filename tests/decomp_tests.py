import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time

from planar_ising import PlanarGraphGenerator, DecompGraph, \
        DecompInferenceAndSampling, SmallInferenceAndSampling

np.random.seed(42)
matplotlib.rcParams.update({'font.size': 15})


def make_small_graph(size):

    edges = []

    for vertex1 in range(size):
        for vertex2 in range(vertex1 + 1, size):
            edges.append([vertex1, vertex2])

    return np.array(edges)

def choose_connection_vertices(graph, node_index, connection_size):

    node = graph.nodes[node_index]

    if graph.is_small_node[node_index]:
        return np.random.choice(node.max() + 1, size=connection_size, replace=False)

    connection_vertices = [np.random.choice(node.size)]

    if connection_size > 1:
        second_vertex_choices = list(node.get_adjacent_vertices(connection_vertices[0]))
        connection_vertices.append(np.random.choice(second_vertex_choices))

    if connection_size == 3:
        third_vertex_choices = list(set(second_vertex_choices) & \
                set(node.get_adjacent_vertices(connection_vertices[-1])))
        connection_vertices.append(np.random.choice(third_vertex_choices))

    return np.array(connection_vertices)

def generate_graph(size):

    max_small_size = 8

    graph = DecompGraph()

    node_size = 3 + np.random.choice(min(size - 2, max_small_size) - 2)
    graph.add_component(True, make_small_graph(node_size))

    graph_size = node_size

    while graph_size != size:

        connection_size = 1 + np.random.choice(3)

        if np.random.rand() > 0.5:

            if size - graph_size <= 3:
                node_size = connection_size + size - graph_size
            else:
                node_size = connection_size + 2 + np.random.choice(min(size - graph_size - 3,
                        max_small_size - connection_size - 1))

            node = make_small_graph(node_size)

            graph.add_component(True, node)
            graph_size += node_size - connection_size

        else:

            if size - graph_size <= 3:
                node_size = connection_size + size - graph_size
            else:
                node_size = connection_size + 2 + np.random.choice(size - graph_size - 3)
                
            node = PlanarGraphGenerator.generate_random_graph(node_size, 1.0)

            graph.add_component(False, node)
            graph_size += node_size - connection_size

        node_index = graph.nodes_count - 1
            
        connection_vertices = choose_connection_vertices(graph, node_index,
                connection_size)

        parent_index = np.random.choice(graph.nodes_count - 1)

        connection_vertices_in_parent = choose_connection_vertices(graph, parent_index,
                connection_size)

        graph.add_connection(parent_index, node_index, connection_vertices_in_parent,
                connection_vertices)

    graph.enumerate()

    return graph

def simulate_and_test_logpf_computation(interaction_values_std):

    start = time.time()

    sizes = np.arange(10, 16)

    models_per_size = 1000

    maximal_relative_error = 0.0

    for size in sizes:
        for sample_index in range(models_per_size):

            graph = generate_graph(size)
            inference = DecompInferenceAndSampling(graph)
            inference.prepare()

            interaction_values = np.random.normal(scale=interaction_values_std,
                    size=graph.edges_count)

            logpf, marginals = inference.compute_logpf(interaction_values,
                    with_marginals=True)

            bf_inference = SmallInferenceAndSampling(graph.get_edges(),
                    np.array([], dtype=int))
            bf_logpf, bf_marginals = bf_inference.compute_logpf(interaction_values,
                    np.array([], dtype=int), with_marginals=True)

            relative_error = np.absolute((logpf - bf_logpf)/bf_logpf)

            if relative_error > maximal_relative_error:
                maximal_relative_error = relative_error

            relative_error = np.linalg.norm(marginals - bf_marginals)/\
                    np.linalg.norm(bf_marginals)

    print('{0} random models of sizes {1} were evaluated.'.format(
            models_per_size*len(sizes), tuple(sizes)))
    print('Maximal relative error is {}.'.format(maximal_relative_error))
    print('{:.2f} seconds passed'.format(time.time() - start))

def collect_kl_statistics(interaction_values_std, model_sizes, models_per_size,
        sample_log2_sizes):

    kl_divergences = []

    start = time.time()
    
    for size in model_sizes:

        print('size', size)

        kl_divergences.append([])

        for model_index in range(models_per_size):

            kl_divergences[-1].append([])

            graph = generate_graph(size)
            edges = graph.get_edges()
            interaction_values = np.random.normal(scale=interaction_values_std,
                    size=graph.edges_count)

            inference_and_sampling = DecompInferenceAndSampling(graph)
            inference_and_sampling.prepare(sampling=True)

            logpf = inference_and_sampling.compute_logpf(interaction_values)

            previous_sample_size = 0

            configuration_counts = {}
            
            for sample_size in 2**sample_log2_sizes:

                configurations = \
                        inference_and_sampling.sample_spin_configurations(sample_size - \
                        previous_sample_size, interaction_values)

                for configuration in configurations:

                    configuration = tuple(configuration)

                    if configuration not in configuration_counts:
                        configuration_counts[configuration] = 0

                    configuration_counts[configuration] += 1

                kl_divergence = 0.0

                for configuration, count in configuration_counts.items():

                    spins = np.array(configuration)

                    minus_energy = (interaction_values*spins[edges[:, 0]]*\
                            spins[edges[:, 1]]).sum()

                    true_logprob = minus_energy - logpf

                    empirical_prob = count/sample_size

                    kl_divergence += empirical_prob*(np.log(empirical_prob) - true_logprob)

                kl_divergences[-1][-1].append(kl_divergence)

                previous_sample_size = sample_size

            print('\tdone with model {0}, {1:.2f} min.'.format(model_index + 1,
                    (time.time() - start)/60))

    return np.array(kl_divergences)

def draw_kl_statistics(model_sizes, sample_log2_sizes, kl_statistics):

    figure = plt.figure(figsize=(8, 5), dpi=100)

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for size_kl_statistics, size, color in zip(kl_statistics, model_sizes, colors):

        on_first_plot = True

        for model_kl_statistics in size_kl_statistics:

            plot_kwargs = {
                'zorder': 0
            }

            if on_first_plot:
                plot_kwargs['label'] = 'N={}'.format(size)
            
            plt.plot(sample_log2_sizes, model_kl_statistics, color + '--',
                    **plot_kwargs)
            plt.scatter(sample_log2_sizes, model_kl_statistics, color='k', s=10,
                    zorder=1)

            on_first_plot = False

    plt.xlabel('$\log_2 M$')
    plt.ylabel('KL-divergence')
    plt.legend()

    figure.savefig('decomp_kl.pdf', bbox_inches='tight')

def measure_execution_times(interaction_values_std, model_log2_sizes, models_per_size):

    execution_times = []

    main_start = time.time()
    
    for size in 2**model_log2_sizes:

        execution_times.append([])

        for model_index in range(models_per_size):

            graph = generate_graph(size)
            interaction_values = np.random.normal(scale=interaction_values_std,
                    size=graph.edges_count)

            inference_and_sampling = DecompInferenceAndSampling(graph)

            start = time.time()
            inference_and_sampling.prepare()
            inference_and_sampling.compute_logpf(interaction_values)
            inference_time = time.time() - start

            start = time.time()
            inference_and_sampling.prepare(sampling=True)
            inference_and_sampling.sample_spin_configurations(1, interaction_values)
            sampling_time = time.time() - start

            execution_times[-1].append([inference_time, sampling_time])

        print('done with size {0}, {1:.2f} min.'.format(size, (time.time() - main_start)/60))

    return np.array(execution_times)

def draw_execution_times(model_log2_sizes, models_per_size, execution_times):

    figure = plt.figure(figsize=(8, 5), dpi=100)

    inference_times = execution_times[:, :, 0].ravel()
    sampling_times = execution_times[:, :, 1].ravel()

    model_log2_sizes_per_point = np.repeat(model_log2_sizes, models_per_size)

    plt.scatter(model_log2_sizes_per_point, np.log2(inference_times), color='r', s=10,
            label='inference')
    plt.scatter(model_log2_sizes_per_point, np.log2(sampling_times), color='b', s=10,
            label='sampling')

    theoretical_complexity_points = 3*model_log2_sizes/2 - 7

    plt.plot(model_log2_sizes, theoretical_complexity_points, c='k',
            label='$O(N^{1.5})$')

    plt.xlabel('$\log_2 N$')
    plt.ylabel('$\log_2$(sec.)')
    plt.legend()

    figure.savefig('decomp_time.pdf', bbox_inches='tight')


if __name__ == '__main__':

    interaction_values_std = 0.1

    simulate_and_test_logpf_computation(interaction_values_std)
    print()

    model_sizes = np.array([10, 25, 40])
    models_per_size = 10
    sample_log2_sizes = np.arange(1, 12)

    kl_statistics = collect_kl_statistics(interaction_values_std, model_sizes,
            models_per_size, sample_log2_sizes)
    print()
    draw_kl_statistics(model_sizes, sample_log2_sizes, kl_statistics)

    model_log2_sizes = np.arange(3, 12)
    models_per_size = 10

    execution_times = measure_execution_times(interaction_values_std, model_log2_sizes,
            models_per_size)
    draw_execution_times(model_log2_sizes, models_per_size, execution_times)

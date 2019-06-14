import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time

from planar_ising import PlanarGraphGenerator, DecompGraph, \
        DecompInferenceAndSampling, SmallInferenceAndSampling

np.random.seed(42)
matplotlib.rcParams.update({'font.size': 15})


def make_k5():

    return np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2],
                     [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])

def generate_k33free_graph(size):

    graph = DecompGraph()
    graph.add_component(True, make_k5())
    graph_size = 5

    while graph_size != size:

        parent_index = np.random.choice(graph.nodes_count)
        parent_node = graph.nodes[parent_index]
        
        if graph.is_small_node[parent_index]:
            parent_edges_count = parent_node.shape[0]
        else:
            parent_edges_count = parent_node.edges_count

        parent_virtual_edge_index = np.random.choice(parent_edges_count)

        if graph.is_small_node[parent_index]:
            parent_connection_vertices = parent_node[parent_virtual_edge_index]
        else:
            parent_connection_vertices = \
                    np.array([parent_node.edges.vertex1[parent_virtual_edge_index],
                              parent_node.edges.vertex2[parent_virtual_edge_index]])

        if graph_size + 3 <= size and np.random.rand() > 0.5:

            node = make_k5()
            edges_count = 10

            graph.add_component(True, node)
            graph_size += 3

        else:

            new_vertices_count = 1 + np.random.choice(size - graph_size)

            node = PlanarGraphGenerator.generate_random_graph(new_vertices_count + 2, 1.0)
            edges_count = node.edges_count

            graph.add_component(False, node)
            graph_size += new_vertices_count
   
        node_index = graph.nodes_count - 1

        virtual_edge_index = np.random.choice(edges_count)

        if graph.is_small_node[node_index]:
            connection_vertices = node[virtual_edge_index]
        else:
            connection_vertices = np.array([node.edges.vertex1[virtual_edge_index],
                                            node.edges.vertex2[virtual_edge_index]])
            
        graph.add_connection(parent_index, node_index, parent_connection_vertices,
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

            graph = generate_k33free_graph(size)
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

            if relative_error > maximal_relative_error:
                maximal_relative_error = relative_error

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

            graph = generate_k33free_graph(size)
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

    figure.savefig('k33free_kl.pdf', bbox_inches='tight')

def measure_execution_times(interaction_values_std, model_log2_sizes, models_per_size):

    execution_times = []

    main_start = time.time()
    
    for size in 2**model_log2_sizes:

        execution_times.append([])

        for model_index in range(models_per_size):

            graph = generate_k33free_graph(size)
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

    figure.savefig('k33free_time.pdf', bbox_inches='tight')

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

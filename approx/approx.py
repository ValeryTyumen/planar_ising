import numpy as np
from scipy import optimize, special
import multiprocessing

from planar_ising import PlanarGraphConstructor, DecompGraph, InferenceAndSampling, \
        DecompInferenceAndSampling, SmallInferenceAndSampling, PlanarIsingModel


def compute_exact_grid_logpf(magnetic_fields, horizontal_interactions,
        vertical_interactions):

    height, width = magnetic_fields.shape

    log_weights = np.zeros((height*width, 1 << (width + 1)))

    logpf = None
    log_probs = None

    vertex_marginals = np.zeros_like(magnetic_fields)
    horizontal_marginals = np.zeros_like(horizontal_interactions)
    vertical_marginals = np.zeros_like(vertical_interactions)

    unary_signs = np.tile([-1, 1], 1 << width)
    binary_horizontal_signs = np.tile([1, -1, -1, 1], 1 << (width - 1))
    binary_vertical_signs = np.tile([1, -1], 1 << (width - 1))
    binary_vertical_signs = np.concatenate((binary_vertical_signs, -binary_vertical_signs))

    for is_forward_pass in [True, False]:
 
        if is_forward_pass:
            indices = range(height*width)
        else:

            indices = range(height*width - 2, -1, -1)
            logpf = special.logsumexp(log_weights[-1, :2])

            log_probs = log_weights[-1] - logpf - np.log(2)*width
            vertex_marginals[-1] = np.exp(special.logsumexp(log_probs[1::2]))*2 - 1

        for index in indices:

            vertex_y = index//width
            vertex_x = index%width

            if is_forward_pass:

                log_weights[index] = magnetic_fields[vertex_y, vertex_x]*unary_signs

                if vertex_x < width - 1:
                    log_weights[index] += horizontal_interactions[vertex_y, vertex_x]*\
                            binary_horizontal_signs

                if vertex_y < height - 1:
                    log_weights[index] += vertical_interactions[vertex_y, vertex_x]*\
                            binary_vertical_signs

                if index > 0:
                    log_weights[index] += np.tile(special.logsumexp(
                            log_weights[index - 1].reshape(1 << width, 2), axis=1), 2)

            else:

                cond_logprobs = log_weights[index]
                cond_logprobs -= np.repeat(special.logsumexp(cond_logprobs.reshape(1 << width, 2),
                        axis=1), 2)

                log_probs = cond_logprobs + np.repeat(special.logsumexp(
                        log_probs.reshape(2, 1 << width), axis=0), 2)

                vertex_marginals[vertex_y, vertex_x] = np.exp(special.logsumexp(
                        log_probs[unary_signs > 0]))*2 - 1

                if vertex_x < width - 1:
                    horizontal_marginals[vertex_y, vertex_x] = np.exp(special.logsumexp(
                            log_probs[binary_horizontal_signs > 0]))*2 - 1

                if vertex_y < height - 1:
                    vertical_marginals[vertex_y, vertex_x] = np.exp(special.logsumexp(
                            log_probs[binary_vertical_signs > 0]))*2 - 1

    return logpf, np.concatenate((vertex_marginals.ravel(), horizontal_marginals.ravel(),
            vertical_marginals.ravel()))

def bound_logpf_trw(magnetic_fields, horizontal_interactions, vertical_interactions):

    height, width = magnetic_fields.shape

    #TODO: remove
    vertices = np.arange(height*width).reshape(height, width)
    vertices1 = np.concatenate((vertices[:, :-1].ravel(), vertices[:-1, :].ravel()))
    vertices2 = np.concatenate((vertices[:, 1:].ravel(), vertices[1:, :].ravel()))

    iterations_count = 5

    mu = find_maximal_spanning_tree(np.zeros(height*(width - 1) + (height - 1)*width), height,
            width)

    logpf_bound = None
    result_x = None

    for iter_index in range(iterations_count):

        if logpf_bound is None:
            logpf_bound, result_x = bound_logpf_trw_fixed_mu(mu, magnetic_fields,
                    horizontal_interactions, vertical_interactions, None)

        grad = get_unary_x_binary_x_and_mutual_information(result_x, vertices1, vertices2)[2]
        update_mu = find_maximal_spanning_tree(grad, height, width)

        alpha = 0.99

        new_mu = update_mu*alpha + mu*(1 - alpha)
        new_logpf_bound, result_x = bound_logpf_trw_fixed_mu(new_mu, magnetic_fields,
                horizontal_interactions, vertical_interactions, result_x)

        step = 0.5
        c1 = 1e-4

        armijo_iterations_count = 5
        armijo_iter_index = 0

        while logpf_bound - new_logpf_bound <= c1*alpha*grad.dot(new_mu - mu) and \
                armijo_iter_index < armijo_iterations_count:

            alpha *= step
            new_mu = update_mu*alpha + mu*(1 - alpha)
            new_logpf_bound, result_x = bound_logpf_trw_fixed_mu(new_mu, magnetic_fields,
                    horizontal_interactions, vertical_interactions, result_x)

            armijo_iter_index += 1

        mu = new_mu
        logpf_bound = new_logpf_bound

    vertex_marginal_probs = result_x[:height*width]
    edge_marginal_probs = result_x[height*width:]*2 + 1 - vertex_marginal_probs[vertices1] - \
            vertex_marginal_probs[vertices2]

    marginals = np.concatenate((vertex_marginal_probs, edge_marginal_probs))*2 - 1

    return logpf_bound, marginals

def bound_logpf_trw_fixed_mu(mu, magnetic_fields, horizontal_interactions, vertical_interactions,
        start_x):

    height, width = magnetic_fields.shape
    vertices_count = height*width
    edges_count = height*(width - 1) + (height - 1)*width

    vertices = np.arange(height*width).reshape(height, width)
    vertices1 = np.concatenate((vertices[:, :-1].ravel(), vertices[:-1, :].ravel()))
    vertices2 = np.concatenate((vertices[:, 1:].ravel(), vertices[1:, :].ravel()))

    magnetic_fields = magnetic_fields.ravel()
    interactions = np.concatenate((horizontal_interactions.ravel(), vertical_interactions.ravel()))

    if start_x is None:

        vertex_x = np.random.rand(vertices_count)

        edge_x_lower_bound = np.maximum(vertex_x[vertices1] + vertex_x[vertices2] - 1, 0)
        edge_x_upper_bound = np.minimum(vertex_x[vertices1], vertex_x[vertices2])

        edge_x = edge_x_lower_bound + np.random.rand(edges_count)*(edge_x_upper_bound - \
                edge_x_lower_bound)

        start_x = np.concatenate((vertex_x, edge_x))

    bounds = [(0, 1)]*vertices_count + [(0, None)]*edges_count

    constraints = []

    for edge_index in range(edges_count):

        for is_vertex1 in [True, False]:

            constraints.append({'type': 'ineq', 'fun': trw_ineq_constraint_func1,
                    'jac': trw_ineq_constraint_jac1, 'args': (edge_index, is_vertex1, vertices1,
                    vertices2)})

        constraints.append({'type': 'ineq', 'fun': trw_ineq_constraint_func2,
                'jac': trw_ineq_constraint_jac2, 'args': (edge_index, vertices1, vertices2)})

    result = optimize.minimize(trw_func, start_x, args=(magnetic_fields, interactions, vertices1,
            vertices2, mu), method='SLSQP', constraints=constraints, bounds=bounds,
            options={'maxiter': 10000, 'ftol': 0.01})

    return -result.fun, result.x

def trw_ineq_constraint_func1(x, *args):

    return trw_ineq_constraint_jac1(x, *args).dot(x)

def trw_ineq_constraint_jac1(x, edge_index, is_vertex1, vertices1, vertices2):

    vertices_count = max(vertices1.max(), vertices2.max()) + 1

    jac = np.zeros_like(x)
    jac[vertices_count + edge_index] = -1

    if is_vertex1:
        jac[vertices1[edge_index]] = 1
    else:
        jac[vertices2[edge_index]] = 1

    return jac

def trw_ineq_constraint_func2(x, *args):

    return trw_ineq_constraint_jac2(x, *args).dot(x) + 1

def trw_ineq_constraint_jac2(x, edge_index, vertices1, vertices2):

    vertices_count = max(vertices1.max(), vertices2.max()) + 1

    jac = np.zeros_like(x)
    jac[vertices_count + edge_index] = 1
    jac[vertices1[edge_index]] = -1
    jac[vertices2[edge_index]] = -1

    return jac

def trw_func(x, magnetic_fields, interactions, vertices1, vertices2, mu):

    unary_x, binary_x, mutual_information = get_unary_x_binary_x_and_mutual_information(x,
            vertices1, vertices2)

    unary_params = np.array([[1, -1]])*magnetic_fields[:, None]
    binary_params = np.array([[[1, -1], [-1, 1]]])*interactions[:, None, None]

    return (unary_x*log(unary_x)).sum() + mu.dot(mutual_information) - \
            (unary_x*unary_params).sum() - (binary_x*binary_params).sum()

def get_unary_x_binary_x_and_mutual_information(x, vertices1, vertices2):

    vertices_count = max(vertices1.max(), vertices2.max()) + 1
    edges_count = x.shape[0] - vertices_count

    unary_x = np.zeros((vertices_count, 2))
    unary_x[:, 0] = x[:vertices_count]
    unary_x[:, 1] = 1 - x[:vertices_count]

    binary_x = np.zeros((edges_count, 2, 2))
    binary_x[:, 0, 0] = x[vertices_count:]
    binary_x[:, 0, 1] = unary_x[vertices1, 0] - x[vertices_count:]
    binary_x[:, 1, 0] = unary_x[vertices2, 0] - x[vertices_count:]
    binary_x[:, 1, 1] = 1 + x[vertices_count:] - unary_x[vertices1, 0] - unary_x[vertices2, 0]

    mutual_information = (binary_x*(log(binary_x) - \
            log(binary_x.sum(axis=1, keepdims=True)) - \
            log(binary_x.sum(axis=2, keepdims=True)))).sum(axis=(1, 2))

    return unary_x, binary_x, mutual_information

def log(array):

    eps = 1e-300

    return np.log(np.maximum(array, eps))

def find_maximal_spanning_tree(weights, height, width):

    horizontal_weights = weights[:height*(width - 1)].reshape(height, width - 1)
    vertical_weights = weights[height*(width - 1):].reshape(height - 1, width)

    horizontal_edges_mask = np.zeros_like(horizontal_weights)
    vertical_edges_mask = np.zeros_like(vertical_weights)

    sorted_edge_indices = np.argsort(weights)

    sets = [set([i]) for i in range(height*width)]

    for edge_index in sorted_edge_indices[::-1]:

        is_horizontal = (edge_index < height*(width - 1))

        if is_horizontal:
 
            edge_y = edge_index//(width - 1)
            edge_x = edge_index%(width - 1)

            vertex1 = edge_y*width + edge_x
            vertex2 = vertex1 + 1

        else:

            vertex1 = edge_index - height*(width - 1)
            vertex2 = vertex1 + width

            edge_y = vertex1//width
            edge_x = vertex1%width

        if vertex2 in sets[vertex1]:
            continue

        if is_horizontal:
            horizontal_edges_mask[edge_y, edge_x] = 1
        else:
            vertical_edges_mask[edge_y, edge_x] = 1

        if len(sets[vertex1]) < len(sets[vertex2]):
            vertex1, vertex2 = vertex2, vertex1

        for vertex in sets[vertex2]:
            sets[vertex1].add(vertex)

        sets[vertex2] = sets[vertex1]

    return np.concatenate((horizontal_edges_mask.ravel(), vertical_edges_mask.ravel()))

def bound_logpf(magnetic_fields, horizontal_interactions, vertical_interactions, use_planar):

    height, width = magnetic_fields.shape

    comp_edge_mappings = []
    comp_inference = []

    for is_vertical_split in [True, False]:

        if is_vertical_split:
            current_height, current_width = height, width
        else:
            current_height, current_width = width, height

        for sep_index in range(current_width - 2):

            if use_planar:
                apex_edge_indices, horizontal_edge_indices, vertical_edge_indices, graph = \
                        make_apex_grid_planar_subgraph(current_width, current_height, sep_index)
            else:
                apex_edge_indices, horizontal_edge_indices, vertical_edge_indices, graph = \
                        make_apex_grid_decomp_subgraph(current_width, current_height, sep_index)

            if is_vertical_split:
                edge_mapping = np.concatenate([apex_edge_indices.ravel(),
                        horizontal_edge_indices.ravel(), vertical_edge_indices.ravel()])
            else:
                edge_mapping = np.concatenate([apex_edge_indices.T.ravel(),
                        vertical_edge_indices.T.ravel(), horizontal_edge_indices.T.ravel()])

            comp_edge_mappings.append(edge_mapping)

            if use_planar:
                model = PlanarIsingModel(graph, np.zeros(graph.edges_count))
                inference = InferenceAndSampling(model)
                inference.prepare()
            else:
                inference = DecompInferenceAndSampling(graph)
                inference.prepare()

            comp_inference.append(inference)

    apex_tree = PlanarGraphConstructor.construct_from_ordered_adjacencies(
            [[height*width]]*height*width + [list(range(height*width))])
    model = PlanarIsingModel(apex_tree, np.zeros(height*width))
    inference = InferenceAndSampling(model)
    inference.prepare()

    edge_non_apex_vertices = np.minimum(apex_tree.edges.vertex1, apex_tree.edges.vertex2)

    comp_inference.append(inference)

    comp_edge_mappings.append(np.concatenate((np.argsort(edge_non_apex_vertices),
            -np.ones(height*(width - 1) + (height - 1)*width, dtype=int))))

    comp_edge_mappings = np.asarray(comp_edge_mappings)

    interactions = np.concatenate((magnetic_fields.ravel(), horizontal_interactions.ravel(),
            vertical_interactions.ravel()))

    zero_field_logpf, marginals = get_logpf_upper_bound(interactions, comp_edge_mappings,
            comp_inference)

    logpf = zero_field_logpf - np.log(2)

    return logpf, marginals

def get_logpf_upper_bound(interactions, comp_edge_mappings, comp_inference):

    comps_count = comp_edge_mappings.shape[0]

    start_weights = np.ones(comps_count)
    comp_interactions = [np.zeros((m != -1).sum()) for m in comp_edge_mappings]

    result = optimize.minimize(upper_bound_func_and_jac, start_weights, args=(interactions,
            comp_edge_mappings, comp_interactions, comp_inference), method='L-BFGS-B',
            jac=True, bounds=[(0, None)]*comps_count, options={'gtol': 1})

    probs = result.x/result.x.sum()

    _, comp_marginals = get_logpfs_and_marginals(comp_inference, comp_interactions)

    marginals = np.zeros(comp_edge_mappings.shape[1])

    for prob, edge_mapping, c_marginals in zip(probs, comp_edge_mappings, comp_marginals):
        marginals[edge_mapping != -1] += prob*c_marginals[edge_mapping[edge_mapping != -1]]

    marginals /= ((comp_edge_mappings != -1)*probs[:, None]).sum(axis=0)

    return result.fun, marginals

def upper_bound_func_and_jac(weights, interactions, comp_edge_mappings, comp_interactions,
        comp_inference):

    weight_sum = weights.sum()
    normalized_weights = weights/weight_sum

    new_comp_interactions = minimize_wrt_interactions(interactions,
            comp_edge_mappings, normalized_weights, comp_interactions,
            comp_inference)

    for index in range(len(new_comp_interactions)):
        comp_interactions[index] = new_comp_interactions[index]

    logpfs, marginals = get_logpfs_and_marginals(comp_inference, comp_interactions)

    entropies = np.array(list(l - m.dot(i) for l, m, i in zip(logpfs, marginals,
            comp_interactions)))

    value = normalized_weights.dot(logpfs)
    gradient = (entropies*weight_sum - weights.dot(entropies))/(weight_sum**2)

    return value, gradient

def minimize_wrt_interactions(interactions, comp_edge_mappings, probs,
        start_comp_interactions, comp_inference):

    offsets = np.cumsum([len(x) for x in start_comp_interactions])
    offsets = np.concatenate(([0], offsets))

    start_comp_interactions = np.concatenate(start_comp_interactions)

    result = optimize.minimize(inter_func_and_jac, start_comp_interactions,
            args=(offsets, comp_edge_mappings, probs, interactions, comp_inference),
            method='L-BFGS-B', jac=True, options={'gtol': 5e-1})

    comp_interactions = [result.x[offsets[i]:offsets[i + 1]] \
                for i in range(offsets.shape[0] - 1)]
    
    comp_interactions = project(comp_interactions, comp_edge_mappings,
            probs, interactions)

    return comp_interactions

def inter_func_and_jac(comp_interactions, offsets, comp_edge_mappings, probs,
        interactions, comp_inference):

    comp_interactions = [comp_interactions[offsets[i]:offsets[i + 1]] \
            for i in range(offsets.shape[0] - 1)]
    comp_interactions = project(comp_interactions, comp_edge_mappings,
            probs, interactions)

    logpfs, marginals = get_logpfs_and_marginals(comp_inference, comp_interactions)

    gradients = project([m*p for m, p in zip(marginals, probs)], comp_edge_mappings,
            probs, np.zeros_like(interactions))
    gradients = np.concatenate(gradients)

    return probs.dot(logpfs), gradients

def get_logpfs_and_marginals(comp_inference, comp_interactions):

    logpfs = []
    marginals = []

    with multiprocessing.Pool(10) as pool:

        for logpf, c_marginals in pool.map(do_inference_job, zip(comp_inference,
                comp_interactions)):
            logpfs.append(logpf)
            marginals.append(c_marginals)

    return logpfs, marginals

def do_inference_job(params):

    inference, c_interactions = params

    if inference.__class__.__name__ == 'InferenceAndSampling':
        inference.register_new_interactions(c_interactions)
        logpf, c_marginals = inference.compute_logpf(with_marginals=True)
    else:
        logpf, c_marginals = inference.compute_logpf(c_interactions,
                with_marginals=True)

    return logpf, c_marginals

def project(comp_values, comp_edge_mappings, probs, free_coef):

    comps_count = len(comp_edge_mappings)
    edges_count = free_coef.shape[0]

    total_comp_values = np.zeros((comps_count, edges_count))

    for comp_index, (edge_mapping, values) in enumerate(zip(comp_edge_mappings,
            comp_values)):
        total_comp_values[comp_index, edge_mapping != -1] = \
                values[edge_mapping[edge_mapping != -1]]

    comp_edges_mask = (comp_edge_mappings != -1)

    projected_values = total_comp_values - (probs.dot(total_comp_values) - \
            free_coef)[None, :]*probs[:, None]/(np.linalg.norm(probs[:, None]*\
            comp_edges_mask, axis=0)[None, :]**2)

    comp_values = []

    for comp_index, edge_mapping in enumerate(comp_edge_mappings):

        values = np.zeros((edge_mapping != -1).sum())
        values[edge_mapping[edge_mapping != -1]] = \
                projected_values[comp_index, edge_mapping != -1]
        comp_values.append(values)

    return comp_values

def make_apex_grid_planar_subgraph(width, height, sep_index):

    ordered_adjacencies = []

    apex_vertex = width*height
    vertex = 0

    for y in range(height):
        for x in range(width):

            ordered_adjacencies.append([])

            is_apex_neighbor = False

            for shift_x, shift_y in [(0, -1), (-1, 0), (0, 1), (1, 0)]:

                neighbor_x = x + shift_x
                neighbor_y = y + shift_y

                if neighbor_x < 0 or neighbor_x >= width or neighbor_y < 0 or \
                        neighbor_y >= height or (shift_y == 0 and \
                        ((x == sep_index - 1 and shift_x == 1) or \
                        (x == sep_index and shift_x == -1) or \
                        (x == sep_index + 1 and shift_x == 1) or \
                        (x == sep_index + 2 and shift_x == -1))):

                    if not is_apex_neighbor:
                        ordered_adjacencies[-1].append(apex_vertex)
                        is_apex_neighbor = True

                else:
                    ordered_adjacencies[-1].append(neighbor_x + neighbor_y*width)

            vertex += 1

    apex_adjacencies = list(sep_index + np.arange(height)*width)[::-1]
    apex_adjacencies += list(sep_index + 1 + np.arange(height)*width)

    if sep_index > 0:

        apex_adjacencies += list(np.arange(height)*width)[::-1]

        if sep_index > 1:

            apex_adjacencies += list(np.arange(1, sep_index - 1))
            apex_adjacencies += list(sep_index - 1 + np.arange(height)*width)
            apex_adjacencies += list((height - 1)*width + np.arange(1, sep_index - 1))[::-1]

    if sep_index < width - 2:

        apex_adjacencies += list(sep_index + 2 + np.arange(height)*width)[::-1]

        if sep_index < width - 3:

            apex_adjacencies += list(np.arange(sep_index + 3, width - 1))
            apex_adjacencies += list(width - 1 + np.arange(height)*width)
            apex_adjacencies += list((height - 1)*width + \
                    np.arange(sep_index + 3, width - 1))[::-1]

    ordered_adjacencies.append(apex_adjacencies)

    graph = PlanarGraphConstructor.construct_from_ordered_adjacencies(ordered_adjacencies)

    apex_edge_indices = -np.ones((height, width), dtype=int)
    horizontal_edge_indices = -np.ones((height, width - 1), dtype=int)
    vertical_edge_indices = -np.ones((height - 1, width), dtype=int)

    for edge_index, (vertex1, vertex2) in enumerate(zip(graph.edges.vertex1,
            graph.edges.vertex2)):

        vertex1, vertex2 = min(vertex1, vertex2), max(vertex1, vertex2)

        x, y = vertex1%width, vertex1//width

        if vertex2 == apex_vertex:
            apex_edge_indices[y, x] = edge_index
        elif vertex2 == vertex1 + 1:
            horizontal_edge_indices[y, x] = edge_index
        else:
            vertical_edge_indices[y, x] = edge_index

    return apex_edge_indices, horizontal_edge_indices, vertical_edge_indices, graph

def make_apex_grid_decomp_subgraph(width, height, sep_index):

    graph = DecompGraph()
    comp_vertex_mappings = []
    planar_nodes_count = 0

    apex_vertex = height*width

    if sep_index > 0:

        planar_component, vertex_mapping = make_planar_component(sep_index, height, 0, width)

        graph.add_component(False, planar_component)
        comp_vertex_mappings.append(vertex_mapping)

        planar_nodes_count = 1

    if sep_index < width - 3:

        planar_component, vertex_mapping = make_planar_component(width - sep_index - 3, height,
                sep_index + 3, width)

        graph.add_component(False, planar_component)
        comp_vertex_mappings.append(vertex_mapping)

        planar_nodes_count += 1

    small_edges1 = np.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 4], [1, 4], [2, 4], [3, 4]])

    small_edges2 = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [0, 3], [1, 4], [2, 5],
            [3, 6], [4, 7], [5, 8], [0, 9], [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9],
            [8, 9]])

    vertices = np.arange(height*width).reshape(height, width)

    for offset in range(0, height, 3):

        graph.add_component(True, small_edges2)

        if offset != 0:

            graph.add_connection(graph.nodes_count - 2, graph.nodes_count - 1, np.array([2, 3, 4]),
                    np.array([0, 1, 9]))

        vertex_mapping = vertices[offset:offset + 3, sep_index:sep_index + 3].ravel()
        vertex_mapping = np.concatenate((vertex_mapping, [apex_vertex]))

        comp_vertex_mappings.append(vertex_mapping)

        if offset + 3 != height:

            graph.add_component(True, small_edges1)

            graph.add_connection(graph.nodes_count - 2, graph.nodes_count - 1, np.array([6, 7, 9]),
                        np.array([0, 1, 4]))

            vertex_mapping = vertices[offset + 2:offset + 4, sep_index:sep_index + 2].ravel()
            vertex_mapping = np.concatenate((vertex_mapping, [apex_vertex]))

            comp_vertex_mappings.append(vertex_mapping)

    middle_node_index = planar_nodes_count + ((height//3)//2)*2
    middle_offset = ((height//3)//2)*3 + 1

    if sep_index > 0:

        graph.add_component(True, small_edges1)

        vertex_mapping = vertices[middle_offset:middle_offset + 2, sep_index - 1:sep_index + 1]
        vertex_mapping = np.concatenate((vertex_mapping.ravel(), [apex_vertex]))
        comp_vertex_mappings.append(vertex_mapping)

        graph.add_connection(0, graph.nodes_count - 1, np.array([sep_index*height,
                sep_index*(middle_offset + 1) - 1, sep_index*(middle_offset + 2) - 1]),
                np.array([4, 0, 2]))

        graph.add_connection(middle_node_index, graph.nodes_count - 1, np.array([9, 3, 6]),
                np.array([4, 1, 3]))

    if sep_index < width - 3:

        graph.add_component(True, small_edges1)

        vertex_mapping = vertices[middle_offset:middle_offset + 2, sep_index + 2:sep_index + 4]
        vertex_mapping = np.concatenate((vertex_mapping.ravel(), [apex_vertex]))
        comp_vertex_mappings.append(vertex_mapping)

        current_width = width - 3 - sep_index

        graph.add_connection(planar_nodes_count - 1, graph.nodes_count - 1,
                np.array([current_width*height, current_width*middle_offset,
                current_width*(middle_offset + 1)]), np.array([4, 1, 3]))

        graph.add_connection(middle_node_index, graph.nodes_count - 1, np.array([9, 5, 8]),
                np.array([4, 0, 2]))

    graph.enumerate()

    apex_edge_indices = -np.ones((height, width), dtype=int)
    horizontal_edge_indices = -np.ones((height, width - 1), dtype=int)
    vertical_edge_indices = -np.ones((height - 1, width), dtype=int)

    for node, is_small_node, graph_edge_indices, vertex_mapping in zip(graph.nodes,
            graph.is_small_node, graph.graph_edge_indices, comp_vertex_mappings):

        if is_small_node:
            edges = node
        else:
            edges = np.concatenate((node.edges.vertex1[:, None], node.edges.vertex2[:, None]),
                    axis=1)

        for (vertex1, vertex2), graph_edge_index in zip(edges, graph_edge_indices):

            if graph_edge_index == -1:
                continue

            grid_vertex1 = vertex_mapping[vertex1]
            grid_vertex2 = vertex_mapping[vertex2]

            if grid_vertex1 > grid_vertex2:
                grid_vertex1, grid_vertex2 = grid_vertex2, grid_vertex1

            grid_vertex1_y = grid_vertex1//width
            grid_vertex1_x = grid_vertex1%width

            if grid_vertex2 == apex_vertex:
                apex_edge_indices[grid_vertex1_y, grid_vertex1_x] = graph_edge_index
            elif grid_vertex1 + 1 == grid_vertex2:
                horizontal_edge_indices[grid_vertex1_y, grid_vertex1_x] = graph_edge_index
            else:
                vertical_edge_indices[grid_vertex1_y, grid_vertex1_x] = graph_edge_index

    return apex_edge_indices, horizontal_edge_indices, vertical_edge_indices, graph

def make_planar_component(width, height, offset, global_width):

    ordered_adjacencies = []

    apex_vertex = height*width

    for y in range(height):
        for x in range(width):

            ordered_adjacencies.append([])

            is_apex_neighbor = False

            for shift_x, shift_y in [(0, -1), (-1, 0), (0, 1), (1, 0)]:

                neighbor_x = x + shift_x
                neighbor_y = y + shift_y

                if neighbor_x < 0 or neighbor_x >= width or neighbor_y < 0 or neighbor_y >= height:

                    if not is_apex_neighbor:
                        ordered_adjacencies[-1].append(apex_vertex)
                        is_apex_neighbor = True

                else:
                    ordered_adjacencies[-1].append(neighbor_x + neighbor_y*width)

    ordered_adjacencies.append(list(np.arange(height)*width)[::-1])

    if width > 1:

        ordered_adjacencies[-1] += list(np.arange(1, width - 1))
        ordered_adjacencies[-1] += list(width - 1 + np.arange(height)*width)
        ordered_adjacencies[-1] += list((height - 1)*width + np.arange(1, width - 1))[::-1]

    vertex_mapping = np.arange(height*global_width).reshape(height, global_width)[:,
            offset:offset + width]
    vertex_mapping = np.concatenate((vertex_mapping.ravel(), [height*global_width]))

    return PlanarGraphConstructor.construct_from_ordered_adjacencies(ordered_adjacencies), \
            vertex_mapping

if __name__ == '__main__':

    np.random.seed(45)

    width = 6
    height = 6

    magnetic_fields = np.random.rand(height, width)*0.1 - 0.05
    horizontal_interactions = np.random.rand(height, width - 1)*4 - 2
    vertical_interactions = np.random.rand(height - 1, width)*4 - 2

    print('Exact logpf:', compute_exact_grid_logpf(magnetic_fields, horizontal_interactions,
            vertical_interactions)[0])

    #print('Planar bound:', bound_logpf(magnetic_fields, horizontal_interactions,
    #        vertical_interactions, True)[0])
 
    #print('Decomp bound:', bound_logpf(magnetic_fields, horizontal_interactions,
    #        vertical_interactions, False)[0])

    print('TRW bound:', bound_logpf_trw(magnetic_fields, horizontal_interactions,
            vertical_interactions)[0])

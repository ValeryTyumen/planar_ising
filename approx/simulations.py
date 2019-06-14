import numpy as np
import multiprocessing
import time
from approx import compute_exact_grid_logpf, bound_logpf, bound_logpf_trw


def generate_model(height, width, min_field, max_field, min_interaction, max_interaction,
        seed):

    np.random.seed(seed)

    magnetic_fields = min_field + np.random.rand(height, width)*(max_field - min_field)
    horizontal_interactions = min_interaction + np.random.rand(height, width - 1)*\
            (max_interaction - min_interaction)
    vertical_interactions = min_interaction + np.random.rand(height - 1, width)*\
            (max_interaction - min_interaction)

    return magnetic_fields, horizontal_interactions, vertical_interactions

def do_exact(args):

    model = generate_model(*args)

    start = time.time()

    result = compute_exact_grid_logpf(*model)
 
    print('\tDone with exact', *args)

    return result[0], result[1], (time.time() - start)/60

def do_planar(args):

    magnetic_fields, horizontal_interactions, vertical_interactions = generate_model(*args)

    start = time.time()

    result = bound_logpf(magnetic_fields, horizontal_interactions, vertical_interactions, True)
 
    print('\tDone with planar', *args)

    return result[0], result[1], (time.time() - start)/60

def do_decomp(args):

    magnetic_fields, horizontal_interactions, vertical_interactions = generate_model(*args)

    start = time.time()

    result = bound_logpf(magnetic_fields, horizontal_interactions, vertical_interactions, False)
 
    print('\tDone with decomp', *args)

    return result[0], result[1], (time.time() - start)/60

def do_trw(args):

    model = generate_model(*args)

    start = time.time()

    result = bound_logpf_trw(*model)

    print('\tDone with TRW', *args)

    return result[0], result[1], (time.time() - start)/60

if __name__ == '__main__':

    height = 15
    width = 15
    sample_size = 100

    alphas = np.arange(1.0, 3.1, 0.2)

    min_fields = np.repeat([-0.5]*len(alphas), sample_size)
    max_fields = -min_fields

    min_interactions = np.repeat(-alphas, sample_size)
    max_interactions = -min_interactions

    heights = [height]*len(min_fields)
    widths = [width]*len(min_fields)
    seeds = np.arange(len(min_fields))

    params = list(zip(heights, widths, min_fields, max_fields, min_interactions,
            max_interactions, seeds))

    pool_size = 15

    print('Starting with exact')

    exact_logpfs = []
    exact_marginals = []
    exact_time = []

    for index in range(0, len(params), pool_size):

        batch_start = index
        batch_end = min(index + pool_size, len(params))

        batch_params = params[batch_start:batch_end]

        with multiprocessing.Pool(pool_size) as pool:
            batch_results = list(pool.map(do_exact, batch_params))

        exact_logpfs += [x[0] for x in batch_results]
        exact_marginals += [x[1] for x in batch_results]
        exact_time += [x[2] for x in batch_results]

        np.savez('exact', params=np.array(params[:batch_end]), logpfs=np.array(exact_logpfs),
                marginals=np.array(exact_marginals), time=np.array(exact_time))

        print(batch_end, 'models processed')

    print('Starting with planar')

    planar_logpfs = []
    planar_marginals = []
    planar_time = []

    for index in range(0, len(params), pool_size):

        batch_start = index
        batch_end = min(index + pool_size, len(params))

        batch_params = params[batch_start:batch_end]

        batch_results = list(map(do_planar, batch_params))

        planar_logpfs += [x[0] for x in batch_results]
        planar_marginals += [x[1] for x in batch_results]
        planar_time += [x[2] for x in batch_results]

        np.savez('planar', params=np.array(params[:batch_end]),
                logpfs=np.array(planar_logpfs), marginals=np.array(planar_marginals),
                time=np.array(planar_time))

        print(batch_end, 'models processed')

    print('Starting with decomp')

    decomp_logpfs = []
    decomp_marginals = []
    decomp_time = []

    for index in range(0, len(params), pool_size):

        batch_start = index
        batch_end = min(index + pool_size, len(params))

        batch_params = params[batch_start:batch_end]

        batch_results = list(map(do_decomp, batch_params))

        decomp_logpfs += [x[0] for x in batch_results]
        decomp_marginals += [x[1] for x in batch_results]
        decomp_time += [x[2] for x in batch_results]

        np.savez('decomp', params=np.array(params[:batch_end]),
                logpfs=np.array(decomp_logpfs), marginals=np.array(decomp_marginals),
                time=np.array(decomp_time))

        print(batch_end, 'models processed')

    print('Starting with trw')

    trw_logpfs = []
    trw_marginals = []
    trw_time = []

    for index in range(0, len(params), pool_size):

        batch_start = index
        batch_end = min(index + pool_size, len(params))

        batch_params = params[batch_start:batch_end]

        with multiprocessing.Pool(pool_size) as pool:
            batch_results = list(pool.map(do_trw, batch_params))

        trw_logpfs += [x[0] for x in batch_results]
        trw_marginals += [x[1] for x in batch_results]
        trw_time += [x[2] for x in batch_results]

        np.savez('trw', params=np.array(params[:batch_end]),
                logpfs=np.array(trw_logpfs), marginals=np.array(trw_marginals),
                time=np.array(trw_time))

        print(batch_end, 'models processed')

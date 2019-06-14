import numpy as np
from scipy import special


class SmallInferenceAndSampling:

    def __init__(self, edges, constraint_vertices):

        self._edges = edges
        self._constraint_vertices = constraint_vertices

    def _get_constrained_data(self, interaction_values, constraint_spins):

        size = self._edges.max() + 1
        edges = self._edges

        spins = ((np.arange(1 << size)[:, None] >> np.arange(size)[None, :]) & 1)*2 - 1
        second_order_spins = spins[:, edges[:, 0]]*spins[:, edges[:, 1]]

        configurations_mask = (spins[:, self._constraint_vertices] == \
                constraint_spins[None, :]).all(axis=1)

        spins = spins[configurations_mask]
        second_order_spins = second_order_spins[configurations_mask]

        weights = (second_order_spins*interaction_values[None, :]).sum(axis=1)

        logpf = special.logsumexp(weights)
        probs = np.exp(weights - logpf)

        return spins, second_order_spins, probs, logpf

    def compute_logpf(self, interaction_values, constraint_spins, with_marginals=False):

        _, second_order_spins, probs, logpf = self._get_constrained_data(interaction_values,
                constraint_spins)

        if not with_marginals:
            return logpf

        return logpf, (second_order_spins*probs[:, None]).sum(axis=0)

    def sample_spin_configurations(self, sample_size, interaction_values, constraint_spins):

        spins, _, probs, _ = self._get_constrained_data(interaction_values, constraint_spins)

        random_indices = np.random.choice(probs.shape[0], p=probs, size=sample_size)

        return spins[random_indices]

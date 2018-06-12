# planar_ising

Inference and sampling of Planar Zero-Field Ising Models
Author - Valerii Likhosherstov


## Preamble

In the general case, inference/sampling of [Ising Model](https://en.wikipedia.org/wiki/Ising_model) is intractable, i.e. requires exponential time. However, when (a) its topology has a planar structure and (b) magnetic fields are set to zero, it is possible to find exact value of partition function and draw samples from model's distrubiton exactly in *O(n<sup>3/2</sup>)* time, where *n* is the number of spins.

The framework for that is implemented in this repository. Its description and proof of correctness isn't published yet, but we're working on that. The key feature of the framework is its universality and scalability, allowing to work with any planar models with thousands of spins.


## Background and idea

There are plenty of results dedicated to Planar Ising Models inference [1, 2, 3, 4] and sampling [5, 6]. We use one-to-one correspondence between spin configurations and dimer coverings on the expanded dual graph [2, 3] to switch to perfect matching sampling problem. The routine for that was elaborated by Wilson [7], which is based on recusive subdivision of the planar graphs using planar separators [8] (*nested dissection*). 

The idea to use Wilson's algorithm for spin configuration sampling was suggested in [5], but authors only apply it for the special case of square lattice topologies. Indeed, square lattices have a very simple planar separators, while a general Lipton-Tarjan is very computationally intensive. What we propose is an improved version of Wilson's algorithm, where nested dissection computation can be put aside into preprocessing time and reused over multiple samplings.


## About the code

The list of code dependencies includes NumPy, [Numba](https://numba.pydata.org/), [lipton_tarjan](https://github.com/ValeryTyumen/lipton_tarjan) and [sparse_lu](https://github.com/ValeryTyumen/sparse_lu). You will need matplotlib to draw plots in [inference_tests](https://github.com/ValeryTyumen/planar_ising/blob/master/tests/inference_tests.ipynb) and [sampling_tests](https://github.com/ValeryTyumen/planar_ising/blob/master/tests/sampling_tests.ipynb) notebooks. The code is tested under Python 3.6.4 from Anaconda distribution, NumPy 1.13.3 and Numba 0.37.0.


## Testing

See [inference_tests](https://github.com/ValeryTyumen/planar_ising/blob/master/tests/inference_tests.ipynb) and [sampling_tests](https://github.com/ValeryTyumen/planar_ising/blob/master/tests/sampling_tests.ipynb) notebooks, justifying correctness of the inference and sampling code on randomly generated Planar Ising Models of different size and illustrating execution time complexity of the framework.


## Documentation


Check docstrings and see [presentation](https://github.com/ValeryTyumen/planar_ising/blob/master/presentation.ipynb) notebook to know more about available functionality in the framework.


## Further development

- The construction exploited in the framework seems to be compatible with exact conditional inference/sampling, as well as back-propagation (marginal probabilities inference).
- The support for multiprecision arithmetic would reinforce numerical stability of LU-decompositions for large-scale models with significant diversity in interaction values. So far all computations are conducted in double-precision arithmetic.

## References

[1] - M. Kac and J. C. Ward, "A combinatorial solution of the two-dimensional ising model", Phys. Rev., vol. 88, pp. 1332–1337, Dec 1952.

[2] - P. W. Kasteleyn, "Dimer statistics and phase transitions", vol. 4, pp. 287–293, 02 1963.

[3] - M. E. Fisher, J. Math. Phys. 7, 1776 (1966).

[4] - N. N. Schraudolph and D. Kamenetsky, "Efficient exact inference in Planar Ising models", in Advances in Neural Information Processing Systems 21 (D. Koller, D. Schuurmans, Y. Bengio, and L. Bottou, eds.), pp. 1417–1424, Curran Associates, Inc., 2009.

[5] - C. K. Thomas and A. A. Middleton, "Exact algorithm for sampling the two-dimensional Ising spin glass", Phys. Rev. E, vol. 80, p. 046708, Oct 2009.

[6] - C. K. Thomas and A. A. Middleton, "Numerically exact correlations and sampling in the
two-dimensional ising spin glass", arXiv:1301.1252, 2013.

[7] -  D. B. Wilson, Proc. 8th Symp. Discrete Algorithms 258, (1997).

[8] - R. J. Lipton and R. E. Tarjan, "A separator theorem for planar graphs", tech. rep., Stanford, CA, USA, 1977.
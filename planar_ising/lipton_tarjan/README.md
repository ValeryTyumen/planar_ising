# lipton_tarjan

Python implementation of Lipton-Tarjan algorithm for planar separation


## Preamble

Lipton and Tarjan [1] prove the following theorem:

**Theorem**. Let *G* be a normal *n*-vertex planar graph having non-negative vertex costs summing to no more than one. Then the vertices of *G* can be partitioned into three sets *A, B, C* such that no edge joins a vertex in *A* with a vertex in *B*, neither *A* nor *B* has total cost exceeding *2/3*, and *C* contains no more than *2<sup>3/2</sup> n<sup>1/2</sup>* vertices.

Lipton and Tarjan also propose an algorithm to compute *A, B, C* partitioning in *O(n)* time. This repository is dedicated to implementation of this algorithm in Python.

The need in a separator of *O(n<sup>1/2</sup>)* size dividing graph into almost equal parts appears in various algorithms exploiting *divide-and-conquer* strategy.  For instance, it is used in *Generalized Nested Dissection* [2], allowing to solve systems of linear equations whose sparsity structure corresponds to a planar graph. Check the [wikipedia page](https://en.wikipedia.org/wiki/Planar_separator_theorem) for other applications and further reading.

Efficient graph separation can be leveraged for tractable Markov Random Field inference and sampling algorithms. In fact, the main reason for this implementation to appear is its usage in [Planar Zero-Field Ising Model inference and sampling project](https://github.com/ValeryTyumen/planar_ising).

## About the code

The list of code dependencies includes NumPy, [Numba](https://numba.pydata.org/). You need matplotlib to draw plots in the [tests](https://github.com/ValeryTyumen/lipton_tarjan/blob/master/tests/tests.ipynb) notebook. The code is tested under Python 3.6.4 from Anaconda distribution, NumPy 1.13.3 and Numba 0.37.0.

## Testing

See [tests](https://github.com/ValeryTyumen/lipton_tarjan/blob/master/tests/tests.ipynb) notebook, which validates correctness of the separation code on randomly generated planar graphs of different size. It also illustrates that execution time of the algorithm is linear indeed.

## Documentation

Check docstrings and see [presentation](https://github.com/ValeryTyumen/lipton_tarjan/blob/master/presentation.ipynb) notebook to know more about available functionality in the framework.

## References

[1] - R. J. Lipton and R. E. Tarjan, "A separator theorem for planar graphs", tech. rep., Stanford, CA, USA, 1977.

[2] - R. J. Lipton; D. J. Rose; R. E. Tarjan (1979), "Generalized nested dissection", _SIAM Journal on Numerical Analysis_, 16 (2): 346–358.

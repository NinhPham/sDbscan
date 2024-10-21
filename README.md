## sDbscan - A scalable density-based clustering library

sDbscan is a density-based clustering library, built on top of random projection method [CEOs](https://dl.acm.org/doi/10.1145/3447548.3467345) to fast approximate neighborhood of each point with cosine distance.
The new random projection property based on the asymptotic property of the concomitant of extreme order statistics shows that the projections of x onto closest or furthest vector to q preserves the dot product <x, q>.
sDbscan needs a significantly large random projection vectors (e.g. 1024) to speed up the process of identifying core points and their neighborhoods.
It uses the [FFHT](https://github.com/FALCONN-LIB/FFHT) to speed up the Gaussian matrix-vector multiplication, though the main computational bottleneck is distance computations.

sDbscan supports Cosine, and other distances, including L1, L2, Chi2, Jensen-Shannon distance via kernel embeddings.
sDbscan also implements sOPTICS for visualizing and selecting relevant metrics (via randomized kernel embedding) and parameter of Eps.
sDbscan supports multi-threading by adding only ```#pragma omp parallel for``` when discovering the neighborhoods.
We call [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) that supports SIMD dot product computation.

sDbscan also features [sngDbscan](https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan), a sampling-based method to speed up DBSCAN for comparison.

Empirically, sDbscan and sOptics run significantly faster, use much smaller memory, and provide similar outputs as scikit-learn.

## Prerequisites

* A compiler with C++17 support
* CMake >= 3.27 (test on Ubuntu 20.04 and Python3)
* Ninja >= 1.10 
* Eigen >= 3.3
* Pybinding11 (https://pypi.org/project/pybind11/) 

## Installation

Just clone this repository and run

```bash
python3 setup.py install
```

or 

```bash
mkdir build && cd build && cmake .. && make
```


## Test call

* Python: Dataset must be d x n matrix.
See test/test_sDbscan.py or test/plotOptics.py for Python examples.
* C++: Dataset has the size n x d. See src/main.cpp for C++ example.


## Authors

It is mainly developed by Ninh Pham. It grew out of a bachelor (honour) research project of Haochuan Xu.
If you want to cite sDbscan in a publication, please use

> [sDbscan](https://neurips.cc/virtual/2024/poster/94318)
> Haochuan Xu, Ninh Pham
> NeurIPS 2024


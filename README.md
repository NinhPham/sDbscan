## sDbscan - A scalable density-based clustering libraries

sDbscan is a DBSCAN library, built on top of random projection method [CEOs](https://dl.acm.org/doi/10.1145/3447548.3467345) to fast approximate neighborhood of each point with cosine distance.
The random projection property bases on the asymptotic property of the concomitant of extreme order statistics where the projections of $x$ onto closest or furthest vector to $q$ preserves the dot product $x^T q$.
sDbscan utilizes many random projection vectors and uses the [FFHT](https://github.com/FALCONN-LIB/FFHT) to speed up the Gaussian matrix-vector multiplication.

sDbscan supports Cosine, and L1, L2, Chi2, Jensen-Shannon distance (via kernel embeddings).
sDbscan also implements sOPTICS for visualizing and selecting relevant metrics (via randomized kernel embedding) and parameter of $\epsilon$.
sDbscan supports multi-threading by adding only ```#pragma omp parallel for``` when discovering the neighborhoods.
We call [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) that supports SIMD dot product computation and [Boost](https://www.boost.org/) with binary histogram

sDbscan also features [sngDbscan](https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan).


## Prerequisites

* A compiler with C++17 support
* CMake >= 3.27 (test on Ubuntu 20.04 and Python3)
* Ninja >= 1.10 
* Eigen >= 3.3
* Boost >= 1.71
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


Dataset must be d x n matrix.

See test/test_sDbscan.py or test/plotOptics.py for Python examples and src/main.cpp for C++ example.


## Authors

It is mainly developed by Ninh Pham. It grew out of a master research project of Haochuan Xu.
If you want to cite sDbscan in a publication, please use

> [sDbscan](https://arxiv.org/pdf/2402.15679)
> Haochuan Xu, Ninh Pham
> ArXiv 2024



# sDbscan

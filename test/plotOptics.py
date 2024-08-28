import numpy as np
import math
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from matplotlib import pyplot as plt

# Finding top-k distance of a few points
import sys
import math
import numpy as np
from matplotlib import pyplot as plt

def plot_sOptics():
    
    path = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/test/data/"
    dataset = np.loadtxt(path + 'mnist_all_X')

    dataset_t = np.transpose(dataset)
    dataset_t.dtype == np.float32

    y = np.loadtxt(path + 'mnist_all_y_70K_784')

    n, d = np.shape(dataset)

    # Param
    numProj = 1024
    k = 5
    m = 50
    numEmbed = 1024
    sigma = 2600
    dist = "Cosine"
    clusterNoise = 0
    output = 'sOptics'
    numThreads = 64
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 0.01
    seed = -1

    import sDbscan

    dbs = sDbscan.sDbscan(n, d)
    dbs.setParams(numProj, k, m, dist, numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose, numThreads, seed, output)

    start = timeit.default_timer()
    dbs.fit_sOptics(dataset_t, 0.25, 50)
    end = timeit.default_timer()
    print("sOPTICS Time: ", end - start)

    s_reachDist = np.array(dbs.reachability_)
    idx = np.where(s_reachDist < 0)
    s_reachDist[idx] = math.inf
    sOptics = np.take(s_reachDist, np.array(dbs.ordering_))
    
    plt.plot(sOptics)
    plt.show()


# main
plot_sOptics()




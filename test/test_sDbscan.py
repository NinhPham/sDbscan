import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
import math
import timeit
    
def NMI_fit_sDbscan_sngDbscan_metrics():

    """
    sDbscan and sngDbscan with 5 metrics: Cosine, Chi2, JS, L1, L2 on mnistAll data set
    Return NMI for each method
    """

    path = "/home/npha145/Dataset/Clustering/"
    dataset = np.loadtxt(path + 'mnist_all_X')
    n, d = np.shape(dataset)

    dataset_t = np.transpose(dataset)
    dataset_t.dtype == np.float32

    y = np.loadtxt(path + 'mnist_all_y_70K_784')

    # Param
    numProj = 1024
    k = 5
    m = 50
    minPts = 50
    numEmbed = 1024
    sigma = 100
    clusterNoise = 0  # set 1 for sDbscan-1NN
    output = 'sDbscan'
    numThreads = 16 # Use all threads
    verbose = False
    intervalSampling = 0.4  # default param for Chi2 & JS
    samplingRatio = 0.01 # prob for sDbscan-1NN
    seed = -1

    import sDbscan

    dbs = sDbscan.sDbscan(n, d)

    print("--------- Cosine -----------")
    dbs.set_params(numProj, k, m, "Cosine", numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio, verbose, output, numThreads, seed)
    # dbs.test_sDbscan(dataset_t, 0.1, 0.01, 6, minPts)

    baseEps = 0.1
    rangeEps = 0.01

    for i in range(6):

        new_eps = baseEps + rangeEps * i

        dbs.clear()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        print("sDBSCAN Cosine Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        print("sDBSCAN Cosine Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

        # Note: sngDbscan need to use noiseCluster = 1 to get reasonable accuracy
        # The released sngDbscan does not implement exactly the algorithm description in their paper

        # dbs.clear()
        # dbs.set_clusterNoise(2) # 2 for sngDbscan heuristic
        # dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        # print("sngDBSCAN Cosine Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        # print("sngDBSCAN Cosine Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

    print("--------- Chi2 -----------")
    dbs.set_params(numProj, k, m, "Chi2", numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio, verbose, output, numThreads, seed)
    # dbs.test_sDbscan(dataset_t, 0.1, 0.01, 6, minPts)

    baseEps = 0.1
    rangeEps = 0.01
    for i in range(6):
        new_eps = baseEps + rangeEps * i

        dbs.clear()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        print("sDBSCAN Chi2 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        print("sDBSCAN Chi2 Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

        # dbs.clear()
        # dbs.set_clusterNoise(2)  # 2 for sngDbscan heuristic
        # dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        # print("sngDBSCAN Chi2 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        # print("sngDBSCAN Chi2 Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

    print("--------- JS -----------")
    dbs.set_params(numProj, k, m, "JS", numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio, verbose, output, numThreads, seed)
    # dbs.test_sDbscan(dataset_t, 0.1, 0.01, 6, minPts)

    baseEps = 0.1
    rangeEps = 0.01
    for i in range(6):
        new_eps = baseEps + rangeEps * i

        dbs.clear()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        print("sDBSCAN JS Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        print("sDBSCAN JS Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

        # dbs.clear()
        # dbs.set_clusterNoise(2)  # 2 for sngDbscan heuristic
        # dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        # print("sngDBSCAN JS Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        # print("sngDBSCAN JS Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

    print("--------- L1 -----------")
    sigma = 16000
    dbs.set_params(numProj, k, m, "L1", numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio,
                   verbose, output, numThreads, seed)

    # dbs.test_sDbscan(dataset_t, 8000, 1000, 5, minPts)

    baseEps = 8000
    rangeEps = 1000
    for i in range(5):
        new_eps = baseEps + rangeEps * i

        dbs.clear()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        print("sDBSCAN L1 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        print("sDBSCAN L1 Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

        # dbs.clear()
        # dbs.set_clusterNoise(2)  # 2 for sngDbscan heuristic
        # dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        # print("sngDBSCAN L1 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        # print("sngDBSCAN L1 Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

    print("--------- L2 -----------")
    sigma = 2600
    dbs.set_params(numProj, k, m, "L2", numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio,
                   verbose, output, numThreads, seed)
    # dbs.test_sDbscan(dataset_t, 1200, 50, 5, minPts)

    baseEps = 1200
    rangeEps = 50
    for i in range(5):
        new_eps = baseEps + rangeEps * i

        dbs.clear()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        print("sDBSCAN L2 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        print("sDBSCAN L2 Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))

        # dbs.clear()
        # dbs.set_clusterNoise(2)  # 2 for sngDbscan heuristic
        # dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        # print("sngDBSCAN L2 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
        # print("sngDBSCAN L2 Acc: AMI %f." % adjusted_mutual_info_score(dbs.labels_, y))


NMI_fit_sDbscan_sngDbscan_metrics()

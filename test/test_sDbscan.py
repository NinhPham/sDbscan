import numpy as np
import math
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from matplotlib import pyplot as plt
import timeit

def fit_sOptics_sngOptics():

    """
    We test fit_sOptics and fit_sngOptics
    """

    path = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/test/data/"
    filename = path + 'mnist_all_X'
    y = np.loadtxt(path + 'mnist_all_y_70K_784')

    dataset = np.load(filename)
    dataset_t = np.transpose(dataset)
    dataset_t.dtype == np.float32


    n, d = np.shape(dataset)

    # Param
    numProj = 1024
    k = 5
    m = 50
    numEmbed = 1024
    sigma = 2600
    dist = "L2"
    clusterNoise = 0
    output = 'sOptics'
    numThreads = 64
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 0.01
    seed = -1

    import sDbscan

    dbs = sDbscan.sDbscan(n, d)
    dbs.set_params(numProj, k, m, dist, numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
                  numThreads, seed, output)

    start = timeit.default_timer()
    dbs.fit_sOptics(dataset_t, 3600, 50)
    end = timeit.default_timer()
    print("sOPTICS Time: ", end - start)

    s_reachDist = np.array(dbs.reachability_)
    idx = np.where(s_reachDist < 0)
    s_reachDist[idx] = math.inf
    sOptics = np.take(s_reachDist, np.array(dbs.ordering_))

    dbs.clear()
    start = timeit.default_timer()
    dbs.fit_sngOptics(dataset_t, 1800, 50)
    end = timeit.default_timer()
    print("sngOPTICS Time: ", end - start)

    sng_reachDist = np.array(dbs.reachability_)
    idx = np.where(sng_reachDist < 0)
    sng_reachDist[idx] = math.inf
    sngOptics = np.take(sng_reachDist, np.array(dbs.ordering_))

    # Plot two different figure
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(sOptics)
    ax2.plot(sngOptics)
    plt.show()


def load_fit_sOptics_sngOptics():

    """
    Test load_fit_sOptics and load_fit_sngOptics
    """

    path = "/home/npha145/Dataset/Clustering/"
    filename = path + 'mnist8m_X'
    y = np.loadtxt(path + 'mnist8m_y_8100000_784')

    n = 8100000 # 70000
    d = 784

    # Param
    numProj = 1024
    k = 10
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
    dbs.set_params(numProj, k, m, dist, numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
                  numThreads, seed, output)

    start = timeit.default_timer()
    dbs.load_fit_sOptics(dataset_t, 0.25, 50)
    end = timeit.default_timer()
    print("sOPTICS Time: ", end - start)

    s_reachDist = np.array(dbs.reachability_)
    idx = np.where(s_reachDist < 0)
    s_reachDist[idx] = math.inf
    sOptics = np.take(s_reachDist, np.array(dbs.ordering_))

    dbs.clear()
    start = timeit.default_timer()
    dbs.load_fit_sngOptics(dataset_t, 0.3, 50)
    end = timeit.default_timer()
    print("sngOPTICS Time: ", end - start)

    sng_reachDist = np.array(dbs.reachability_)
    idx = np.where(sng_reachDist < 0)
    sng_reachDist[idx] = math.inf
    sngOptics = np.take(sng_reachDist, np.array(dbs.ordering_))

    # Plot two different figure
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(sOptics)
    ax2.plot(sngOptics)
    plt.show()

def load_test_sDbscan_mnist8m():

    """
    Test load_test_sDbscan on mnist8m data set
    """

    # path = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/test/data/"
    # filename = path + 'mnist_all_X'
    # y = np.loadtxt(path + 'mnist_all_y_70K_784')

    path = "/home/npha145/Dataset/Clustering/"
    filename = path + 'mnist8m_X'
    y = np.loadtxt(path + 'mnist8m_y_8100000_784')

    n = 8100000 # 70000
    d = 784

    # Param
    numProj = 1024
    k = 10
    m = 50
    numEmbed = 1024
    sigma = 2600
    dist = "Cosine"
    clusterNoise = 1
    output = 'sDbscan'
    numThreads = 64
    verbose = True
    intervalSampling = 0.4
    samplingRatio = 0.01
    seed = 1

    import sDbscan

    dbs = sDbscan.sDbscan(n, d)

    dbs.set_params(numProj, k, m, "Cosine", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
                  numThreads, seed, output)
    eps = 0.14
    rangeEps = 0.01
    minPts = 50

    dbs.load_test_sDbscan(filename, eps, rangeEps, minPts)




def NMI_fit_sDbscan_sngDbscan_sngDbscan():

    path = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/test/data/"
    dataset = np.loadtxt(path + 'mnist_all_X')

    dataset_t = np.transpose(dataset)
    dataset_t.dtype == np.float32

    # print('Normalizing the dataset for sngDbscan')
    # dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    # dataset.dtype == np.float32

    y = np.loadtxt(path + 'mnist_all_y_70K_784')

    n, d = np.shape(dataset)

    # Param
    numProj = 1024
    k = 5
    m = 50
    numEmbed = 1024
    sigma = 2600
    dist = "JS"
    clusterNoise = 0 # 0: no, 1: Ceos, 2: heuristic
    output = 'sDbscan'
    numThreads = 64
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 0.01
    seed = -1

    import sDbscan
    from SubsampledNeighborhoodGraphDBSCAN import SubsampledNeighborhoodGraphDBSCAN

    eps = 0.1
    rangeEps = 0.01
    minPts = 50

    dbs = sDbscan.sDbscan(n, d)
    dbs.set_params(numProj, k, m, dist, numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose, numThreads, seed, output)

    for i in range(5):

        new_eps = eps + rangeEps * i

        print("-------------")
        print("Eps: %.2f" % new_eps)

        dbs.clear()
        start = timeit.default_timer()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        end = timeit.default_timer()
        print("sDBSCAN Time: ", end - start)
        print("sDBSCAN NMI %.3f." % normalized_mutual_info_score(dbs.labels_, y))


        dbs.clear()
        start = timeit.default_timer()
        dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        end = timeit.default_timer()
        print("Our sngDBSCAN Time: ", end - start)
        print("Our sngDBSCAN NMI %.3f." % normalized_mutual_info_score(dbs.labels_, y))

        # sngDbscan only support L2
        start = timeit.default_timer()
        # new_eps = math.sqrt(2 * new_eps) # only support L2 so have to convert to L2
        dbscan_ng = SubsampledNeighborhoodGraphDBSCAN(p=0.01, eps=new_eps, minPts=minPts)
        y_pred = dbscan_ng.fit_predict(dataset)
        end = timeit.default_timer()
        print("sngDBSCAN Time: ", end - start)
        print("sngDBSCAN NMI %.3f." % normalized_mutual_info_score(y_pred, y))
def NMI_fit_sDbscan_sngDbscan_metrics():

    """
    Test own implementation of sDbscan and sngDbscan with 5 metrics: Cosine, Chi2, JS, L1, L2 on mnistAll data set
    """

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
    minPts = 50
    numEmbed = 1024
    sigma = 2600
    eps = 0.1
    dist = "Cosine"
    clusterNoise = 0
    output = 'sDbscan'
    numThreads = 64
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 0.02
    seed = -1

    import sDbscan

    dbs = sDbscan.sDbscan(n, d)

    print("--------- Cosine -----------")
    dbs.set_params(numProj, k, m, "Cosine", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    # dbs.test_sDbscan(dataset_t, 0.1, 0.01, minPts)

    for i in range(5):

        new_eps = eps + 0.01 * i

        dbs.clear()
        dbs.fit_sDbscan(dataset_t, new_eps, minPts)
        print("sDBSCAN Cosine Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))

        dbs.clear()
        dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
        print("sngDBSCAN Cosine Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))


    # print("--------- Chi2 -----------")
    # dbs.set_params(numProj, k, m, "Chi2", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
    #                numThreads, seed, output)
    # # dbs.test_sDbscan(dataset_t, 0.1, 0.01, minPts)
    #
    # for i in range(5):
    #     new_eps = eps + 0.01 * i
    #
    #     dbs.clear()
    #     dbs.fit_sDbscan(dataset_t, new_eps, minPts)
    #     print("sDBSCAN Chi2 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))

    #     dbs.clear()
    #     dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
    #     print("sngDBSCAN Chi2 Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
    #
    #
    # print("--------- JS -----------")
    # dbs.set_params(numProj, k, m, "JS", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
    #                numThreads, seed, output)
    # # dbs.test_sDbscan(dataset_t, 0.1, 0.01, minPts)
    #
    # for i in range(5):
    #     new_eps = eps + 0.01 * i
    #
    #     dbs.clear()
    #     dbs.fit_sDbscan(dataset_t, new_eps, minPts)
    #     print("sDBSCAN JS Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
    #     dbs.clear()
    #     dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
    #     print("sngDBSCAN JS Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
    #
    #
    # print("--------- L1 -----------")
    # dbs.set_params(numProj, k, m, "L1", numEmbed, 12000, intervalSampling, samplingRatio, clusterNoise, verbose,
    #               numThreads, seed, output)
    # # dbs.test_sDbscan(dataset_t, 8000, 1000, minPts)
    #
    # eps = 8000
    # for i in range(5):
    #     new_eps = eps + 1000 * i
    #
    #     dbs.clear()
    #     dbs.fit_sDbscan(dataset_t, new_eps, minPts)
    #     print("sDBSCAN Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
    #     dbs.clear()
    #     dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
    #     print("sngDBSCAN Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
    #
    #
    # print("--------- L2 -----------")
    # dbs.set_params(numProj, k, m, "L2", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
    #               numThreads, seed, output)
    # # dbs.test_sDbscan(dataset_t, 1200, 50, minPts)
    #
    # eps = 1200
    # for i in range(5):
    #     new_eps = eps + 50 * i
    #
    #     dbs.clear()
    #     dbs.fit_sDbscan(dataset_t, new_eps, minPts)
    #     print("sDBSCAN Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))
    #     dbs.clear()
    #     dbs.fit_sngDbscan(dataset_t, new_eps, minPts)
    #     print("sngDBSCAN Acc: NMI %f." % normalized_mutual_info_score(dbs.labels_, y))

def test_sDbscan_metrics():

    """
    Output sDbscan into files, using 5 metrics
    """

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
    minPts = 50
    numEmbed = 1024
    sigma = 1
    eps = 0.1
    dist = "Cosine"
    clusterNoise = 0
    output = 'sDbscan'
    numThreads = 64
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 0.02
    seed = -1

    import sDbscan

    dbs = sDbscan.sDbscan(n, d)

    print("--------- Cosine -----------")
    dbs.set_params(numProj, k, m, "Cosine", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    dbs.test_sDbscan(dataset_t, 0.1, 0.01, minPts)

    print("--------- Chi2 -----------")
    dbs.set_params(numProj, k, m, "Chi2", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
                   numThreads, seed, output)
    dbs.test_sDbscan(dataset_t, 0.1, 0.01, minPts)

    print("--------- JS -----------")
    dbs.set_params(numProj, k, m, "JS", numEmbed, sigma, intervalSampling, samplingRatio, clusterNoise, verbose,
                   numThreads, seed, output)
    dbs.test_sDbscan(dataset_t, 0.1, 0.01, minPts)

    print("--------- L1 -----------")
    dbs.set_params(numProj, k, m, "L1", numEmbed, 12000, intervalSampling, samplingRatio, clusterNoise, verbose,
                  numThreads, seed, output)
    dbs.test_sDbscan(dataset_t, 8000, 1000, minPts)

    print("--------- L2 -----------")
    dbs.set_params(numProj, k, m, "L2", numEmbed, 2600, intervalSampling, samplingRatio, clusterNoise, verbose,
                  numThreads, seed, output)
    dbs.test_sDbscan(dataset_t, 1200, 50, minPts)
def NMI_sDbscan_output():

    """
    Get accuracy from output file
    """

    # y = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_y_70K_784')
    # y = np.loadtxt('/home/npha145/Dataset/Clustering/pamap2_y_no_0_1770131_51')
    y = np.loadtxt('/home/npha145/Dataset/Clustering/mnist8m_y_8100000_784')
    # path = '/home/npha145/Dropbox (Uni of Auckland)/DbscanCEOs/pamap0-Test/'
    path = '//home/npha145/Dropbox (Uni of Auckland)/Working/_Code/Python/Clustering/sDbscan//'

    s = 5
    eps_start = 0.14
    eps_gap = 0.01
    for j in range(s):

        eps = eps_start + eps_gap * j

        filename = (path + 'sDbscan_Cosine_Eps_' + str(round(eps * 1000)) +
                    '_MinPts_50_KerFeatures_784_NumProj_1024_TopM_50_TopK_10')

        y_fastDbscan = np.loadtxt(filename)
        print("sDBSCAN eps: ", round(eps * 1000), " NMI: ", normalized_mutual_info_score(y_fastDbscan, y))

def test_sngDbscan_metrics():

    """
    Output sngDbscan into files, using 5 metrics
    """

    path = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/test/data/"
    dataset = np.loadtxt(path + 'mnist_all_X')
    dataset_t = np.transpose(dataset)
    dataset_t.dtype == np.float32

    y = np.loadtxt(path + 'mnist_all_y_70K_784')

    n, d = np.shape(dataset)

    # Param
    clusterNoise = 0
    output = 'sngDbscan'
    numThreads = 64
    verbose = False
    samplingRatio = 0.02
    seed = -1

    import sDbscan

    minPts = 50
    dbs = sDbscan.sDbscan(n, d)

    print("--------- Cosine -----------")
    dbs.set_sngParams("Cosine", samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    dbs.test_sngDbscan(dataset_t, 0.1, 0.01, minPts)

    print("--------- Chi2 -----------")
    dbs.set_sngParams("Chi2", samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    dbs.test_sngDbscan(dataset_t, 0.1, 0.01, minPts)

    print("--------- JS -----------")
    dbs.set_sngParams("JS", samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    dbs.test_sngDbscan(dataset_t, 0.1, 0.01, minPts)

    print("--------- L1 -----------")
    dbs.set_sngParams("L1", samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    dbs.test_sngDbscan(dataset_t, 8000, 1000, minPts)

    print("--------- L2 -----------")
    dbs.set_sngParams("L2", samplingRatio, clusterNoise, verbose, numThreads, seed, output)
    dbs.test_sngDbscan(dataset_t, 1200, 50, minPts)
def NMI_sngDbscan_output():

    """
    Get accuracy from output file
    """

    y = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_y_70K_784')
    # y = np.loadtxt('/home/npha145/Dataset/Clustering/pamap2_y_no_0_1770131_51')
    # y = np.loadtxt('/home/npha145/Dataset/Clustering/mnist8m_y_8100000_784')
    # path = '/home/npha145/Dropbox (Uni of Auckland)/DbscanCEOs/pamap0-Test/'
    path = '//home/npha145/Dropbox (Uni of Auckland)/Working/_Code/Python/Clustering/sDbscan//'

    s = 5
    eps_start = 1200
    eps_gap = 50
    for j in range(s):

        eps = eps_start + eps_gap * j

        filename = (path + 'sngDbscan_L2_Eps_' + str(round(eps * 1000)) +
                    '_MinPts_50_Prob_20')

        y_fastDbscan = np.loadtxt(filename)
        print("sngDBSCAN eps: ", round(eps * 1000), " NMI: ", normalized_mutual_info_score(y_fastDbscan, y))


""" Call fit_sDbscan and fit_sngDbscan with supported metrics"""
# NMI_fit_sDbscan_sngDbscan_metrics() # test our own implementations sDbscan and sngDbscan

""" Call fit_sDbscan, fit_sngDbscan and sngDbscan (Github) with supported metrics"""
# NMI_fit_sDbscan_sngDbscan_sngDbscan()

""" Call test_sDbscan() with supported metrics """
# test_sDbscan_metrics() # test sDbscan with several metrics
# NMI_sDbscan_output() # get accuracy from output file

""" Call test_sngDbscan() with supported metrics """
# test_sngDbscan_metrics() # test sDbscan with several metrics
# NMI_sngDbscan_output() # get accuracy from output file


""" Call load_test_sDbscan() with supported metrics on Mnist8m """
load_test_sDbscan_mnist8m()
# NMI_sDbscan_output() # get accuracy from output file

""" Call fit_sOptics and fit_sngOptics and plot 2 graphs"""
# fit_sOptics_sngOptics()

""" Call load_fit_sOptics and load_fit_sngOptics on Mnist8m and plot 2 graphs"""
# load_fit_sOptics_sngOptics()
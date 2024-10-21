//
// Created by npha145 on 22/05/24.
//

#include "sDbscan.h"
#include "Utilities.h"

/**
 * Maintain the projection matrix MATRIX_FHT for parallel processing.
 * Store binary bits for FHWT transform, especially we have to use them for Dbscan-1NN
 * Using priority queue to extract top-k and top-m
 *
 * - For each point Xi, compute its dot product, extract top-k close/far random vectors
 * - For each random vector Ri, reuse dot product matrix, extract top-k close/far points
 *
 */
void sDbscan::rp_parIndex()
{
    /** Param for embedding L1 and L2 **/
    int iFourierEmbed_D = sDbscan::ker_n_features / 2; // This is because we need cos() and sin()

    // See: https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
    if (sDbscan::distance == "L1")
        sDbscan::matrix_R = cauchyGenerator(iFourierEmbed_D, sDbscan::n_features, 0, 1.0 / sDbscan::ker_sigma, sDbscan::seed); // K(x, y) = exp(-gamma * L1_dist(X, y))) where gamma = 1/sigma
    else if (sDbscan::distance == "L2")
        sDbscan::matrix_R = gaussGenerator(iFourierEmbed_D, sDbscan::n_features, 0, 1.0 / sDbscan::ker_sigma, sDbscan::seed); // std = 1/sigma, K(x, y) = exp(-gamma * L2_dist^2(X, y))) where gamma = 1/2 sigma^2

    /** Param for random projection via FHT **/
    MatrixXf MATRIX_FHT = MatrixXf::Zero(sDbscan::n_proj, sDbscan::n_points);

    int log2Project = log2(sDbscan::fhtDim);
    bitHD3Generator(sDbscan::fhtDim * sDbscan::n_rotate, sDbscan::seed, sDbscan::bitHD3);

    /** Param for index **/
    // the first topK is for close vectors, the second topK is for far away vectors
    sDbscan::matrix_topK = MatrixXi::Zero(2 * sDbscan::topK, sDbscan::n_points);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads); // already set it outside

    /**
    Parallel for each the point Xi: (1) Compute and store dot product, and (2) Extract top-k close/far random vectors
    **/
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        /**
        Random embedding
        TODO: create buildKernelFeatures and random projection as a new function since sDbscan-1NN also use it
        **/
        VectorXf vecX = sDbscan::matrix_X.col(n);
        VectorXf vecEmbed = VectorXf::Zero(sDbscan::ker_n_features); // sDbscan::ker_n_features >= n_features

        // NOTE: Already ensure ker_n_features = n_features on Cosine when initializing the object
        if (sDbscan::distance == "Cosine")
            vecEmbed.segment(0, sDbscan::n_features) = vecX;
        else if ((sDbscan::distance == "L1") || (sDbscan::distance == "L2"))
        {
            VectorXf vecProject = sDbscan::matrix_R * vecX;
            vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
            vecEmbed.segment(iFourierEmbed_D, iFourierEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
        }
        else if (sDbscan::distance == "Chi2")
            embedChi2(vecX, sDbscan::ker_n_features, sDbscan::n_features, sDbscan::ker_intervalSampling, vecEmbed);
        else if (sDbscan::distance == "JS")
            embedJS(vecX, sDbscan::ker_n_features, sDbscan::n_features, sDbscan::ker_intervalSampling, vecEmbed);

        /**
        Random projection
        **/

        VectorXf vecRotation = VectorXf::Zero(sDbscan::fhtDim); // n_proj > ker_n_features
        vecRotation.segment(0, sDbscan::ker_n_features) = vecEmbed;

        for (int r = 0; r < sDbscan::n_rotate; ++r)
        {
            // Component-wise multiplication with a random sign
            for (int d = 0; d < sDbscan::fhtDim; ++d)
            {
                vecRotation(d) *= (2 * (int)sDbscan::bitHD3[r * sDbscan::fhtDim + d] - 1);
            }

            // Multiple with Hadamard matrix by calling FWHT transform
            fht_float(vecRotation.data(), log2Project);
        }

        // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
        MATRIX_FHT.col(n) = vecRotation.segment(0, sDbscan::n_proj); // only get up to #n_proj

        /**
        Extract top-k closes and furtherest random vectors
        **/

        Min_PQ_Pair minCloseTopK;
        Min_PQ_Pair minFarTopK;

        for (int d = 0; d < sDbscan::n_proj; ++d)
        {
            float fValue = vecRotation(d); // take the value up to n_proj - it might be safer to use MATRIX_FHT.col(n)

            /**
            1) For each random vector Ri, get top-MinPts closest index and top-MinPts furthest index
            - Will process it later using projection matrix
            2) For each point Xi, get top-K closest random vector and top-K furthest random vector
            **/

            // (1) Close: Using priority queue to find top-k closest vectors for each point
            if ((int)minCloseTopK.size() < sDbscan::topK)
                minCloseTopK.emplace(d, fValue);
            else
            {
                if (fValue > minCloseTopK.top().m_fValue)
                {
                    minCloseTopK.pop();
                    minCloseTopK.emplace(d, fValue);
                }
            }

            // (2) Far: Using priority queue to find top-k furthest vectors
            if ((int)minFarTopK.size() < sDbscan::topK)
                minFarTopK.emplace(d, -fValue);
            else
            {
                if (-fValue > minFarTopK.top().m_fValue)
                {
                    minFarTopK.pop();
                    minFarTopK.emplace(d, -fValue);
                }
            }
        }

        // Get (sorted by projection value) top-k closest and furthest vector for each point
        for (int k = sDbscan::topK - 1; k >= 0; --k)
        {
            sDbscan::matrix_topK(k, n) = minCloseTopK.top().m_iIndex;
            minCloseTopK.pop();

            sDbscan::matrix_topK(k + sDbscan::topK, n) = minFarTopK.top().m_iIndex;
            minFarTopK.pop();
        }

    }


    /**
    For each random vector, extract top-m close/far data points
    **/
    sDbscan::matrix_topM = MatrixXi::Zero(2 * sDbscan::topM, sDbscan::n_proj); // the first topM is for close, the second topM is for far away

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads); // TODO: do we need to set it for each for call?

    /**
    Parallel for each random vector, getting 2*top-m as close and far candidates
    **/
#pragma omp parallel for
    for (int d = 0; d < sDbscan::n_proj; ++d)
    {
        // sort(begin(matProject.col(d)), end(matProject.col(d)), [](float lhs, float rhs){return rhs > lhs});

        Min_PQ_Pair minPQ_Close;
        Min_PQ_Pair minPQ_Far;

        VectorXf vecProject = MATRIX_FHT.row(d); // it must be row since D x N

        for (int n = 0; n < sDbscan::n_points; ++n)
        {
            float fValue = vecProject(n);

            // Close
            if ((int)minPQ_Close.size() < sDbscan::topM)
                minPQ_Close.emplace(n, fValue);
            else
            {
                if (fValue > minPQ_Close.top().m_fValue)
                {
                    minPQ_Close.pop();
                    minPQ_Close.emplace(n, fValue);
                }
            }

            // Far
            if ((int)minPQ_Far.size() < sDbscan::topM)
                minPQ_Far.emplace(n, -fValue);
            else
            {
                if (-fValue > minPQ_Far.top().m_fValue)
                {
                    minPQ_Far.pop();
                    minPQ_Far.emplace(n, -fValue);
                }
            }
        }

        for (int m = sDbscan::topM - 1; m >= 0; --m)
        {
            // Close
            sDbscan::matrix_topM(m, d) = minPQ_Close.top().m_iIndex;
            minPQ_Close.pop();

            // Far
            sDbscan::matrix_topM(m + sDbscan::topM, d) = minPQ_Far.top().m_iIndex;
            minPQ_Far.pop();
        }
    }
}

/**
 * Finding core points using the lightweight RP index
 *
 * @param eps
 * @param minPts
 */
void sDbscan::rp_findCorePoints(float eps, int minPts)
{
    // Space and time overheads of unordered_set<int> are significant compared to vector since we only has approximate 2km neighbors
    // In case we add x to q's neighbor, and q to x's neighbor, # neighbors > 2km
    sDbscan::vec2D_Neighbors = vector<IVector> (sDbscan::n_points, IVector());

    // bitset work since # core points tend to be relatively large compared to n_points
    sDbscan::bit_CorePoints = boost::dynamic_bitset<>(sDbscan::n_points);

//    chrono::steady_clock::time_point begin;
//    begin = chrono::steady_clock::now();

    // Initialize the OPENMP lock
//    omp_lock_t ompLock;
//    omp_init_lock(&ompLock);

    //TODO: If single thread, then we can improve if we store (X1, X2) s.t. <X1,X2> >= threshold

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads); // already set it outside
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        // Get top-k closese/furthest vectors
        VectorXf vecXn = sDbscan::matrix_X.col(n);
        VectorXi vecTopK = sDbscan::matrix_topK.col(n); // size 2K: first K is close, last K is far

        /**
        Choice of data structure: We are using bitSet of size N bits to avoid duplicate distance computation
        If using vector: 2k * m * 32 (32 bits / element) << N (bits)
        If using unorder_set or set: 2k * m * 36 * 8 (36 bytes / element) << N (bits)
        N = 1M, k = 10, m = 100, then bitSet seems to be okie regarding space complexity
        For Optics, bitset will be more useful if using larger k and m
        **/
//        IVector vecNeighborhood;
//        unordered_set<int> approxNeighbor;
//        set<int> approxNeighbor;
        boost::dynamic_bitset<> approxNeighbor(sDbscan::n_points); // if n = 8M, then 8M bits ~ 1MB of RAM. With 64 threads, we use up to 64 MB of RAM

        for (int k = 0; k < sDbscan::topK; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            for (int i = 0; i < sDbscan::topM; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = sDbscan::matrix_topM(i, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
//                if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = true;

                    float fDist = computeDist(vecXn, sDbscan::matrix_X.col(iPointIdx), sDbscan::distance);

		     // We need to store both (x1,x2) and (x2, x1) to increase the neighborhood. Otherwise, some core points are missed classified as border points.
		     // Hence, a big cluster will be broken into several small clusters if missing some connected core points
                    if (fDist <= eps)
                    {
#pragma omp critical
                        {
//                        set2D_Dbscan_Neighbor[n].insert(iPointIdx);  // set is very slow and take much memory
//                        set2D_Dbscan_Neighbor[iPointIdx].insert(n);
//                            omp_set_lock(&ompLock);
                            vec2D_Neighbors[n].push_back(iPointIdx); // allow duplicate, at most double so vector is much faster than set()
                            vec2D_Neighbors[iPointIdx].push_back(n); // e.g. 1 = {3, 5}, and 3 = {1 6}
//                            omp_unset_lock(&ompLock);
                        }
                    }

                }
            }

            // Far
            Ri = vecTopK(k + sDbscan::topK);

            for (int i = 0; i < sDbscan::topM; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = sDbscan::matrix_topM(i + sDbscan::topM, Ri);

                if (iPointIdx == n)
                    continue;

                // Check already compute distance
//                if (approxNeighbor.find(iPointIdx) == approxNeighbor.end()) // cannot find
                if (!approxNeighbor[iPointIdx])
                {
//                    approxNeighbor.insert(iPointIdx);
                    approxNeighbor[iPointIdx] = true;

                    float fDist = computeDist(vecXn, sDbscan::matrix_X.col(iPointIdx), sDbscan::distance);

                    if (fDist <= eps)
                    {
                        // TODO: We might need asymmetric update vec2D_Neighbors to increase parallel capacity
                        // TODO: GPUs version might not suffer it

                        // https://stackoverflow.com/questions/33441767/difference-between-omp-critical-and-omp-single
#pragma omp critical
                        {
                        // set is very slow de to larger memory which is bottleneck on multi-threading
                        // Note that the collision rate is not large, and the size of vector is << 2km 
//                        set2D_DBSCAN_Neighbor[n].insert(iPointIdx);  
//                        set2D_DBSCAN_Neighbor[iPointIdx].insert(n);

                            vec2D_Neighbors[n].push_back(iPointIdx);
                            vec2D_Neighbors[iPointIdx].push_back(n);
                        }
                    }
                }
            }
        }

//        cout << "Number of used distances for the point of " << n << " is: " << approxNeighbor.size() << endl;
    }

//    omp_destroy_lock(&ompLock);

//    if (sDbscan::verbose)
//        cout << "Find neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

//    begin = chrono::steady_clock::now();

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads);

#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        // Using the set to clear duplicates
        unordered_set<int> setNeighbor(vec2D_Neighbors[n].begin(), vec2D_Neighbors[n].end());
        vec2D_Neighbors[n].clear();

        // Decide core points
        if ((int)setNeighbor.size() >= minPts - 1) // we count the point itself
        {
            bit_CorePoints[n] = true;

//            if ( n < 1000 )
//                cout << setNeighbor.size() << endl;

            // Only need neighborhood if it is core point to reduce the memory
            vec2D_Neighbors[n].insert(vec2D_Neighbors[n].end(), setNeighbor.begin(), setNeighbor.end());
        }

          // We might keep the neighborhood of noisy points in case we have to cluster the noise by its labeled neighborhood
          // However, we do not support it as neighborhood of noisy points is often tiny given small minPts
//        else if (sDbscan::clusterNoise == 3)
//        {
//            vec2D_Neighbors[n].insert(vec2D_Neighbors[n].end(), setNeighbor.begin(), setNeighbor.end());
//        }

    }

    if (sDbscan::verbose){
//        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
        cout << "Number of core points: " << bit_CorePoints.count() << endl;
    }


}

/**
 * This is a multi-thread friendly implementation of sngDBSCAN, much faster than the original sngDBSCAN on large data.
 *
 * @param eps
 * @param minPts
**/
void sDbscan::sng_findCorePoints(float eps, int minPts)
{
    // must init
    sDbscan::vec2D_Neighbors = vector<IVector> (sDbscan::n_points, IVector());
    sDbscan::bit_CorePoints = boost::dynamic_bitset<>(sDbscan::n_points);

    int iNumSamples = ceil(1.0 * sDbscan::n_points * sDbscan::samplingProb);

//    chrono::steady_clock::time_point begin;
//    begin = chrono::steady_clock::now();

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads); // set outside
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        VectorXf vecXn = sDbscan::matrix_X.col(n);

        if (sDbscan::samplingProb < 1.0)
        {
            // Sampling points to identify core points
            unsigned seed = chrono::system_clock::now().time_since_epoch().count();
            if (sDbscan::seed >= 0)
                seed = sDbscan::seed;
            default_random_engine generator(seed);

//            random_device rd;  // a seed source for the random number engine
//            mt19937 generator(rd()); // mersenne_twister_engine seeded with rd()

            uniform_int_distribution<> distrib(0, sDbscan::n_points - 1);

            // Compute distance from sampled Xn to all points in X
            int s = 0;
            while (s < iNumSamples)
            {
                int iPointIdx = distrib(generator);
                if (iPointIdx == n)
                    continue;

                s++; // increase s
                float fDist = computeDist(vecXn, sDbscan::matrix_X.col(iPointIdx), sDbscan::distance);

                if (fDist <= eps) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
                {
#pragma omp critical
                    {
                        sDbscan::vec2D_Neighbors[n].push_back(iPointIdx);
                        sDbscan::vec2D_Neighbors[iPointIdx].push_back(n);
                    }
                }
            }
        }
        else // exact Dbscan as sampling-prob = 1
        {
            // Compute distance from sampled Xn to all points in X
            for (int s = 0; s < sDbscan::n_points; ++s)
            {
                if (s == n)
                    continue;

                float fDist = computeDist(vecXn, sDbscan::matrix_X.col(s), sDbscan::distance);

                if (fDist <= eps) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
                    sDbscan::vec2D_Neighbors[n].push_back(s);
            }
        }
    }

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads); already set outside
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        // We do not need to use the set to clear duplicate as the chance of getting duplicate by sampling is very tiny
        if ((int)sDbscan::vec2D_Neighbors[n].size() >= minPts - 1)
        {
            sDbscan::bit_CorePoints[n] = true;
        }
        else // if (!sDbscan::clusterNoise) : do not support cluster Noise for sngDbscan as it does not increase accuracy with prob = 0.001
            sDbscan::vec2D_Neighbors[n].clear();
    }

    if (sDbscan::verbose)
    {
//        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
        cout << "Number of core points: " << sDbscan::bit_CorePoints.count() << endl;
    }

}

/**
 * A simple DFS algorithms to form clustering based on only identified core points and its approx neighbors
 */
void sDbscan::formCluster()
{
    // Must init here
    sDbscan::labels_ = IVector(sDbscan::n_points, -1); //noise = -1
    sDbscan::n_clusters_ = 0;

    int iNewClusterID = -1; // The cluster ID starts from 0

    // Fast enough so might not need multi-threading
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        // Skip: (1) core-point with assigned labels_, (2) non-core points

        if ((!sDbscan::bit_CorePoints[n]) || (sDbscan::labels_[n] != -1) )  //only consider core points and point without labels_
            continue;


        /** Note: There is a tradeoff between multi-threading speedup and clustering time
        TODO: Better parallel forming cluster with asymmetric neighborhood (perhaps it is useful on MapReduce/Spark/MPC
        - If call findCorePoints_Asym(), then we have to consider several seedSet and it will connect to the clustered point before
        Therefore, forming cluster takes time
        - If call rp_findCorePoints(), similar points are inserted into both arrays, then clusters tend to connect each other well.
        Therefore, forming clustering is very fast
        One can test with cout << n << endl;
        **/

        // Always start from the core points without any labels_
        iNewClusterID++;

        unordered_set<int> seedSet; //seedSet only contains core points
        seedSet.insert(n);

        /**
        # core points detected for each connected component is small (due to the approximation), so unorder_set() is fine.
        However, if this number is large, bitSet might be a good choice
//        boost::dynamic_bitset<> seedSet(PARAM_DATA_N);
//        seedSet[n] = 1;
        **/

        /**
        connectedPoints tend to have many points (e.g. n/20), then bitSet is better than unordered_set()
//        unordered_set<int> connectedPoints;
//        connectedPoints.insert(n);
        **/

	// We use bitset for large data since the cluster size tends to be large
	// If the cluster size is tiny or number of cluster is large, then unordered_set might be faster
        boost::dynamic_bitset<> connectedPoints(sDbscan::n_points);
        connectedPoints[n] = true;

        // unordered_set<int> is slow if there are many core points - google::dense_hash_set or bitSet might be faster
        // however, clustering is very fast compared to computing core points and its neighborhood - no need to improve at this stage
//        while (seedSet.count() > 0)
        while (seedSet.size() > 0)
        {
            int Xi = *seedSet.begin();
            seedSet.erase(seedSet.begin());

//            int Xi = seedSet.find_first();
//            seedSet[Xi] = 0;

            // Get neighborhood of the core Xi
            IVector Xi_neighborhood = vec2D_Neighbors[Xi];

            // Find the core points, connect them together, and check if one of them already has assigned labels_
            for (auto const& Xj : Xi_neighborhood)
            {
                // If core point and not connected, then add into seedSet
                if (bit_CorePoints[Xj])
                {
//                    if (connectedCore.find(Xj) == connectedCore.end())
                    if (! connectedPoints[Xj])
                    {
                        connectedPoints[Xj] = true;

                        if (sDbscan::labels_[Xj] == -1) // only insert into seedSet for non-labeled core; otherwise used the label of this labeled core
                            seedSet.insert(Xj);
//                            seedSet[Xj] = 1;
                    }
                }
                else
                    connectedPoints[Xj] = true;
            }
        }

//        cout << "Connecting component time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        // Assign the cluster label
//        begin = chrono::steady_clock::now();
//        for (auto const& Xj : connectedCore)
        size_t Xj = connectedPoints.find_first();
        while (Xj != boost::dynamic_bitset<>::npos)
        {
            // Get all neighborhood
            if (sDbscan::labels_[Xj] == -1)
                sDbscan::labels_[Xj] = iNewClusterID; // assign core

            // This code increases the accuracy for the non-core points Xi which has non-core points neighbor
            // It might be suitable for the data set with many noise
//            IVector Xj_neighborhood = vec2D_Neighbors[Xj];
//            for (auto const& Xk : Xj_neighborhood)
//            {
//                if (p_vecLabels[Xk] == -1)
//                    p_vecLabels[Xk] = iClusterID;
//            }

            Xj = connectedPoints.find_next(Xj);
        }

        // Update the largest cluster ID
        sDbscan::n_clusters_ = iNewClusterID;

//        cout << "Labelling components time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
    }

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    // Assign cluster label to noisy point -- it will increase NMI if comparing with the class labels_
    if (sDbscan::clusterNoise)
        labelNoise();

    if (sDbscan::verbose)
        cout << "Clustering noisy point time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    sDbscan::n_clusters_ = sDbscan::n_clusters_ + 2; // increase by 2 since we count -1 as noisy cluster, and cluster ID start from 0

    if (sDbscan::verbose)
        cout << "Number of clusters: " << sDbscan::n_clusters_ << endl;
}

/**
Cluster noisy points to increase NMI when comparing with class labels_
We only support (1) (2) among several methods:
1) Heuristic: Use any labeled points in the neighborhood to assign label to the noisy points.
2) Sampling 0.01n core points, use CEOs to approximate 1NN classifier
3) Assign label to random vectors, and use labels_ of random vectors
4) Sampling 0.01n core points to compute build 1NN classifier
**/
void sDbscan::labelNoise()
{
    // Counting noisy points
    IVector vecNoise;
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        if (sDbscan::labels_[n] == -1)
            vecNoise.push_back(n);
    }
    if (sDbscan::verbose)
        cout << "Number of noisy points: " << vecNoise.size() << endl;

    if (sDbscan::clusterNoise == 2)
    {
#pragma omp parallel for
        for (auto const& Xi : vecNoise)
        {
            // Find the core points, connect them together, and check if one of them already has assigned labels_
            // This solution slightly improve the NMI
            // Found it here: https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan/blob/master/subsampled_neighborhood_graph_dbscan.h
            // cluster remaining

//                cout << "Noise Point Idx: " << Xi << endl;

            for (auto const& Xj : sDbscan::vec2D_Neighbors[Xi])
            {
//                    cout << "Neighbor Point Idx: " << Xj << endl;
                // Two methods: only consider the core point or any labeled points in the neighborhood
//                if (sDbscan::labels_[Xj] >= 0)
                if (sDbscan::bit_CorePoints[Xj])
                {
                    sDbscan::labels_[Xi] = sDbscan::labels_[Xj];
                    break;
                }
            }
        }

    }
    else // sDbscan-1NN:
    {
        // First we get vector of sampled core points

        IVector vecSampledCore;
        int iNumCore = sDbscan::bit_CorePoints.count();

        /**
        Sampling 0.01 n # core points
        **/
        // Hack: We only consider 1% * n # core points
        float fProb = 1.0;
        if (iNumCore >= sDbscan::samplingProb * sDbscan::n_points)
            fProb = sDbscan::samplingProb * sDbscan::n_points / iNumCore;

        size_t Xi = bit_CorePoints.find_first();
        while (Xi != boost::dynamic_bitset<>::npos)
        {
            // Store the core point Idx
            if (fProb == 1.0)
                vecSampledCore.push_back(Xi);
            else if (rand() / (RAND_MAX + 1.) <= fProb)
                vecSampledCore.push_back(Xi);

            Xi = bit_CorePoints.find_next(Xi);
        }

        /** Compute again their random projections **/
        iNumCore = vecSampledCore.size(); // now it is sampled core points
        if (sDbscan::verbose)
            cout << "Number of sampled core points: " << vecSampledCore.size() << endl;

        MatrixXf matCoreEmbeddings = MatrixXf::Zero(iNumCore, sDbscan::n_proj);

        /** Param for embedding L1 and L2 **/
        int iFourierEmbed_D = sDbscan::ker_n_features / 2; // This is because we need cos() and sin()
        int log2Project = log2(sDbscan::fhtDim);

//    omp_set_num_threads(sDbscan::n_threads); // already set outside
#pragma omp parallel for
        for (int i = 0; i < iNumCore; ++i) // iNumCore is number of sampled core points
        {
            int Xi = vecSampledCore[i];

            /**
            Random embedding
            **/
            VectorXf vecX = sDbscan::matrix_X.col(Xi);
            VectorXf vecEmbed = VectorXf::Zero(sDbscan::ker_n_features); // PARAM_KERNEL_EMBED_D >= D

            if (sDbscan::distance == "Cosine")
                vecEmbed.segment(0, sDbscan::n_features) = vecX;
            else if ((sDbscan::distance == "L1") || (sDbscan::distance == "L2"))
            {
                VectorXf vecProject = sDbscan::matrix_R * vecX;
                vecEmbed.segment(0, iFourierEmbed_D) = vecProject.array().cos();
                vecEmbed.segment(iFourierEmbed_D, iFourierEmbed_D) = vecProject.array().sin(); // start from iEmbbed, copy iEmbed elements
            }
            else if (sDbscan::distance == "Chi2")
                embedChi2(vecX, sDbscan::ker_n_features, sDbscan::n_features, sDbscan::ker_intervalSampling, vecEmbed);
            else if (sDbscan::distance == "JS")
                embedJS(vecX, sDbscan::ker_n_features, sDbscan::n_features, sDbscan::ker_intervalSampling, vecEmbed);

            /**
            Random projection
            **/

            VectorXf vecRotation = VectorXf::Zero(sDbscan::fhtDim); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
            vecRotation.segment(0, sDbscan::ker_n_features) = vecEmbed;

            for (int r = 0; r < sDbscan::n_rotate; ++r)
            {
                // Component-wise multiplication with a random sign
                for (int d = 0; d < sDbscan::fhtDim; ++d)
                {
                    vecRotation(d) *= (2 * (int)sDbscan::bitHD3[r * sDbscan::fhtDim + d] - 1);
                }

                // Multiple with Hadamard matrix by calling FWHT transform
                fht_float(vecRotation.data(), log2Project);
            }

            // Store projection matrix for faster parallel, and no need to scale since we keep top-k and top-MinPts
            matCoreEmbeddings.row(i) = vecRotation.segment(0, sDbscan::n_proj); // only get up to #n_proj
        }

        /** Estimate distance using CEOs and 1NN classifer **/
        // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads);
#pragma omp parallel for
        for (auto const& Xi : vecNoise)
        {
            // Get top-k closest/furthest random vectors
            VectorXi vecRandom = sDbscan::matrix_topK.col(Xi);
            VectorXf vecDotEst = VectorXf::Zero(iNumCore);

            for (int k = 0; k < sDbscan::topK; ++k)
            {
//                int closeRi = vecRandom(k);
                vecDotEst += matCoreEmbeddings.col(vecRandom(k));

//                int farRi = vecRandom(k + sDbscan::topK);
                vecDotEst -= matCoreEmbeddings.col(vecRandom(k + sDbscan::topK));
            }

            // Eigen::VectorXf::Index max_index;
            // vecDotEst.maxCoeff(&max_index);

            int iCoreIdx = -1;
            float best_so_far = NEG_INF;

            for (int i = 0; i < iNumCore; ++i)
            {
                if (vecDotEst(i) > best_so_far)
                {
                    best_so_far = vecDotEst(i);
                    iCoreIdx = vecSampledCore[i];
                }
            }

            sDbscan::labels_[Xi] = sDbscan::labels_[iCoreIdx];

        }

    }

    // Counting noisy points after label assignment
//    vecNoise.clear();
//    for (int n = 0; n < sDbscan::n_points; ++n)
//    {
//        if (p_vecLabels[n] == -1)
//            vecNoise.push_back(n);
//    }
//
//    if (sDbscan::verbose)
//        cout << "After labelling, the number of noisy points: " << vecNoise.size() << endl;

}

/**
 *
 * @param MATRIX_X
 * @param eps
 * @param minPts
 */
void sDbscan::fit_sDbscan(const Ref<const MatrixXf> & MATRIX_X, float eps, int minPts)
{
    if (sDbscan::verbose)
    {
        cout << "eps: " << eps << endl;
        cout << "minPts: " << minPts << endl;

        cout << "n_points: " << sDbscan::n_points << endl;
        cout << "n_features: " << sDbscan::n_features << endl;
        cout << "n_proj: " << sDbscan::n_proj << endl;
        cout << "topK: " << sDbscan::topK << endl;
        cout << "topM: " << sDbscan::topM << endl;
        cout << "distance: " << sDbscan::distance << endl;
        cout << "cluster noise: " << sDbscan::clusterNoise << endl;
        cout << "kernel features: " << sDbscan::ker_n_features << endl;
        cout << "sigma: " << sDbscan::ker_sigma << endl;
        cout << "interval sampling: " << sDbscan::ker_intervalSampling << endl;
        cout << "sDbscan-1NN prob: " << sDbscan::samplingProb << endl;
        cout << "n_threads: " << sDbscan::n_threads << endl;
    }

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    sDbscan::matrix_X = MATRIX_X;
    transformData(sDbscan::matrix_X, sDbscan::distance);
    if (sDbscan::verbose)
        cout << "Check X supporting distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    rp_parIndex();
    if (sDbscan::verbose)
        cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    rp_findCorePoints(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formCluster();
    if (sDbscan::verbose)
        cout << "Form clusters time  (including clustering noise) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (sDbscan::verbose)
    {
        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_KerFeatures_" + int2str(sDbscan::ker_n_features) +
                           "_NumProj_" + int2str(sDbscan::n_proj) +
                           "_TopM_" + int2str(sDbscan::topM) +
                           "_TopK_" + int2str(sDbscan::topK);

        outputDbscan(sDbscan::labels_, sFileName);
    }
}

/**
 * We need to load data into sDbscan::matrix_X first to save memory
 * We then generate the indexing structures, including matrix_topK, matrix_topM that contain potential candidates
 * For each point, compute the distance with candidates suggested in the index
 * Given the list of core points and its neighborhood, we form density-based clustering
 * Note we can use sDbscan-1NN to cluster noise or border/core points that are misclassified by the random projection
 *
 * @param dataset: the filename of data set. We load data and store into sDbscan::matrix_X for saving memory
 * @param eps
 * @param minPts
 */
void sDbscan::load_fit_sDbscan(const string& dataset, float eps, int minPts)
{
    if (sDbscan::verbose)
    {
        cout << "eps: " << eps << endl;
        cout << "minPts: " << minPts << endl;
        cout << "dataset filename: " << dataset << endl;

        cout << "n_points: " << sDbscan::n_points << endl;
        cout << "n_features: " << sDbscan::n_features << endl;
        cout << "n_proj: " << sDbscan::n_proj << endl;
        cout << "topK: " << sDbscan::topK << endl;
        cout << "topM: " << sDbscan::topM << endl;
        cout << "distance: " << sDbscan::distance << endl;
        cout << "Cluster noise: " << sDbscan::clusterNoise << endl;
        cout << "kernel features: " << sDbscan::ker_n_features << endl;
        cout << "sigma: " << sDbscan::ker_sigma << endl;
        cout << "interval sampling: " << sDbscan::ker_intervalSampling << endl;
        cout << "sDbscan-1NN prob: " << sDbscan::samplingProb << endl;
        cout << "n_threads: " << sDbscan::n_threads << endl;
    }

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin, start;
    begin = chrono::steady_clock::now();
    loadtxtData(dataset, sDbscan::distance, sDbscan::n_points, sDbscan::n_features, sDbscan::matrix_X);
    if (sDbscan::verbose)
        cout << "Loading data time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    start = chrono::steady_clock::now();

    begin = chrono::steady_clock::now();
    rp_parIndex();
    if (sDbscan::verbose)
        cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    rp_findCorePoints(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formCluster();
    if (sDbscan::verbose)
        cout << "Form clusters time (including clustering noise) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Clustering time (excluding loading data) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (sDbscan::verbose)
    {
        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_KerFeatures_" + int2str(sDbscan::ker_n_features) +
                           "_NumProj_" + int2str(sDbscan::n_proj) +
                           "_TopM_" + int2str(sDbscan::topM) +
                           "_TopK_" + int2str(sDbscan::topK);

        outputDbscan(sDbscan::labels_, sFileName);
    }
}

/**
 * We need to generate the indexing structures, including matrix_topK, matrix_topM that contain potential candidates
 * For each point, compute the distance with candidates suggested in the index
 * Given the list of core points and its neighborhood, we form density-based clustering
 * Note we can use sDbscan-1NN to cluster noise or border/core points that are misclassified by the random projection
 *
 * @param MATRIX_X
 * @param eps
 * @param rangeEps: We repeat n_tests times, each use eps + i * rangeEps
 * @param n_tests
 * @param minPts
 */
void sDbscan::test_sDbscan(const Ref<const MatrixXf> & MATRIX_X, float eps, float rangeEps, int n_tests, int minPts)
{
    cout << "base eps: " << eps << endl;
    cout << "range eps: " << rangeEps << endl;
    cout << "n_tests: " << n_tests << endl;
    cout << "minPts: " << minPts << endl;

    cout << "n_points: " << sDbscan::n_points << endl;
    cout << "n_features: " << sDbscan::n_features << endl;
    cout << "n_proj: " << sDbscan::n_proj << endl;
    cout << "topK: " << sDbscan::topK << endl;
    cout << "topM: " << sDbscan::topM << endl;
    cout << "distance: " << sDbscan::distance << endl;
    cout << "Cluster noise: " << sDbscan::clusterNoise << endl;
    cout << "kernel features: " << sDbscan::ker_n_features << endl;
    cout << "sigma: " << sDbscan::ker_sigma << endl;
    cout << "interval sampling: " << sDbscan::ker_intervalSampling << endl;
    cout << "sDbscan-1NN prob: " << sDbscan::samplingProb << endl;
    cout << "n_threads: " << sDbscan::n_threads << endl;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    sDbscan::matrix_X = MATRIX_X;
    transformData(sDbscan::matrix_X, sDbscan::distance);
    sDbscan::verbose = true; // set true since we want to test

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    rp_parIndex();
    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Try several n_tests
    for (int i = 0; i < n_tests; ++i)
    {
        float new_eps = eps + 1.0 * i * rangeEps;

        cout << "Eps: " << new_eps << endl;

        begin = chrono::steady_clock::now();
        rp_findCorePoints(new_eps, minPts);
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        begin = chrono::steady_clock::now();
        formCluster();
        cout << "Form clusters time (including clustering noise) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * new_eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_KerFeatures_" + int2str(sDbscan::ker_n_features) +
                           "_NumProj_" + int2str(sDbscan::n_proj) +
                           "_TopM_" + int2str(sDbscan::topM) +
                           "_TopK_" + int2str(sDbscan::topK);

        outputDbscan(sDbscan::labels_, sFileName);

    }
}

/**
 * We need to generate the indexing structures, including matrix_topK, matrix_topM that contain potential candidates
 * For each point, compute the distance with candidates suggested in the index
 * Given the list of core points and its neighborhood, we form density-based clustering
 * Note we can use sDbscan-1NN to cluster noise or border/core points that are misclassified by the random projection
 *
 * @param MATRIX_X
 * @param eps
 * @param rangeEps: We repeat n_tests times, each use eps + i * rangeEps
 * @param n_tests
 * @param minPts
 */
void sDbscan::load_test_sDbscan(const string& dataset, float eps, float rangeEps, int n_tests, int minPts)
{
    cout << "base eps: " << eps << endl;
    cout << "range eps: " << rangeEps << endl;
    cout << "n_tests: " << n_tests << endl;
    cout << "minPts: " << minPts << endl;
    cout << "dataset: " << dataset << endl;

    cout << "n_points: " << sDbscan::n_points << endl;
    cout << "n_features: " << sDbscan::n_features << endl;
    cout << "n_proj: " << sDbscan::n_proj << endl;
    cout << "topK: " << sDbscan::topK << endl;
    cout << "topM: " << sDbscan::topM << endl;
    cout << "distance: " << sDbscan::distance << endl;
    cout << "Cluster noise: " << sDbscan::clusterNoise << endl;
    cout << "kernel features: " << sDbscan::ker_n_features << endl;
    cout << "sigma: " << sDbscan::ker_sigma << endl;
    cout << "interval sampling: " << sDbscan::ker_intervalSampling << endl;
    cout << "sDbscan-1NN prob: " << sDbscan::samplingProb << endl;
    cout << "n_threads: " << sDbscan::n_threads << endl;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    sDbscan::verbose = true; // set true since we want to test

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    loadtxtData(dataset, sDbscan::distance, sDbscan::n_points, sDbscan::n_features, sDbscan::matrix_X);
    cout << "Loading data time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    rp_parIndex();
    cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Try several eps
    for (int i = 0; i < n_tests; ++i)
    {
        float new_eps = eps + 1.0 * i * rangeEps;

        cout << "Eps: " << new_eps << endl;

        begin = chrono::steady_clock::now();
        rp_findCorePoints(new_eps, minPts);
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        begin = chrono::steady_clock::now();
        formCluster();
        cout << "Form clusters time (including clustering noise) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * new_eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_KerFeatures_" + int2str(sDbscan::ker_n_features) +
                           "_NumProj_" + int2str(sDbscan::n_proj) +
                           "_TopM_" + int2str(sDbscan::topM) +
                           "_TopK_" + int2str(sDbscan::topK);

        outputDbscan(sDbscan::labels_, sFileName);

    }
}

/**
 * This is sngDbscan that find core using sampling method (NeurIPS 20)
 * https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan
 *
 * Note:
 * - Our sngDBSCAN only connects points with their neighborhoods if they are core points
 * - The origin sngDBSCAN always connects points with their neighborhood though they are not core points
 * (see Line 44 - 45 of https://github.com/jenniferjang/subsampled_neighborhood_graph_dbscan/blob/master/subsampled_neighborhood_graph_dbscan_preallocated.h)
 * - The origin sngDBSCAN might give higher NMI in case noise X1 connecting to non-core X2, X3, and by some change X3 is connected to the core point C1
 *
 * @param MATRIX_X
 * @param eps
 * @param minPts
 */
void sDbscan::fit_sngDbscan(const Ref<const MatrixXf> & MATRIX_X, float eps, int minPts)
{
    if (sDbscan::verbose)
    {
        cout << "eps: " << eps << endl;
        cout << "minPts: " << minPts << endl;
        cout << "n_points: " << sDbscan::n_points << endl;
        cout << "n_features: " << sDbscan::n_features << endl;
        cout << "distance: " << sDbscan::distance << endl;
        cout << "sampling prob: " << sDbscan::samplingProb << endl;
        cout << "n_threads: " << sDbscan::n_threads << endl;
        cout << "cluster noise: " << sDbscan::clusterNoise << endl;
    }

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    sDbscan::matrix_X = MATRIX_X;
    transformData(sDbscan::matrix_X, sDbscan::distance);
    if (sDbscan::verbose)
        cout << "Check X supporting distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // we do not support sngDbscan as it does not improve the accuracy since prob = 0.01 or 0.001 tends to be very small
    sDbscan::clusterNoise = false;

    // Find core point
    begin = chrono::steady_clock::now();
    sng_findCorePoints(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formCluster();
    if (sDbscan::verbose)
        cout << "Form clusters (including clustering noise) time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (sDbscan::verbose)
    {
        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_Prob_" + int2str(round(1000 * sDbscan::samplingProb));

        outputDbscan(sDbscan::labels_, sFileName);
    }
}

/**
 * We need to generate the indexing structures, including matrix_topK, matrix_topM that contain potential candidates
 * For each point, compute the distance with candidates suggested in the index
 * Given the list of core points and its neighborhood, we form density-based clustering
 * Note we can use sDbscan-1NN to cluster noise or border/core points that are misclassified by the random projection
 *
 * @param MATRIX_X
 * @param eps
 * @param rangeEps: We repeat 5 times, each use eps + i * rangeEps
 * @param minPts
 */
void sDbscan::test_sngDbscan(const Ref<const MatrixXf> & MATRIX_X, float eps, float rangeEps, int n_tests, int minPts)
{
    cout << "base eps: " << eps << endl;
    cout << "range eps: " << rangeEps << endl;
    cout << "n_tests: " << n_tests << endl;
    cout << "minPts: " << minPts << endl;

    cout << "minPts: " << minPts << endl;
    cout << "n_points: " << sDbscan::n_points << endl;
    cout << "n_features: " << sDbscan::n_features << endl;
    cout << "distance: " << sDbscan::distance << endl;
    cout << "sampling prob: " << sDbscan::samplingProb << endl;
    cout << "n_threads: " << sDbscan::n_threads << endl;
    cout << "cluster noise: " << sDbscan::clusterNoise << endl;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    sDbscan::matrix_X = MATRIX_X;
    transformData(sDbscan::matrix_X, sDbscan::distance);

    sDbscan::verbose = true; // set true since we want to test

    // Try several eps
    for (int i = 0; i < n_tests; ++i)
    {
        float new_eps = eps + 1.0 * i * rangeEps;

        cout << "Eps: " << new_eps << endl;

        begin = chrono::steady_clock::now();
        sng_findCorePoints(new_eps, minPts);
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        begin = chrono::steady_clock::now();
        formCluster();
        cout << "Form clusters time (including clustering noise) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * new_eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_Prob_" + int2str(round(1000 * sDbscan::samplingProb));

        outputDbscan(sDbscan::labels_, sFileName);

    }
}

/**
 * We need to generate the indexing structures, including matrix_topK, matrix_topM that contain potential candidates
 * For each point, compute the distance with candidates suggested in the index
 * Given the list of core points and its neighborhood, we form density-based clustering
 * Note we can use sDbscan-1NN to cluster noise or border/core points that are misclassified by the random projection
 *
 * @param dataset
 * @param eps
 * @param rangeEps: We repeat 5 times, each use eps + i * rangeEps
 * @param minPts
 */
void sDbscan::load_test_sngDbscan(const string& dataset, float eps, float rangeEps, int n_tests, int minPts)
{
    cout << "base eps: " << eps << endl;
    cout << "range eps: " << rangeEps << endl;
    cout << "n_tests: " << n_tests << endl;
    cout << "minPts: " << minPts << endl;
    cout << "dataset: " << dataset << endl;

    cout << "minPts: " << minPts << endl;
    cout << "n_points: " << sDbscan::n_points << endl;
    cout << "n_features: " << sDbscan::n_features << endl;
    cout << "distance: " << sDbscan::distance << endl;
    cout << "sampling prob: " << sDbscan::samplingProb << endl;
    cout << "n_threads: " << sDbscan::n_threads << endl;
    cout << "cluster noise: " << sDbscan::clusterNoise << endl;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    sDbscan::verbose = true; // set true since we want to test

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    loadtxtData(dataset, sDbscan::distance, sDbscan::n_points, sDbscan::n_features, sDbscan::matrix_X);
    cout << "Loading data time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Try several eps
    for (int i = 0; i < n_tests; ++i)
    {
        float new_eps = eps + 1.0 * i * rangeEps;

        cout << "Eps: " << new_eps << endl;

        begin = chrono::steady_clock::now();
        sng_findCorePoints(new_eps, minPts);
        cout << "Find core points time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        begin = chrono::steady_clock::now();
        formCluster();
        cout << "Form clusters time (including clustering noise) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        string sFileName = sDbscan::output + + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * new_eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_Prob_" + int2str(round(1000 * sDbscan::samplingProb));

        outputDbscan(sDbscan::labels_, sFileName);

    }
}

/**
 * Finding core points and its approximate neighborhood and approximate core distance using random projection
 * - Store them in vec2D_NeighborDist
 * - Used for sOptics
 *
 * @param eps
 * @param minPts
 */
void sDbscan::rp_findCoreDist(float eps, int minPts)
{
    sDbscan::vec2D_NeighborDist = vector< vector< pair<int, float> > > (sDbscan::n_points, vector< pair<int, float> >());

//    chrono::steady_clock::time_point begin;
//    begin = chrono::steady_clock::now();

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads);
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        VectorXf vecXn = sDbscan::matrix_X.col(n);

        VectorXi vecTopK = sDbscan::matrix_topK.col(n); // size 2K: first K is close, last K is far

        boost::dynamic_bitset<> approxNeighbor(sDbscan::n_points);

        for (int k = 0; k < sDbscan::topK; ++k)
        {
            // Closest
            int Ri = vecTopK(k);

            for (int i = 0; i < sDbscan::topM; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = sDbscan::matrix_topM(i, Ri);

                if (iPointIdx == n)
                    continue;

                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = true;

//                    cout << "The point: " << n << endl;
//                    cout << vecXn << endl;
//
//                    cout << "The point: " << iPointIdx << endl;
//                    cout << MATRIX_X.col(iPointIdx) << endl;

                    float fDist = computeDist(vecXn, sDbscan::matrix_X.col(iPointIdx), sDbscan::distance);

//                    if (fDist < 0)
//                    {
//                        cout << n << " " << iPointIdx << ": " << fDist << endl;
//                        cout << 1 - vecXn.dot(MATRIX_X.col(iPointIdx)) << endl;
//                    }

                    if (fDist <= eps)
                    {

#pragma omp critical
                        {
//                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDist));
//                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDist));

                            sDbscan::vec2D_NeighborDist[n].emplace_back(iPointIdx, fDist); // duplicate at most twice
                            sDbscan::vec2D_NeighborDist[iPointIdx].emplace_back(n, fDist); // so vector is much better than map()
                        }
                    }
                }
            }

            // Far
            Ri = vecTopK(k + sDbscan::topK);

            for (int i = 0; i < sDbscan::topM; ++i)
            {
                // Compute distance between Xn and Xi
                int iPointIdx = sDbscan::matrix_topM(i + sDbscan::topM, Ri);

                if (iPointIdx == n)
                    continue;

                if (!approxNeighbor[iPointIdx]) // cannot find
                {
                    approxNeighbor[iPointIdx] = true;

                    float fDist = computeDist(vecXn, sDbscan::matrix_X.col(iPointIdx), sDbscan::distance);

                    if (fDist <= eps)
                    {
                        // omp_set_dynamic(0);     // Explicitly disable dynamic teams
                        omp_set_num_threads(sDbscan::n_threads);
#pragma omp critical
                        {
//                        map2D_DBSCAN_Neighbor[n].insert(make_pair(iPointIdx, fDist));
//                        map2D_DBSCAN_Neighbor[iPointIdx].insert(make_pair(n, fDist));

                            sDbscan::vec2D_NeighborDist[n].emplace_back(iPointIdx, fDist);
                            sDbscan::vec2D_NeighborDist[iPointIdx].emplace_back(n, fDist);

                        }
                    }
                }
            }
        }
    }

//    if (sDbscan::verbose)
//        cout << "Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

//    begin = chrono::steady_clock::now();

    sDbscan::bit_CorePoints = boost::dynamic_bitset<>(sDbscan::n_points);
    sDbscan::vec_CoreDist = FVector(sDbscan::n_points, 0.0);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads);
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        // Store idx and float in the same order
        unordered_map<int, float> mapNeighborhood(sDbscan::vec2D_NeighborDist[n].begin(), sDbscan::vec2D_NeighborDist[n].end());

        sDbscan::vec2D_NeighborDist[n].clear();
        sDbscan::vec2D_NeighborDist[n].insert(sDbscan::vec2D_NeighborDist[n].end(), mapNeighborhood.begin(), mapNeighborhood.end());
        mapNeighborhood.clear();

        if ((int)sDbscan::vec2D_NeighborDist[n].size() >= minPts - 1)
        {
            sDbscan::bit_CorePoints[n] = true;

            // Only sort if this is the core
            FVector vecNeighborhood;
            for (const auto &ifPair : sDbscan::vec2D_NeighborDist[n])
                vecNeighborhood.push_back(ifPair.second);

            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + minPts - 2, vecNeighborhood.end()); // default is X1 < X2 < ...

            // Store core dist
            sDbscan::vec_CoreDist[n] = vecNeighborhood[minPts - 2]; // scikit learn includes the point itself, and C++ index start from 0

            // test
//            cout << "Core dist: " << vec_CORE_DIST[n] << endl;
//            sort(vecNeighborhood.begin(), vecNeighborhood.end());
//
//            cout << "Begin testing " << endl;
//            for (int i = 0; i < vecNeighborhood.size(); ++i)
//            {
//                cout << vecNeighborhood[i] << endl;
//            }
//
//            cout << "End testing " << endl;
        }
    }
//    float fSum = 0.0;
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//        fSum += vec2D_OPTICS_NeighborDist[n].size();
//    cout << "Size of data structure: " << fSum << endl;

    if (sDbscan::verbose)
    {
//        cout << "CoreDist time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        cout << "Number of core points: " << sDbscan::bit_CorePoints.count() << endl;
    }


}

/**
 * Finding core points and its approximate neighborhood and approximate core distance using random sampling
 * - Store them in vec2D_NeighborDist
 * - Used for sOptics
 *
 * @param eps
 * @param minPts
 */
void sDbscan::sng_findCoreDist(float eps, int minPts)
{
    sDbscan:: vec2D_NeighborDist = vector< vector< pair<int, float> > > (sDbscan::n_points, vector< pair<int, float> >());

//    chrono::steady_clock::time_point begin;
//    begin = chrono::steady_clock::now();
    int iNumSamples = ceil(sDbscan::samplingProb * sDbscan::n_points);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads);
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        VectorXf vecXn = sDbscan::matrix_X.col(n);

        // Sampling points to identify core points
        random_device rd;  // a seed source for the random number engine
        mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
        uniform_int_distribution<> distrib(0, sDbscan::n_points - 1);

        // Compute distance from sampled Xn to all points in X
        for (int s = 0; s < iNumSamples; ++s) {
            int iPointIdx = distrib(gen);
            if (iPointIdx == n)
                continue;

            float fDist = computeDist(vecXn, sDbscan::matrix_X.col(iPointIdx), sDbscan::distance);

            if (fDist <= eps) // if (vecXn.dot(MATRIX_X_EMBED.col(iPointIdx)) >= fThresDot)
            {
#pragma omp critical
                {
                    sDbscan:: vec2D_NeighborDist[n].emplace_back(iPointIdx, fDist); // duplicate at most twice
                    sDbscan:: vec2D_NeighborDist[iPointIdx].emplace_back(n, fDist); // so vector is much better than map()
                }
            }
        }
    }

//    if (sDbscan::verbose)
//        cout << "Neighborhood time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

//    begin = chrono::steady_clock::now();

    sDbscan::bit_CorePoints = boost::dynamic_bitset<>(sDbscan::n_points);
    sDbscan::vec_CoreDist = FVector(sDbscan::n_points);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(sDbscan::n_threads);
#pragma omp parallel for
    for (int n = 0; n < sDbscan::n_points; ++n)
    {
        // Store idx and float in the same order
        unordered_map<int, float> mapNeighborhood(sDbscan:: vec2D_NeighborDist[n].begin(), sDbscan:: vec2D_NeighborDist[n].end());

        sDbscan:: vec2D_NeighborDist[n].clear();
        sDbscan:: vec2D_NeighborDist[n].insert(sDbscan:: vec2D_NeighborDist[n].end(), mapNeighborhood.begin(), mapNeighborhood.end());
        mapNeighborhood.clear();

        if ((int)sDbscan:: vec2D_NeighborDist[n].size() >= minPts - 1)
        {
            sDbscan::bit_CorePoints[n] = true;

            // Only sort if this is the core
            FVector vecNeighborhood;
            for (const auto &ifPair : sDbscan:: vec2D_NeighborDist[n])
                vecNeighborhood.push_back(ifPair.second);

            nth_element(vecNeighborhood.begin(), vecNeighborhood.begin() + minPts - 2, vecNeighborhood.end()); // default is X1 < X2 < ...

            // Store core dist
            sDbscan::vec_CoreDist[n] = vecNeighborhood[minPts - 2]; // scikit learn includes the point itself, and C++ index start from 0

            // test
//            cout << "Core dist: " << vec_CORE_DIST[n] << endl;
//            sort(vecNeighborhood.begin(), vecNeighborhood.end());
//
//            cout << "Begin testing " << endl;
//            for (int i = 0; i < vecNeighborhood.size(); ++i)
//            {
//                cout << vecNeighborhood[i] << endl;
//            }
//
//            cout << "End testing " << endl;
        }
    }
//    float fSum = 0.0;
//    for (int n = 0; n < PARAM_DATA_N; ++n)
//        fSum += vec2D_OPTICS_NeighborDist[n].size();
//    cout << "Size of data structure: " << fSum << endl;

    if (sDbscan::verbose)
    {
//        cout << "CoreDist time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

        cout << "Number of core points: " << sDbscan::bit_CorePoints.count() << endl;
    }


}

/**
 * Form Optics by computing ordering_ and reachability_ for each point
 */
void sDbscan::formOptics_scikit()
{
    sDbscan::reachability_ = FVector(sDbscan::n_points, NEG_INF); // NEG_INF = -2^31
    sDbscan::ordering_ = IVector();

    boost::dynamic_bitset<> processSet(sDbscan::n_points);

    for (int Xi = 0; Xi < sDbscan::n_points; ++Xi)
    {
        if (processSet[Xi])
            continue;

        processSet[Xi] = true;
        sDbscan::ordering_.push_back(Xi);


        // Only deal with core points since they affect reachable-dist
        // If it is not a core point, then its reachable-dist will be updated later
        if (sDbscan::bit_CorePoints[Xi])
        {
//            unordered_map<int, float> seedSet; // One point might be appear several time in the PQ, but max # element = n * MinPts * 2k
            Min_PQ_Pair seedSet;

            float Xi_core_dist = sDbscan::vec_CoreDist[Xi];
            vector< pair<int, float> > Xi_neighborhood = sDbscan::vec2D_NeighborDist[Xi];

            // For all: Xj is neighbor of core Xi
            for (int j = 0; j < (int)Xi_neighborhood.size(); ++j)
            {
                int Xj = Xi_neighborhood[j].first;

                // only update if it is not processed
                if (processSet[Xj])
                    continue;

                float dist_XiXj = Xi_neighborhood[j].second;

                float Xj_reach_dist_from_Xi = max(Xi_core_dist, dist_XiXj);

                if (sDbscan::reachability_[Xj] == NEG_INF)
                {
                    sDbscan::reachability_[Xj] = Xj_reach_dist_from_Xi;
//                    seedSet.insert(make_pair(Xj, p_vecReachDist[Xj]));
                    seedSet.emplace(Xj, sDbscan::reachability_[Xj]); // reach from Xi

                }

                else if (sDbscan::reachability_[Xj] > Xj_reach_dist_from_Xi) // Xj is already reached by some point, but reach from Xi smaller
                {
                    sDbscan::reachability_[Xj] = Xj_reach_dist_from_Xi;
//                    seedSet[Xj] = p_vecReachDist[Xj];
                    seedSet.emplace(Xj, sDbscan::reachability_[Xj]);

                }
            }

            while (!seedSet.empty())
            {
                // Get minimum value by iterative the seedSet
//                int Xj = seedSet.begin()->first;
//                float fMin = seedSet.begin()->second;
//                for (auto i = seedSet.begin(); i != seedSet.end(); i++)
//                {
//                    if (i->second < fMin)
//                    {
//                        Xj = i->first;
//                        fMin = i->second;
//                    }
//                }
//                seedSet.erase(Xj);

                int Xj = seedSet.top().m_iIndex;
                seedSet.pop();

                if (processSet[Xj])
                    continue;

                processSet[Xj] = true; // process
                sDbscan::ordering_.push_back(Xj);

                if (sDbscan::bit_CorePoints[Xj])
                {
                    float Xj_core_dist = sDbscan::vec_CoreDist[Xj];
                    vector< pair<int, float> > Xj_neighborhood = sDbscan::vec2D_NeighborDist[Xj];

                    // Xj is neighbor of core Xi
                    for (int k = 0; k < (int)Xj_neighborhood.size(); ++k)
                    {
                        int Xk = Xj_neighborhood[k].first;

                        // only update if it is not processed
                        if (processSet[Xk])
                            continue;

                        float dist_XjXk = Xj_neighborhood[k].second;
                        float Xk_reach_dist_from_Xj = max(Xj_core_dist, dist_XjXk);

                        if (sDbscan::reachability_[Xk] == NEG_INF)
                        {
                            sDbscan::reachability_[Xk] = Xk_reach_dist_from_Xj;
//                            seedSet.insert(make_pair(Xk, p_vecReachDist[Xk]));
                            seedSet.emplace(Xk, sDbscan::reachability_[Xk]);
                        }

                        else if (sDbscan::reachability_[Xk] > Xk_reach_dist_from_Xj)
                        {
                            sDbscan::reachability_[Xk] = Xk_reach_dist_from_Xj;
//                            seedSet[Xk] = p_vecReachDist[Xk];
                            seedSet.emplace(Xk, sDbscan::reachability_[Xk]);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Get Optics: ordering_ and reachablity_
 * @param MATRIX_X
 * @param eps
 * @param minPts
 */
void sDbscan::fit_sOptics(const Ref<const MatrixXf> & MATRIX_X, float eps, int minPts)
{
    if (sDbscan::verbose)
    {
        cout << "eps: " << eps << endl;
        cout << "minPts: " << minPts << endl;
        cout << "n_points: " << sDbscan::n_points << endl;
        cout << "n_features: " << sDbscan::n_features << endl;
        cout << "n_proj: " << sDbscan::n_proj << endl;
        cout << "topK: " << sDbscan::topK << endl;
        cout << "topM: " << sDbscan::topM << endl;
        cout << "distance: " << sDbscan::distance << endl;
        cout << "kernel features: " << sDbscan::ker_n_features << endl;
        cout << "sigma: " << sDbscan::ker_sigma << endl;
        cout << "interval sampling: " << sDbscan::ker_intervalSampling << endl;
        cout << "n_threads: " << sDbscan::n_threads << endl;
    }

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    sDbscan::matrix_X = MATRIX_X;
    transformData(sDbscan::matrix_X, sDbscan::distance);

    if (sDbscan::verbose)
        cout << "Check X supporting distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    rp_parIndex();
    if (sDbscan::verbose)
        cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    sDbscan::rp_findCoreDist(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formOptics_scikit();
    if (sDbscan::verbose)
        cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (!sDbscan::output.empty())
    {
        string sFileName = sDbscan::output + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_NumEmbed_" + int2str(sDbscan::ker_n_features) +
                           "_NumProj_" + int2str(sDbscan::n_proj) +
                           "_TopM_" + int2str(sDbscan::topM) +
                           "_TopK_" + int2str(sDbscan::topK);

        outputOptics(sDbscan::ordering_, sDbscan::reachability_, sFileName);
    }
}

/**
 *
 * @param dataset
 * @param eps
 * @param minPts
 */
void sDbscan::load_fit_sOptics(const string& dataset, float eps, int minPts)
{
    if (sDbscan::verbose)
    {
        cout << "eps: " << eps << endl;
        cout << "minPts: " << minPts << endl;
        cout << "n_points: " << sDbscan::n_points << endl;
        cout << "n_features: " << sDbscan::n_features << endl;
        cout << "n_proj: " << sDbscan::n_proj << endl;
        cout << "topK: " << sDbscan::topK << endl;
        cout << "topM: " << sDbscan::topM << endl;
        cout << "distance: " << sDbscan::distance << endl;
        cout << "kernel features: " << sDbscan::ker_n_features << endl;
        cout << "sigma: " << sDbscan::ker_sigma << endl;
        cout << "interval sampling: " << sDbscan::ker_intervalSampling << endl;
        cout << "n_threads: " << sDbscan::n_threads << endl;
    }

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin, start;

    begin = chrono::steady_clock::now();
    loadtxtData(dataset, sDbscan::distance, sDbscan::n_points, sDbscan::n_features, sDbscan::matrix_X);
    if (sDbscan::verbose)
        cout << "Loading data time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    start = chrono::steady_clock::now();

    begin = chrono::steady_clock::now();
    rp_parIndex();
    if (sDbscan::verbose)
        cout << "Build index time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    sDbscan::rp_findCoreDist(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formOptics_scikit();
    if (sDbscan::verbose)
        cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Form optics time (excluding loading data) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (!sDbscan::output.empty())
    {
        string sFileName = sDbscan::output + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_NumEmbed_" + int2str(sDbscan::ker_n_features) +
                           "_NumProj_" + int2str(sDbscan::n_proj) +
                           "_TopM_" + int2str(sDbscan::topM) +
                           "_TopK_" + int2str(sDbscan::topK);

        outputOptics(sDbscan::ordering_, sDbscan::reachability_, sFileName);
    }
}

/**
Sng-based OPTICS: optimize speed
- We use parallel DBSCAN index to preprocess and finding neighborhoods
- If m is large, then findCoreDist_Asym is faster for multi-threading
    + findCoreDist_Asym only adds x into B(q) if dist(x, q) < eps
    + findCoreDist addx x into B(q) and q into B(x) if dist(x, q) < eps - not multi-threading friendly
**/
void sDbscan::fit_sngOptics(const Ref<const MatrixXf> & MATRIX_X, float  eps,  int  minPts)
{
    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin;

    begin = chrono::steady_clock::now();
    sDbscan::matrix_X = MATRIX_X;
    transformData(sDbscan::matrix_X, sDbscan::distance);

    if (sDbscan::verbose)
        cout << "Check X supporting distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    // Find core point
    begin = chrono::steady_clock::now();
    sDbscan::sng_findCoreDist(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formOptics_scikit();
    if (sDbscan::verbose)
        cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    if (!sDbscan::output.empty())
    {
        string sFileName = sDbscan::output + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_Prob_" + int2str(round(1000 * sDbscan::samplingProb));

        outputOptics(sDbscan::ordering_, sDbscan::reachability_, sFileName);
    }
}

/**
 *
 * @param dataset
 * @param eps
 * @param minPts
 */
void sDbscan::load_fit_sngOptics(const string& dataset, float  eps,  int  minPts)
{
    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(sDbscan::n_threads);

    chrono::steady_clock::time_point begin, start;
    begin = chrono::steady_clock::now();
    loadtxtData(dataset, sDbscan::distance, sDbscan::n_points, sDbscan::n_features, sDbscan::matrix_X);
    if (sDbscan::verbose)
        cout << "Loading data time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    start = chrono::steady_clock::now();

    // Find core point
    begin = chrono::steady_clock::now();
    sDbscan::sng_findCoreDist(eps, minPts);
    if (sDbscan::verbose)
        cout << "Find core points and distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    begin = chrono::steady_clock::now();
    formOptics_scikit();
    if (sDbscan::verbose)
        cout << "Form optics time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;

    cout << "Optics time (excluding load data) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() << "[ms]" << endl;

    if (!sDbscan::output.empty())
    {
        string sFileName = sDbscan::output + "_" + sDbscan::distance +
                           "_Eps_" + int2str(round(1000 * eps)) +
                           "_MinPts_" + int2str(minPts) +
                           "_Prob_" + int2str(round(1000 * sDbscan::samplingProb));

        outputOptics(sDbscan::ordering_, sDbscan::reachability_, sFileName);
    }
}

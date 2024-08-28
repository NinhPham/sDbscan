//
// Created by npha145 on 22/05/24.
//

#ifndef SDBSCAN_H
#define SDBSCAN_H

#include "Header.h"

class sDbscan {

private:
    int n_rotate = 3;
    int fhtDim;

protected:

    int n_points;
    int n_features;

    int n_proj = 1024; // default
    int topK = 5; // default
    int topM = 50; // default

    int n_threads = 8;

    float ker_sigma = 1.0;
    int ker_n_features = 1024;
    float ker_intervalSampling = 0.4;
    float samplingProb = 0.01; // use for sngDbscan and sDbscan-1NN

    bool verbose = false;
    string output;
    int clusterNoise = 0;
    string distance = "Cosine";
    int seed = -1;

    // Need to store it for sDbscan-1NN
    boost::dynamic_bitset<> bitHD3;


    // for Fourier embedding on L1 and L2
    // Need to store it for sDbscan-1NN with L1 and L2
    // Chi^2 and JS do not need as its embeddings are deterministic
    MatrixXf matrix_R;

    MatrixXi matrix_topK; // For each point (each col), keep topK closest/furthest random vectors
    MatrixXi matrix_topM; // For each random vector (each col), keep topM closest/furthest points

    // Data structures of Dbscan
    vector<IVector> vec2D_Neighbors; // vector of approx neighborhoods
    boost::dynamic_bitset<> bit_CorePoints; // bitarray storing core points

    // Data structures of Optics
    vector< vector< pair<int, float> > > vec2D_NeighborDist; // vector of approx neighborhoods and its distances
    FVector vec_CoreDist;

public:

    MatrixXf matrix_X; // public as we will have to load data into it (when data is big)

    // Dbscan's output
    IVector labels_;
    int n_clusters_ = 0;

    // Optics's output
    IVector ordering_;
    FVector reachability_;

    sDbscan(int n, int d){
        n_points = n;
        n_features = d;
    }

    void set_params(int numProj = 1024, int k = 5, int m = 50, string dist = "Cosine",
                   int kDim = 1024, float kSigma = 1.0, float kSam = 0.4, float prob = 0.01, int noiseClustering = 0,
                   bool ver = false, int numThreads = 8, int randomSeed = -1, string filename = ""){
        n_proj = numProj;
        topK = k;
        topM = m;
        distance = dist;
        ker_n_features = kDim;
        ker_sigma = kSigma;
        ker_intervalSampling = kSam;
        samplingProb = prob;
        clusterNoise = noiseClustering;
        verbose = ver;

        set_threads(numThreads);

        seed = randomSeed;
        output = filename;

        // Must set
        if (distance == "Cosine") {
            ker_n_features = n_features;
        }

        // have to set fhtDim
        if (distance == "Cosine")
        {
            // Must set
            ker_n_features = n_features;

            if (n_proj <= n_features)
                fhtDim = 1 << int(ceil(log2(n_features)));
            else
                fhtDim = 1 << int(ceil(log2(n_proj)));
        }
        else // the rest uses kernel embedding
        {
            if (n_proj <= ker_n_features)
                fhtDim = 1 << int(ceil(log2(ker_n_features)));
            else
                fhtDim = 1 << int(ceil(log2(n_proj)));
        }
    }

    void set_sngParams(string dist = "Cosine", float prob = 0.01, int noiseClustering = 0,
                      bool ver = false, int numThreads = -1, int randomSeed = -1, string filename = ""){

        distance = dist;
        samplingProb = prob;

        if ( noiseClustering > 0)
            clusterNoise = 2; // support sngDbscan

        verbose = ver;
        set_threads(numThreads);
        seed = randomSeed;
        output = filename;
    }

    void clear(){
        labels_.clear();
        n_clusters_ = 0;

        // Optics's output
        ordering_.clear();
        reachability_.clear();

        matrix_R.resize(0, 0);

        matrix_topK.resize(0, 0); // For each point (each col), keep topK closest/furthest random vectors
        matrix_topM.resize(0, 0); // For each random vector (each col), keep topM closest/furthest points

        // Data structures of sDbscan
        vec2D_Neighbors.clear(); // vector of approx neighborhoods
        vec2D_NeighborDist.clear(); // vector of approx neighborhoods and its distances

        bit_CorePoints.clear(); // bitarray storing core points
        vec_CoreDist.clear();
    }
    ~sDbscan(){
        matrix_X.resize(0, 0);
        clear();
    }

    void set_topM(int m){ topM = m; }
    void set_topK(int k){ topK = k; }
    void set_proj(int p){ n_proj = p; }
    void set_clusterNoise(int c){ clusterNoise = c; }
    void set_threads(int t)
    {
        if (t <= 0)
            n_threads = omp_get_max_threads();
        else
            n_threads = t;
    }

    // Dbscan
    void fit_sDbscan(const Ref<const MatrixXf> &, float, int);
    void load_fit_sDbscan(const string&, float, int );

    // Test with eps-base, and eps-range
    void test_sDbscan(const Ref<const MatrixXf> &, float, float, int);
    void load_test_sDbscan(const string&, float, float, int);

    void fit_sngDbscan(const Ref<const MatrixXf> &, float, int);

    // Test with eps-base, and eps-range
    void test_sngDbscan(const Ref<const MatrixXf> &, float, float, int);
    void load_test_sngDbscan(const string&, float, float, int);

    // Optics
    void fit_sOptics(const Ref<const MatrixXf> &, float , int);
    void load_fit_sOptics(const string&, float, int);

    void fit_sngOptics(const Ref<const MatrixXf> &, float, int);
    void load_fit_sngOptics(const string&, float, int);

private:

    // TODO: add build kernel features as a function
//    void buildKernelFeatures();
    void rp_parIndex();

protected:


    void formOptics_scikit();

    void rp_findCorePoints(float, int  );
    void rp_findCoreDist(float , int  );

    void sng_findCorePoints( float ,  int  );
    void sng_findCoreDist( float  ,  int  );

    void formCluster(); // TODO: sngDbscan: create subgraphs of approximate neighbors and connect them all together
    void labelNoise();


};


#endif // SDBSCAN_H

#include "Utilities.h"
#include "Header.h"

#include <fstream> // fscanf, fopen, ofstream
#include <sstream>

/**
 *
 * @param p_Labels
 * @param p_sOutputFile
 */
void outputDbscan(const IVector & p_Labels, const string& p_sOutputFile)
{
//	cout << "Outputing File..." << endl;
    ofstream myfile(p_sOutputFile);

    //cout << p_matKNN << endl;

    for (auto const& i : p_Labels)
    {
        myfile << i << '\n';
    }

    myfile.close();
//	cout << "Done" << endl;
}

/**
 *
 * @param p_vecOrder
 * @param p_vecDist
 * @param p_sOutputFile
 */
void outputOptics(const IVector & p_vecOrder, const FVector & p_vecDist, const string& p_sOutputFile)
{
//	cout << "Outputing File..." << endl;

    ofstream myfile(p_sOutputFile);


    for (int n = 0; n < (int)p_vecOrder.size(); ++n)
    {
        myfile << p_vecOrder[n] << " " << p_vecDist[n] << '\n';
    }

    myfile.close();
//	cout << "Done" << endl;
}

/**
 * Load data (each line is a point) into MatrixXf of size D x N format
 * Check the supporting distance and apply normalization (cosine, chi2, JS)
 *
 * @param dataset
 * @param distance
 * @param numPoints
 * @param numDim
 * @param MATRIX_X
 */
void loadtxtData(const string& dataset, const string& distance, int numPoints, int numDim, MatrixXf & MATRIX_X) {

    FILE *f = fopen(dataset.c_str(), "r");
    if (!f) {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }

    // Important: If use a temporary vector to store data, then it doubles the memory
    MATRIX_X = MatrixXf::Zero(numDim, numPoints);

    // Each line is a vector of D dimensions
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            fscanf(f, "%f", &MATRIX_X(d, n));
        }
    }

    cout << "Finish reading data" << endl;

    //        MATRIX_X.transpose();
    //        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

    /**
    Print the first col (1 x N)
    Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
    **/
    //        cout << MATRIX_X.col(0) << endl << endl;
    //        cout << "In memory (col-major):" << endl;
    //        for (n = 0; n < 10; n++)
    //            cout << *(MATRIX_X.data() + n) << "  ";
    //        cout << endl << endl;

    cout << "Now checking the condition of data given the distance." << endl;
    transformData(MATRIX_X, distance);
}

/**
 * Transform data to support the distance (only needed for Cosine, Chi2, JS)
 * @param MATRIX_X
 * @param distance
 */


/*
 * @param nargs:
 * @param args:
 * @return: Parsing parameter for FalconnPP++
 */
void readParam_sDbscan(int nargs, char** args, sDbscanParam& sParam) {

    if (nargs < 6)
    {
        cerr << "Error: Not enough parameters !" << endl;
        exit(1);
    }

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--n_points") == 0) {
            sParam.n_points = atoi(args[i + 1]);
            cout << "Number of rows/points of X: " << sParam.n_points << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        cerr << "Error: Number of rows/points is missing !" << endl;
        exit(1);
    }

    // Dimension
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--n_features") == 0) {
            sParam.n_features = atoi(args[i + 1]);
            cout << "Number of columns/dimensions: " << sParam.n_features << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess) {
        cerr << "Error: Number of columns/dimensions is missing !" << endl;
        exit(1);
    }


    // MinPTS
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--minPts") == 0) {
            sParam.minPts = atoi(args[i + 1]);
            cout << "minPts: " << sParam.minPts << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess) {
        cerr << "Error: minPts is missing !" << endl;
        exit(1);
    }

    // Eps
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--eps") == 0) {
            sParam.eps = atof(args[i + 1]);
            cout << "Radius eps: " << sParam.eps << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        cerr << "Error: Eps is missing !" << endl;
        exit(1);
    }

    // Clustering noisy points
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--clusterNoise") == 0) {
            sParam.clusterNoise = atoi(args[i + 1]);
            cout << "We cluster noisy points." << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        cout << "Default: We do not cluster the noisy points." << endl;
        sParam.clusterNoise = 0;
    }

    // Verbose
    sParam.verbose = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--verbose") == 0) {
            sParam.verbose = true;
            cout << "verbose = true." << endl;
            break;
        }
    }

    // Distance measurement
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--distance") == 0)
        {
            if (strcmp(args[i + 1], "Cosine") == 0)
            {
                sParam.distance = "Cosine";
                cout << "Cosine distance - no kernel embedding" << endl;
            }
            else if (strcmp(args[i + 1], "L1") == 0)
            {
                sParam.distance = "L1";
                cout << "L1 distance" << endl;
            }
            else if (strcmp(args[i + 1], "L2") == 0)
            {
                sParam.distance = "L2";
                cout << "L2 distance" << endl;
            }
            else if (strcmp(args[i + 1], "Chi2") == 0)
            {
                sParam.distance = "Chi2";
                cout << "Chi2 distance" << endl;
            }
            else if (strcmp(args[i + 1], "JS") == 0)
            {
                sParam.distance = "JS";
                cout << "Jensen-Shannon distance" << endl;
            }
            else
            {
                cout << "Use default cosine distance" << endl;
                sParam.distance = "Cosine";
            }

            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        cout << "Distance is missing so we use default cosine distance" << endl;
        sParam.distance = "Cosine";
    }

    // Top-K close/far random vectors
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--topK") == 0) {
            sParam.topVectors = atoi(args[i + 1]);
            cout << "TopK closest/furthest vectors: " << sParam.topVectors << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        sParam.topVectors = 5;
        cout << "TopK is missing. Use default topK: " << sParam.topVectors << endl;
    }

    // m >= MinPts
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--topM") == 0) {
            sParam.topPoints = atoi(args[i + 1]);
            cout << "TopM: " << sParam.topPoints << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        sParam.topPoints = sParam.minPts;
        cout << "TopM is missing. Use default TopM = minPts = " << sParam.topPoints << endl;
    }


    // Kernel embedding - it should be known before n_proj
    bSuccess = false;
    sParam.ker_n_features = sParam.n_features; // Must set default as n_features for Cosine
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--ker_n_features") == 0) {
            sParam.ker_n_features = atoi(args[i + 1]);
            cout << "Number of kernel embedded dimensions: " << sParam.ker_n_features << endl;
            cout << "If using L1 and L2, it must be an even number. " << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {

        if ( (sParam.distance == "L1") || (sParam.distance == "L2") )
            sParam.ker_n_features = 2 * sParam.n_features; // must be an even number
        else
            sParam.ker_n_features = sParam.n_features; // default for other distance measures

        cout << "Kernel features is missing. Use default number of kernel features: " << sParam.ker_n_features << endl;
    }

    // numProjections
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--n_proj") == 0) {
            sParam.n_proj = atoi(args[i + 1]);
            cout << "Number of projections: " << sParam.n_proj << endl;
            bSuccess = true;
            break;

        }
    }

    // Depending on the distance, we set the relevant # projections
    if (!bSuccess) {
        if (sParam.distance == "Cosine")
        {
            int iTemp = ceil(log2(1.0 * sParam.n_features));
            sParam.n_proj = max(256, 1 << iTemp);
            cout << "Number of projections is missing. Use number of projections: " << sParam.n_proj << endl;
        }
        else
        {
            int iTemp = ceil(log2(1.0 * sParam.ker_n_features));
            sParam.n_proj = max(256, 1 << iTemp);
            cout << "Number of projections is missing. Use number of projections: " << sParam.n_proj << endl;
        }
    }

    // Will be set internally in the sDbscan class
    // Identify PARAM_INTERNAL_FWHT_PROJECTION to use FWHT in case the setting is not power of 2
//    if (sParam.distance == "Cosine")   {
//        if (sParam.n_proj <= sParam.n_features)
//            sParam.fhtDim = 1 << int(ceil(log2(sParam.n_features)));
//        else
//            sParam.fhtDim = 1 << int(ceil(log2(sParam.n_proj)));
//    }
//    else // the rest uses kernel embedding
//    {
//        if (sParam.n_proj <= sParam.ker_n_features)
//            sParam.fhtDim = 1 << int(ceil(log2(sParam.ker_n_features)));
//        else
//            sParam.fhtDim = 1 << int(ceil(log2(sParam.n_proj)));
//    }

    // Scale sigma of kernel L2 and L1
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--ker_sigma") == 0)
        {
            sParam.ker_sigma = atof(args[i + 1]);
            cout << "Sigma: " << sParam.ker_sigma << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        if (sParam.distance == "L1")
        {
            sParam.ker_sigma = sParam.eps;
            cout << "Sigma is missing. Use default sigma = eps for L1: " << sParam.ker_sigma << endl;
        }
        else if (sParam.distance == "L2")
        {
            sParam.ker_sigma = 2 * sParam.eps;
            cout << "Sigma is missing. Use default sigma = 2 * eps for L2: " << sParam.ker_sigma << endl;
        }
    }

    // Sampling ratio used on Chi2 and JS - TPAMI 12 (interval_sampling in scikit-learn)
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--ker_intervalSampling") == 0)
        {
            sParam.ker_intervalSampling = atof(args[i + 1]);
            cout << "Sampling ratio for divergence distance: " << sParam.ker_intervalSampling << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess && ((sParam.distance == "Chi2") || (sParam.distance == "JS")))
    {
        sParam.ker_intervalSampling = 0.4;
        cout << "Interval sampling ratio is missing. Use default sampling ratio for Chi2 and JS distances: " << sParam.ker_intervalSampling << endl;
    }

    // Sampling ratio used on sngDbscan and sDbscan-1NN
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--samplingProb") == 0)
        {
            sParam.samplingProb = atof(args[i + 1]);
            cout << "Sampling ratio for sngDbscan or sDbscan-1NN: " << sParam.samplingProb << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        sParam.samplingProb = 0.01;
        cout << "Sampling probability is missing. Use default sampling ratio: " << sParam.samplingProb << endl;
    }

    // Output
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--output") == 0)
        {
            sParam.output = args[i + 1];
            break;
        }
    }

    if (sParam.output.empty())
    {
        cout << "No output file" << endl;
    }

    // number of threads
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--n_threads") == 0) {
            sParam.n_threads = atoi(args[i + 1]);
            cout << "Number of threads: " << sParam.n_threads << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess) {
        sParam.n_threads = -1;
        cout << "Number of threads is missing. Use all threads: " << sParam.n_threads << endl;
    }


    // Sampling ratio used on sngDbscan and sDbscan-1NN
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--random_seed") == 0)
        {
            sParam.seed = atoi(args[i + 1]);
            cout << "Seed: " << sParam.seed << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        sParam.seed = -1;
        cout << "Use a random seed: " << sParam.seed << endl;
    }

}

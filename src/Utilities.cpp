#include "Utilities.h"
#include "Header.h"

#include <fstream> // fscanf, fopen, ofstream
#include <sstream>


/**
 * WOR sampling index from vectorIndex
 * Use Fisher-Yates shuffle algorithm
 *
 * @param vectorIndex
 * @param K
 * @return random k indexes
 */
//vector<int> samplingWOR(vector<int> vectorIndex, int K)
//{
//    if ( K >= (int)vectorIndex.size() )
//        return vectorIndex;
//
//    vector<int>::iterator iterFirst, iterRandom;
//    iterFirst = vectorIndex.begin();
//    int left = vectorIndex.size() - 1;
//
//    /**
//    int i, j;
//    for (i = K - 1; i > 0; i--)
//    {
//        // Pick a random index from 0 to i
//        j = rand() % (i + 1);
//
//        // Swap arr[i] with the element
//        // at random index
//        iter_swap(vectorIndex.begin() + i, vectorIndex.begin() + j);
//    }
//
//    return vector<int>(vectorIndex.begin(), vectorIndex.begin() + K);
//    **/
//
//    while (K--)
//    {
//        //cout << *iterFirst << endl;
//        iterRandom = iterFirst;
//
//        // increment iterRandom by a random position
////        advance(iterRandom, intUnifRand(0, left - 1));
//        advance(iterRandom, rand() % left);
//
//        //cout << *iterRandom << endl;
//        // Swap value
//        swap(*iterFirst, *iterRandom);
//        //cout << *iterFirst << endl;
//        //cout << *iterRandom << endl;
//
//        // Increase the iterFirst
//        ++iterFirst;
//
//        // Decrease the size of vector
//        --left;
//    }
//
//    return vector<int>(vectorIndex.begin(), iterFirst);
//
//}

/**
 * Generate random bit for FHT
 *
 * @param p_iNumBit
 * @param bitHD
 * @param random_seed
 * return bitHD that contains fhtDim * n_rotate (default of n_rotate = 3)
 */
void bitHD3Generator(int p_iNumBit, boost::dynamic_bitset<> & bitHD, int random_seed)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed >= 0)
        seed = random_seed;

    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD = boost::dynamic_bitset<> (p_iNumBit);

    // Loop col first since we use col-wise
    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD[d] = unifDist(generator) & 1;
    }

}

/**
 * Generate Gaussian distribution N(mean, stddev)
 *
 * @param p_iNumRows
 * @param p_iNumCols
 * @param mean
 * @param stddev
 * @param random_seed
 * @return a matrix of size numRow x numCol
 */
MatrixXf gaussGenerator(int p_iNumRows, int p_iNumCols, float mean, float stddev, int random_seed)
{
    MatrixXf MATRIX_G = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed >= 0)
        seed = random_seed;

    default_random_engine generator(seed);

    normal_distribution<float> normDist(mean, stddev);

//    MATRIX_G = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    // Always iterate col first, then row later due to the col-wise storage
    for (int c = 0; c < p_iNumCols; ++c)
        for (int r = 0; r < p_iNumRows; ++r)
            MATRIX_G(r, c) = normDist(generator);

    return MATRIX_G;
}

/**
 * Generate Cauchy distribution C(x0, gamma)
 *
 * @param p_iNumRows
 * @param p_iNumCols
 * @param x0
 * @param gamma
 * @param random_seed
 * @return a matrix of size numRow x numCol
 */
MatrixXf cauchyGenerator(int p_iNumRows, int p_iNumCols, float x0, float gamma, int random_seed)
{
    MatrixXf MATRIX_C = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed >= 0)
        seed = random_seed;

    default_random_engine generator(seed);

    cauchy_distribution<float> cauchyDist(x0, gamma); // {x0 /* a */, 𝛾 /* b */}

//    MATRIX_C = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    // Always iterate col first, then row later due to the col-wise storage
    for (int c = 0; c < p_iNumCols; ++c)
        for (int r = 0; r < p_iNumRows; ++r)
            MATRIX_C(r, c) = cauchyDist(generator);

    return MATRIX_C;
}

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
 *
 * @param p_vecPoint
 * @param p_vecEmbed
 * @param kerEmbed
 * @param numDim
 * @param kerIntervalSamp
 */
void embedChi2(const Ref<VectorXf>& p_vecPoint, Ref<VectorXf> p_vecEmbed,
                    int kerEmbed, int numDim, float kerIntervalSamp)
{
    int iComponent = (kerEmbed / numDim) - 1; // kappa_1, kappa_2, ...
    iComponent /= 2; // since we take cos and sin

//    cout << "Number of components: " << iComponent << endl;

    // adding sqrt(x L kappa(0)
    for (int d = 0; d < numDim; ++d)
    {
        // Only deal with non zero
        if (p_vecPoint[d] > 0)
            p_vecEmbed[d] = sqrt(p_vecPoint[d] * kerIntervalSamp);
    }

    // adding other component
    for (int i = 1; i <= iComponent; ++i)
    {
        // We need the first D for kappa_0, 2D for kappa_1, 2D for kappa_2, ...
        int iBaseIndex = numDim + (i - 1) * 2 * numDim;

        for (int d = 0; d < numDim; ++d)
        {
            if (p_vecPoint[d] > 0)
            {
                float fFactor = sqrt(2.0 * p_vecPoint[d] * kerIntervalSamp / cosh(PI * i * kerIntervalSamp));

                p_vecEmbed[iBaseIndex + d] = fFactor * cos(i * kerIntervalSamp * log(p_vecPoint[d]));
                p_vecEmbed[iBaseIndex + numDim + d] = fFactor * sin(i * kerIntervalSamp * log(p_vecPoint[d]));
            }
        }
    }
}

/**
 *
 * @param p_vecPoint
 * @param p_vecEmbed
 * @param kerEmbed
 * @param numDim
 * @param kerIntervalSamp
 */
void embedJS(const Ref<VectorXf>& p_vecPoint, Ref<VectorXf> p_vecEmbed,
             int kerEmbed, int numDim, float kerIntervalSamp)
{
    int iComponent = (kerEmbed / numDim) - 1; // kappa_1, kappa_2, ...
    iComponent /= 2; // since we take cos and sin

    // adding sqrt(x L kappa(0)
    for (int d = 0; d < numDim; ++d)
    {
        // Only deal with non zero
        if (p_vecPoint[d] > 0)
            p_vecEmbed[d] = sqrt(p_vecPoint[d] * kerIntervalSamp * 2.0 / log(4));
    }

    // adding other component
    for (int i = 1; i <= iComponent; ++i)
    {
        // We need the first D for kappa_0, 2D for kappa_1, 2D for kappa_2, ...
        int iBaseIndex = numDim + (i - 1) * 2 * numDim;

        for (int d = 0; d < numDim; ++d)
        {
            if (p_vecPoint[d] > 0)
            {
                // this is kappa(jL)
                float fFactor = 2.0 / (log(4) * (1 + 4 * (i * kerIntervalSamp) * (i * kerIntervalSamp)) * cosh(PI * i * kerIntervalSamp));

                // This is sqrt(2X hkappa)
                fFactor = sqrt(2.0 * p_vecPoint[d] * kerIntervalSamp * fFactor);

                p_vecEmbed[iBaseIndex + d] = fFactor * cos(i * kerIntervalSamp * log(p_vecPoint[d]));
                p_vecEmbed[iBaseIndex + numDim + d] = fFactor * sin(i * kerIntervalSamp * log(p_vecPoint[d]));
            }
        }
    }
}

/** Useful for dense vector
**/
float computeDist(const Ref<VectorXf> & p_vecX, const Ref<VectorXf> & p_vecY, const string& dist)
{
    if (dist == "Cosine")
        return 1 - p_vecX.dot(p_vecY);
    if (dist == "L1")
        return (p_vecX - p_vecY).cwiseAbs().sum();
    else if (dist == "L2")
        return (p_vecX - p_vecY).norm();
    else if (dist == "Chi2") // ChiSquare
    {
        // hack for vectorize to ensure no zero element
        VectorXf vecX = p_vecX;
        VectorXf vecY = p_vecY;

        vecX.array() += EPSILON;
        vecY.array() += EPSILON;

        VectorXf temp = vecX.cwiseProduct(vecY); // x * y
        temp = temp.cwiseQuotient(vecX + vecY); // (x * y) / (x + y)
        temp.array() *= 2.0; // 2(x * y) / (x + y)

        return 1.0 - temp.sum();
    }

    else if (dist == "JS") // Jensen Shannon
    {
        // hack for vectorize
        VectorXf vecX = p_vecX;
        VectorXf vecY = p_vecY;

        vecX.array() += EPSILON;
        vecY.array() += EPSILON;

        VectorXf vecTemp1 = (vecX + vecY).cwiseQuotient(vecX); // (x + y) / x
        vecTemp1 = vecTemp1.array().log() / log(2.0); // log2( (x+y) / x))
        vecTemp1 = vecTemp1.cwiseProduct(vecX); // x * log2( (x+y) / x))

//        cout << vecTemp1.sum() / 2 << endl;

        VectorXf vecTemp2 = (vecX + vecY).cwiseQuotient(vecY);
        vecTemp2 = vecTemp2.array().log() / log(2.0);
        vecTemp2 = vecTemp2.cwiseProduct(vecY);

//        cout << vecTemp2.sum() / 2 << endl;

        return 1.0 - (vecTemp1 + vecTemp2).sum() / 2.0;
    }
    else
    {
        cout << "Error: The distance is not support" << endl;
        return 0;
    }
}

/** Faster with sparse representation
**/
float computeChi2(const Ref<VectorXf>& vecX, const Ref<VectorXf>& vecY)
{
    float dist = 0.0;
    for (int d = 0; d < vecX.size(); ++d)
    {
        if ((vecX(d) > 0) && (vecY(d) > 0))
            dist += 2 * vecX(d) * vecY(d) / (vecX(d) + vecY(d));
    }

    return 1.0 - dist;

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
void transformData(MatrixXf & MATRIX_X, const string& distance)
{
    // Check support distance
    // Doing cross-check for normalize points with cosine, and non-negative values for Chi2 and JS
    int numPoints = MATRIX_X.cols();

    if (distance == "Cosine")
    {
#pragma omp parallel for
        for (int n = 0; n < numPoints; ++n)
            MATRIX_X.col(n) /= MATRIX_X.col(n).norm(); // or MATRIX_X.colwise().normalize() inplace but not work with multi-threading

//        cout << MATRIX_X.col(0).norm() << endl;
//        cout << MATRIX_X.col(10).norm() << endl;
//        cout << MATRIX_X.col(100).norm() << endl;
    }
    else if ((distance == "Chi2") || (distance == "JS"))
    {
        // Ensure non-negative
        if (MATRIX_X.minCoeff() < 0)
        {
            cerr << "Error: X is not non-negative !" << endl;
            exit(1);
        }
        else // normalize to have sum = 1
        {
            // Get colwise.sum is a row array, need to transpose() to make it col array
#pragma omp parallel for
            for (int n = 0; n < numPoints; ++n)
            {
                float fSum = MATRIX_X.col(n).sum();
                if (fSum <= 0)
                {
                    cerr << "Error: There is an zero point !" << endl;
                    exit(1);
                }
                MATRIX_X.col(n) /= fSum;
            }

            // Test
//            cout << MATRIX_X.col(0).sum() << endl;
//            cout << MATRIX_X.col(10).sum() << endl;
//            cout << MATRIX_X.col(100).sum() << endl;
        }
    }

}

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
            sParam.topK = atoi(args[i + 1]);
            cout << "TopK closest/furthest vectors: " << sParam.topK << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        sParam.topK = 5;
        cout << "TopK is missing. Use default topK: " << sParam.topK << endl;
    }

    // m >= MinPts
    bSuccess = false;
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--topM") == 0) {
            sParam.topM = atoi(args[i + 1]);
            cout << "TopM: " << sParam.topM << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess) {
        sParam.topM = sParam.minPts;
        cout << "TopM is missing. Use default TopM = minPts = " << sParam.topM << endl;
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

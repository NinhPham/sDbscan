#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "Header.h"

#include <sstream> // stringstream
#include <time.h> // for time(0) to generate different random number

/**
Convert an integer to string
**/
inline string int2str(int x)
{
    stringstream ss;
    ss << x;
    return ss.str();
}

// https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c
inline float fast_log2 (float val)
{
    int * const    exp_ptr = reinterpret_cast <int *> (&val);
    int            x = *exp_ptr;
    const int      log_2 = ((x >> 23) & 255) - 128;
    x &= ~(255 << 23);
    x += 127 << 23;
    *exp_ptr = x;

    val = ((-1.0f/3) * val + 2) * val - 2.0f/3;   // (1)

    return (val + log_2);
}

inline float fast_log (const float &val)
{
    return (fast_log2 (val) * 0.69314718f);
}

/**
 * Generate random bit for FHT
 *
 * @param p_iNumBit
 * @param bitHD
 * @param random_seed
 * return bitHD that contains fhtDim * n_rotate (default of n_rotate = 3)
 */
inline void bitHD3Generator(int p_iNumBit, int random_seed, boost::dynamic_bitset<> & bitHD)
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
inline MatrixXf gaussGenerator(int p_iNumRows, int p_iNumCols, float mean, float stddev, int random_seed)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed >= 0)
        seed = random_seed;

    default_random_engine generator(seed);
    normal_distribution<float> normDist(mean, stddev);

    MatrixXf MATRIX_G = MatrixXf::Zero(p_iNumRows, p_iNumCols);

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
inline MatrixXf cauchyGenerator(int p_iNumRows, int p_iNumCols, float x0, float gamma, int random_seed)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    if (random_seed >= 0)
        seed = random_seed;

    default_random_engine generator(seed);
    cauchy_distribution<float> cauchyDist(x0, gamma); // {x0 /* a */, 𝛾 /* b */}

    MatrixXf MATRIX_C = MatrixXf::Zero(p_iNumRows, p_iNumCols);

    // Always iterate col first, then row later due to the col-wise storage
    for (int c = 0; c < p_iNumCols; ++c)
        for (int r = 0; r < p_iNumRows; ++r)
            MATRIX_C(r, c) = cauchyDist(generator);

    return MATRIX_C;
}

/**
 *
 * @param p_vecPoint
 * @param p_vecEmbed
 * @param kerEmbed
 * @param numDim
 * @param kerIntervalSamp
 */
inline void embedChi2(const Ref<VectorXf>& p_vecPoint, int kerEmbed, int numDim, float kerIntervalSamp, Ref<VectorXf> p_vecEmbed)
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
inline void embedJS(const Ref<VectorXf>& p_vecPoint, int kerEmbed, int numDim, float kerIntervalSamp, Ref<VectorXf> p_vecEmbed)
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
inline float computeDist(const Ref<VectorXf> & p_vecX, const Ref<VectorXf> & p_vecY, const string& dist)
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
//        VectorXf vecX = p_vecX;
//        VectorXf vecY = p_vecY;
//
//        vecX.array() += EPSILON;
//        vecY.array() += EPSILON;
//
//        VectorXf temp = vecX.cwiseProduct(vecY); // x * y
//        temp = temp.cwiseQuotient(vecX + vecY); // (x * y) / (x + y)
//        temp.array() *= 2.0; // 2(x * y) / (x + y)
//
//        return 1.0 - temp.sum();

        float temp = 0.0;
        for (int d = 0; d < p_vecX.size(); ++d)
        {
            if ((p_vecX(d) > 0) && (p_vecY(d) > 0))
                temp += p_vecX(d) * p_vecY(d) / (p_vecX(d) + p_vecY(d));
        }

        return 1.0 - 2.0 * temp;
    }

    else if (dist == "JS") // Jensen Shannon
    {
        // hack for vectorize
//        VectorXf vecX = p_vecX;
//        VectorXf vecY = p_vecY;
//
//        vecX.array() += EPSILON;
//        vecY.array() += EPSILON;
//
//        VectorXf vecTemp1 = (vecX + vecY).cwiseQuotient(vecX); // (x + y) / x
//        vecTemp1 = vecTemp1.array().log() / log(2.0); // log2( (x+y) / x))
//        vecTemp1 = vecTemp1.cwiseProduct(vecX); // x * log2( (x+y) / x))
//
////        cout << vecTemp1.sum() / 2 << endl;
//
//        VectorXf vecTemp2 = (vecX + vecY).cwiseQuotient(vecY); // (x + y) / y
//        vecTemp2 = vecTemp2.array().log() / log(2.0); // log2( (x+y) / y))
//        vecTemp2 = vecTemp2.cwiseProduct(vecY); // y * log2( (x+y) / y))
//
////        cout << vecTemp2.sum() / 2 << endl;
//
//        return 1.0 - (vecTemp1 + vecTemp2).sum() / 2.0;

        // TODO: Remove EPSILON for JS. It might affect the accuracy since the value of eps will change
        float temp = 0.0;
        for (int d = 0; d < p_vecX.size(); ++d)
        {
//            if ((p_vecX(d) > 0) && (p_vecY(d) > 0))
//            {
                temp += (p_vecX(d) + p_vecY(d) + 2 * EPSILON) * fast_log2(p_vecX(d) + p_vecY(d) + 2 * EPSILON)
                        -  (p_vecX(d) + EPSILON) * fast_log2(p_vecX(d) + EPSILON)
                        -  (p_vecY(d) + EPSILON) * fast_log2(p_vecY(d) + EPSILON);
//            }
        }

        return 1.0 - temp / 2.0;
    }
    else
    {
        cout << "Error: The distance is not support" << endl;
        return 0;
    }
}

inline void transformData(MatrixXf & MATRIX_X, const string& distance)
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

/**
    Convert a = a + b, b = a - b
**/
//void inline wht_bfly (float& a, float& b)
//{
//    float tmp = a;
//    a += b;
//    b = tmp - b;
//}

/**
    Fast in-place Walsh-Hadamard Transform (http://www.musicdsp.org/showone.php?id=18)
    also see (http://stackoverflow.com/questions/22733444/fast-sequency-ordered-walsh-hadamard-transform/22752430#22752430)
    - Note that the running time is exactly NlogN

    Or: https://github.com/vegarant/fastwht?tab=readme-ov-file
**/
//void inline FWHT (Ref<VectorXf> data)
//{
//    int n = (int)data.size();
//    int nlog2 = log2(n);
//
//    int l, m;
//    for (int i = 0; i < nlog2; ++i)
//    {
//        l = 1 << (i + 1);
//        for (int j = 0; j < n; j += l)
//        {
//            m = 1 << i;
//            for (int k = 0; k < m; ++k)
//            {
//                //cout << data (j + k) << endl;
//                data (j + k) = data (j + k);
//                //cout << data (j + k) << endl;
//
//                //cout << data (j + k + m) << endl;
//                data (j + k + m) = data (j + k + m);
//                //cout << data (j + k + m) << endl;
//
//                wht_bfly (data (j + k), data (j + k + m));
//                //cout << data (j + k) << endl;
//                //cout << data (j + k + m) << endl;
//
//            }
//
//        }
//    }
//}

// Saving
void outputDbscan(const IVector &, const string&);
void outputOptics(const IVector &, const FVector &, const string&);

// Parsing input and param
void readParam_sDbscan(int , char**, sDbscanParam & );
void loadtxtData(const string&, const string&, int , int, MatrixXf & ); // load data from filename

#endif // UTILITIES_H_INCLUDED

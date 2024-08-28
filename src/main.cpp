#include <iostream>
#include "Header.h"
#include "Utilities.h"

#include "sDbscan.h"

#include <time.h> // for time(0) to generate different random number
#include <stdlib.h>
#include <sys/time.h> // for gettimeofday
#include <stdio.h>
#include <unistd.h>

#include <omp.h>

int main(int nargs, char** args)
{
    srand(time(NULL)); // should only be called once for random generator

//    cout << "RAM before loading data" << endl;
//    getRAM();

    /************************************************************************/
//	int iType = loadInput(nargs, args);

    sDbscanParam sParam;
    readParam_sDbscan(nargs, args, sParam);

    sDbscan dbscan(sParam.n_points, sParam.n_features);
    dbscan.set_params(sParam.n_proj, sParam.topK, sParam.topM, sParam.distance, sParam.ker_n_features,
                     sParam.ker_sigma, sParam.ker_intervalSampling, sParam.samplingProb,
                     sParam.clusterNoise, sParam.verbose, sParam.n_threads, sParam.seed, sParam.output);

    // Read data
    string dataset = "";
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--dataset") == 0) {
            dataset = args[i + 1]; // convert char* to string
            break;
        }
    }
    if (dataset == "") {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }

    MatrixXf MATRIX_X;
    loadtxtData(dataset, sParam.distance, sParam.n_points, sParam.n_features, MATRIX_X);

    // Saving memory
//    loadtxtData(dataset, sParam.distance, sParam.n_points, sParam.n_features, dbscan.matrix_X);

    chrono::steady_clock::time_point begin, end;
    begin = chrono::steady_clock::now();
    float fRangeEps = 0.01;

//    dbscan.test_sDbscan(MATRIX_X, sParam.eps, fRangeEps, sParam.minPts);

    for (int i = 0; i < 10; ++i)
    {
        float new_eps = sParam.eps + i * fRangeEps;

        dbscan.clear();
        cout << "------------------" << endl;
        begin = chrono::steady_clock::now();
        dbscan.fit_sDbscan(MATRIX_X, new_eps, sParam.minPts); // need to reset baseEps
        end = chrono::steady_clock::now();
        cout << "sDBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;


        dbscan.clear();
        cout << "------------------" << endl;
        begin = chrono::steady_clock::now();
        dbscan.fit_sngDbscan(MATRIX_X, new_eps, sParam.minPts);
        end = chrono::steady_clock::now();
        cout << "sngDBSCAN Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;
    }

    // For big data
    float eps = 0.25;
    int minPts = 50;

    dbscan.load_fit_sDbscan(dataset, eps, minPts);
    dbscan.load_fit_sOptics(dataset, eps, minPts);





}


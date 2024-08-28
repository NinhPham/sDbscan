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

//vector<int> samplingWOR(vector<int>, int);

float computeDist(const Ref<VectorXf> &, const Ref<VectorXf> &, const string& );
float computeChi2(const Ref<VectorXf>&, const Ref<VectorXf>&);

void embedChi2(const Ref<VectorXf> &, Ref<VectorXf>, int, int, float );
void embedJS(const Ref<VectorXf>&, Ref<VectorXf>, int , int , float );

/* Generate Hadamard matrix
*/
void bitHD3Generator(int, boost::dynamic_bitset<> &, int);
MatrixXf gaussGenerator(int, int, float, float, int);
MatrixXf cauchyGenerator(int, int, float, float, int);

// Saving
void outputDbscan(const IVector &, const string&);
void outputOptics(const IVector &, const FVector &, const string&);

// Parsing input and param
void readParam_sDbscan(int , char**, sDbscanParam & );
void loadtxtData(const string&, const string&, int , int, MatrixXf & ); // load data from filename
void transformData(MatrixXf & , const string& );

#endif // UTILITIES_H_INCLUDED

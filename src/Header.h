#pragma once

#include "fht.h"

#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <random>

#include <chrono>
#include <iostream> // cin, cout

//#include <boost/multi_array.hpp>
#include <boost/dynamic_bitset.hpp>

#define PI				3.141592653589793238460
#define NEG_INF        -2147483648 // -2^32
#define POS_INF        2147483647 // 2^31-1
#define EPSILON         0.000001 // 2^31-1

using namespace Eigen;
using namespace std;

typedef vector<float> FVector;
typedef vector<int> IVector;

//typedef vector<uint32_t> I32Vector;
//typedef vector<uint64_t> I64Vector;


//typedef boost::multi_array<int, 3> IVector3D;

//struct myComp
//{
//
//    constexpr bool operator()(
//        pair<double, int> const& a,
//        pair<double, int> const& b)
//    const noexcept
//    {
//        return a.first > b.first;
//    }
//};

struct IFPair
{
    int m_iIndex;
    float	m_fValue;

    IFPair()
    {
        m_iIndex = 0;
        m_fValue = 0.0;
    }

    IFPair(int p_iIndex, double p_fValue)
    {
        m_iIndex = p_iIndex;
        m_fValue = p_fValue;
    }

    // Overwrite operation < to get top K largest entries
    bool operator<(const IFPair& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IFPair& p) const
    {
        return m_fValue > p.m_fValue;
    }
};

struct sDbscanParam
{
    int n_points;
    int n_features;
    float eps;
    int minPts;
    int n_proj;
    int n_threads;
    int topM;
    int topK;
    float ker_sigma;
    int ker_n_features;
    float ker_intervalSampling;
    float samplingProb;
    bool verbose;
    string output;
    int clusterNoise; // 0 is no, 1 is CEOs, 2 is sngDbscan

//    int fhtDim; // will be set in the sDbscan class
    string distance;
    int seed;
};

typedef priority_queue<IFPair, vector<IFPair>, greater<IFPair>> Min_PQ_Pair;



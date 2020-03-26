#ifndef LIBRARY_UTILITIES_H
#define LIBRARY_UTILITIES_H

#include <assert.h>
#include <limits>
#include <vector>

#include "tbb/tbb.h"

///////////////////////////////////
//
// Borrowed from Robert Bridson's
// sample code.
//
////////////////////////////////////

namespace FluidSim2D::Utilities
{

constexpr double PI = 3.1415926535897932384626433832795;

//
// TBB utilities
//

constexpr int tbbLightGrainSize = 1000;
constexpr int tbbHeavyGrainSize = 100;

template<typename StorageType>
void mergeLocalThreadVectors(std::vector<StorageType>& combinedVector,
	tbb::enumerable_thread_specific<std::vector<StorageType>>& parallelVector)
{
	int vectorSize = 0;

	parallelVector.combine_each([&](const std::vector<StorageType>& localVector)
	{
		vectorSize += localVector.size();
	});

	combinedVector.reserve(combinedVector.size() + vectorSize);

	parallelVector.combine_each([&](const std::vector<StorageType>& localVector)
	{
		combinedVector.insert(combinedVector.end(), localVector.begin(), localVector.end());
	});
}

//
// BFS markers
//

enum class VisitedCellLabels { UNVISITED_CELL, VISITED_CELL, FINISHED_CELL };

///////////////////////////////////
//
// Borrowed and modified from
// Robert Bridson's sample code.
//
////////////////////////////////////

#ifdef _MSC_VER
#undef min
#undef max
#endif

using std::min;
using std::max;
using std::swap;

template<typename T>
T sqr(T x)
{
	return x * x;
}

template<typename T>
T cube(T x)
{
	return x * sqr(x);
}

template<typename... Args>
decltype(auto) min(const Args&... values)
{
	return std::min({ values... });
}

template<typename... Args>
decltype(auto) max(const Args&... values)
{
	return std::max({ values... });
}

template<typename T>
void minAndMax(T& minValue, T& maxValue, const T& value0, const T& value1)
{
	minValue = std::min(value0, value1);
	maxValue = std::max(value0, value1);
}

template<typename T>
void updateMinOrMax(T& minValue, T& maxValue, const T& value)
{
	if (value < minValue) minValue = value;
	else if (value > maxValue) maxValue = value;
}

template<typename T>
void updateMinAndMax(T& minValue, T& maxValue, const T& value)
{
	if (value < minValue) minValue = value;
	if (value > maxValue) maxValue = value;
}

template<typename T>
T clamp(const T& value, const T& lower, const T& upper)
{
	if (value < lower) return lower;
	else if (value > upper) return upper;
	else return value;
}

// Transforms even the sequence 0,1,2,3,... into reasonably good random numbers 
// Challenge: improve on this in speed and "randomness"!
// This seems to pass several statistical tests, and is a bijective map (of 32-bit unsigneds)
inline unsigned randhash(unsigned seed)
{
	unsigned i = (seed ^ 0xA3C59AC3u) * 2654435769u;
	i ^= (i >> 16);
	i *= 2654435769u;
	i ^= (i >> 16);
	i *= 2654435769u;
	return i;
}

// the inverse of randhash
inline unsigned unhash(unsigned h)
{
	h *= 340573321u;
	h ^= (h >> 16);
	h *= 340573321u;
	h ^= (h >> 16);
	h *= 340573321u;
	h ^= 0xA3C59AC3u;
	return h;
}

constexpr unsigned maxInteger = std::numeric_limits<unsigned>::max();

// returns repeatable stateless pseudo-random number in [0,1]
inline double randhashd(unsigned seed)
{
	return randhash(seed) / double(maxInteger);
}

inline float randhashf(unsigned seed)
{
	return randhash(seed) / float(maxInteger);
}

// returns repeatable stateless pseudo-random number in [a,b]
inline double randhashd(unsigned seed, double a, double b)
{
	return (b - a) * randhash(seed) / double(maxInteger) + a;
}

inline float randhashf(unsigned seed, float a, float b)
{
	return (b - a) * randhash(seed) / float(maxInteger) + a;
}

template<typename S, typename T>
S lerp(const S& value0, const S& value1, const T& f)
{
	return (1. - f) * value0 + f * value1;
}

template<typename S, typename T>
S bilerp(const S& value00, const S& value10,
	const S& value01, const S& value11,
	const T& fx, const T& fy)
{
	return lerp(lerp(value00, value10, fx),
		lerp(value01, value11, fx),
		fy);
}

template<typename S, typename T>
S trilerp(const S& value000, const S& value100,
	const S& value010, const S& value110,
	const S& value001, const S& value101,
	const S& value011, const S& value111,
	const T& fx, const T& fy, const T& fz)
{
	return lerp(bilerp(value000, value100, value010, value110, fx, fy),
		bilerp(value001, value101, value011, value111, fx, fy),
		fz);
}

// Catmull-Rom cubic interpolation (see https://en.wikipedia.org/wiki/Cubic_Hermite_spline).
template<typename S, typename T>
S cubicInterp(const S& value_1, const S& value0, const S& value1, const S& value2, const T& fx)
{
	T sqrfx = sqr(fx), cubefx = cube(fx);
	return T(0.5) * ((-cubefx + T(2.0) * sqrfx - fx) * value_1
		+ (T(3.0) * cubefx - T(5.0) * sqrfx + T(2.0)) * value0
		+ (-T(3.0) * cubefx + T(4.0) * sqrfx + fx) * value1
		+ (cubefx - sqrfx) * value2);
};

}
#endif
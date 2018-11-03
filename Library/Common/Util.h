#pragma once

#include <limits>
#include <algorithm>

///////////////////////////////////
//
// Borrowed from Robert Bridson's
// sample code.
//
////////////////////////////////////

#ifdef _MSC_VER
	#undef min
	#undef max
#endif

constexpr unsigned intmax = std::numeric_limits<unsigned>::max();

#ifndef M_PI
	constexpr double M_PI = 3.1415926535897932384626433832795;
#endif

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
decltype(auto) min(const Args&... args)
{
	return std::min({ args... });
}

template<typename... Args>
decltype(auto) max(const Args&... args)
{
	return std::max({ args... });
}

template<typename T>
void minmax(T& minval, T& maxval, const T& val0, const T& val1)
{
	minval = std::min(val0, val1);
	maxval = std::max(val0, val1);
}

template<typename T, typename... Args>
void minmax(T& minval, T& maxval, const Args... args)
{
	minval = std::min({ args... });
	maxval = std::max({ args... });
}

template<typename T>
void update_min_max(T& minval, T& maxval, const T& val)
{
	if (val < minval) minval = val;
	else if (val > maxval) maxval = val;
}

template<typename T>
void update_both_minmax(T& minval, T& maxval, const T& val)
{
	if (val < minval) minval = val;
	if (val > maxval) maxval = val;
}

template<typename T, typename S>
T clamp(const T& val, const S& lower, const S& upper)
{
	T templo = static_cast<T>(lower);
	T tempup = static_cast<T>(upper);

	if (val < templo) return templo;
	else if(val > tempup) return tempup;
	else return val;
}

// Transforms even the sequence 0,1,2,3,... into reasonably good random numbers 
// Challenge: improve on this in speed and "randomness"!
// This seems to pass several statistical tests, and is a bijective map (of 32-bit unsigneds)
inline unsigned randhash(unsigned seed)
{
	unsigned i=(seed^0xA3C59AC3u)*2654435769u;
	i^=(i>>16);
	i*=2654435769u;
	i^=(i>>16);
	i*=2654435769u;
	return i;
}

// the inverse of randhash
inline unsigned unhash(unsigned h)
{
	h*=340573321u;
	h^=(h>>16);
	h*=340573321u;
	h^=(h>>16);
	h*=340573321u;
	h^=0xA3C59AC3u;
	return h;
}

// returns repeatable stateless pseudo-random number in [0,1]
inline double randhashd(unsigned seed)
{ return randhash(seed)/(double)intmax; }

inline float randhashf(unsigned seed)
{ return randhash(seed)/(float)intmax; }

// returns repeatable stateless pseudo-random number in [a,b]
inline double randhashd(unsigned seed, double a, double b)
{ return (b-a)*randhash(seed)/(double)intmax + a; }

inline float randhashf(unsigned seed, float a, float b)
{ return ( (b-a)*randhash(seed)/(float)intmax + a); }

template<typename S, typename T>
S lerp(const S& val0, const S& val1, T f)
{
	return (1 - f) * val0 + f * val1;
}

template<typename S, typename T>
S bilerp(const S& val00, const S& val10,
                const S& val01, const S& val11,
                T fx, T fy)
{ 
	return lerp(lerp(val00, val10, fx),
				lerp(val01, val11, fx),
				fy);
}

template<typename S, typename T>
S trilerp(const S& val000, const S& val100,
                 const S& val010, const S& val110,
                 const S& val001, const S& val101,  
                 const S& val011, const S& val111,
                 T fx, T fy, T fz) 
{
	return lerp(bilerp(val000, val100, val010, val110, fx, fy),
				bilerp(val001, val101, val011, val111, fx, fy),
				fz);
}

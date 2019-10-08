#ifndef LIBRARY_VEC_H
#define LIBRARY_VEC_H

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "Util.h"

///////////////////////////////////
//
// Borrowed from Robert Bridson's
// sample code.
//
////////////////////////////////////

template<typename T, unsigned N>
struct Vec
{
	Vec<T, N>()	{}

	explicit Vec<T, N>(T val)
	{
		myVec.fill(val);
	}

	// TODO: Figure out a variadic method that actually works!
	constexpr Vec<T, N>(T v0, T v1) : myVec({v0, v1})
	{
		static_assert(N == 2, "Vector dimensions must match!");
	}
	constexpr Vec<T, N>(T v0, T v1, T v2) : myVec({ v0, v1, v2 })
	{
		static_assert(N == 3, "Vector dimensions must match!");
	}
	constexpr Vec<T, N>(T v0, T v1, T v2, T v3) : myVec({ v0, v1, v2, v3 })
	{
		static_assert(N == 4, "Vector dimensions must match!");
	}
	constexpr Vec<T, N>(T v0, T v1, T v2, T v3, T v4) : myVec({ v0, v1, v2, v3, v4 })
	{
		static_assert(N == 5, "Vector dimensions must match!");
	}

	Vec<T, N>(const Vec<T, N>&) = default;
	Vec<T, N>& operator=(const Vec<T, N>&) = default;
	Vec<T, N>(Vec<T, N>&&) = default;
	Vec<T, N>& operator=(Vec<T, N>&&) = default;

	~Vec<T, N>() = default;

	template <typename S>
	explicit Vec<T, N>(const Vec<S, N>& source)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] = static_cast<T>(source[i]);
	}

	T &operator[](unsigned index)
	{
		return myVec[index];
	}

	const T &operator[](unsigned index) const
	{
		return myVec[index];
	}

	decltype(auto) data() { return myVec.data(); }

	Vec<T, N> operator+=(const Vec<T, N>& rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] += rhs[i];
		return *this;
	}

	Vec<T, N> operator-=(const Vec<T, N>& rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] -= rhs[i];
		return *this;
	}

	Vec<T, N> operator*=(const Vec<T, N>& rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] *= rhs[i];
		return *this;
	}

	Vec<T, N> operator/=(const Vec<T, N>& rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] /= rhs[i];
		return *this;
	}

	Vec<T, N> operator+(Vec<T, N> rhs) const
	{
		rhs += *this;
		return rhs;
	}

	Vec<T, N> operator-(const Vec<T, N>& rhs) const
	{
		Vec<T, N> temp(*this);
		temp -= rhs;
		return temp;
	}

	Vec<T, N> operator*(Vec<T, N> rhs) const
	{
		rhs *= *this;
		return rhs;
	}

	Vec<T, N> operator/(const Vec<T, N>& rhs) const
	{
		Vec<T, N> temp(*this);
		temp /= rhs;
		return temp;
	}

	Vec<T, N> operator-() const // unary minus
	{
		Vec<T, N> vec;

		for (unsigned i = 0; i < N; ++i)
			vec[i] = -myVec[i];

		return vec;
	}

	Vec<T, N> operator+=(T rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] += rhs;

		return *this;
	}

	Vec<T, N> operator-=(T rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] -= rhs;

		return *this;
	}

	Vec<T, N> operator*=(T rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] *= rhs;

		return *this;
	}

	Vec<T, N> operator/=(T rhs)
	{
		for (unsigned i = 0; i < N; ++i)
			myVec[i] /= rhs;

		return *this;
	}

	Vec<T, N> operator*(T rhs) const
	{
		Vec<T, N> vec(*this);
		vec *= rhs;
		return vec;
	}

	Vec<T, N> operator+(T rhs) const
	{
		Vec<T, N> vec(*this);
		vec += rhs;
		return vec;
	}
	
	Vec<T, N> operator/(T rhs) const
	{
		Vec<T, N> vec(*this);
		vec /= rhs;
		return vec;
	}

private:
	std::array<T, N> myVec;
};

using Vec2d = Vec<double, 2>;
using Vec2f = Vec<float, 2>;
using Vec2i = Vec<int, 2>;
using Vec2ui = Vec<unsigned, 2>;
using Vec2st = Vec<size_t, 2>;

using Vec3d = Vec<double, 3>;
using Vec3f = Vec<float, 3>;
using Vec3i = Vec<int, 3>;
using Vec3ui = Vec<unsigned, 3>;
using Vec3st = Vec<size_t, 3>;

using Vec4d = Vec<double, 4>;
using Vec4f = Vec<float, 4>;
using Vec4i = Vec<int, 4>;
using Vec4ui = Vec<unsigned, 4>;
using Vec4st = Vec<size_t, 4>;

using Vec5d = Vec<double, 5>;
using Vec5f = Vec<float, 5>;
using Vec5i = Vec<int, 5>;
using Vec5ui = Vec<unsigned, 5>;
using Vec5st = Vec<size_t, 5>;

using Vec6d = Vec<double, 6>;
using Vec6f = Vec<float, 6>;
using Vec6i = Vec<int, 6>;
using Vec6ui = Vec<unsigned, 6>;
using Vec6st = Vec<size_t, 6>;

template<typename T, unsigned N>
T mag2(const Vec<T, N>& vec)
{
	T val = 0;

	for (unsigned i = 0; i < N; ++i)
		val += Util::sqr(vec[i]);

	return val;
}

template<typename T, unsigned N>
T mag(const Vec<T, N>& vec)
{
	return sqrt(mag2(vec));
}

template<typename T, unsigned N>
T dist2(const Vec<T, N>& vec0, const Vec<T, N>& vec1)
{
	T val = 0;
	
	for (unsigned i = 0; i < N; ++i)
		val += Util::sqr(vec0[i] - vec1[i]);
	
	return val;
}

template<typename T, unsigned N>
T dist(const Vec<T, N>& vec0, const Vec<T, N>& vec1)
{
	return std::sqrt(dist2(vec0, vec1));
}

template<typename T, unsigned N>
void normalizeInPlace(Vec<T, N> &vec)
{
	vec /= mag(vec);
}

template<typename T, unsigned N>
inline Vec<T, N> normalize(Vec<T, N> vec)
{
	normalizeInPlace(vec);
	return vec;
}

template<typename T, unsigned N>
T infNorm(const Vec<T, N>& vec)
{
	T val = 0;

	for (unsigned i = 0; i < N; ++i)
		val = max(std::fabs(vec[i]), val);

	return val;
}

template<typename T, unsigned N>
std::ostream& operator<<(std::ostream& out, const Vec<T, N>& vec)
{
	out << vec[0];

	for (unsigned i = 1; i < N; ++i)
		out << ' ' << vec[i];

	return out;
}

template<typename T, unsigned N>
std::istream& operator>>(std::istream& in, Vec<T, N>& vec)
{
	for (unsigned i = 0; i < N; ++i)
		in >> vec[i];

	return in;
}

template<typename T, unsigned N>
bool operator==(const Vec<T, N>& vec0, const Vec<T, N>& vec1)
{ 
	for (unsigned i = 0; i < N; ++i)
		if (vec0[i] != vec1[i]) return false;

	return true;
}

template<typename T, unsigned N>
bool operator!=(const Vec<T, N>& vec0, const Vec<T, N>& vec1)
{ 
	return !(vec0 == vec1);
}

template<typename S, typename T, unsigned N>
Vec<T, N> operator*(S val, Vec<T, N> vec)
{
	vec *= static_cast<T>(val);
	return vec;
}

template<typename T, unsigned N>
T min(const Vec<T, N>& vec)
{
	T val = vec[0];

	for (unsigned i = 1; i < N; ++i)
		if (vec[i] < val) val = vec[i];

	return val;
}

// The second vector is passed by value because we would have
// to allocate memory anyway.

template<typename T, unsigned N>
Vec<T, N> minUnion(const Vec<T, N>& vec0, Vec<T, N> vec1)
{
	for (unsigned i = 0; i < N; ++i)
		vec1[i] = (vec0[i] < vec1[i]) ? vec0[i] : vec1[i];
   
	return vec1;
}

template<typename T, unsigned N>
inline Vec<T, N> maxUnion(const Vec<T, N>& vec0, Vec<T, N> vec1)
{
	for (unsigned i = 0; i < N; ++i)
		vec1[i] = (vec0[i] > vec1[i]) ? vec0[i] : vec1[i];

	return vec1;
}

template<typename T, unsigned N>
T max(const Vec<T, N>& vec)
{
	T val = vec[0];
   
	for (unsigned i = 1; i < N; ++i)
		if (vec[i] > val) val = vec[i];
	
	return val;
}

template<typename T, unsigned N>
T dot(const Vec<T, N>& vec0, const Vec<T, N>& vec1)
{
	T val = 0;

	for (unsigned i = 0; i < N; ++i)
		val += vec0[i] * vec1[i];

	return val;
}

template<typename T>
T cross(const Vec<T, 2>& vec0, const Vec<T, 2>& vec1)
{
	return vec0[0] * vec1[1] - vec0[1] * vec1[0];
}

template<typename T>
Vec<T, 3> cross(const Vec<T, 3>& vec0, const Vec<T, 3>& vec1)
{
	return Vec<T, 3>(vec0[1] * vec1[2] - vec0[2] * vec1[1],
						vec0[2] * vec1[0] - vec0[0] * vec1[2],
						vec0[0] * vec1[1] - vec0[1] * vec1[0]);
}

template<typename T, unsigned N>
Vec<T, N> round(Vec<T, N> vec)
{ 
	for (unsigned i = 0; i < N; ++i)
		vec[i] = std::round(vec[i]);

	return vec;
}

template<typename T, unsigned N>
Vec<T, N> floor(Vec<T, N> vec)
{ 
	for (unsigned i = 0; i < N; ++i)
		vec[i] = std::floor(vec[i]);

	return vec;
}

template<typename T, unsigned N>
Vec<T, N> ceil(Vec<T, N> vec)
{ 
	for (unsigned i = 0; i < N; ++i)
		vec[i] = std::ceil(vec[i]);

	return vec;
}

template<typename T, unsigned N>
Vec<T, N> fabs(Vec<T, N> vec)
{ 
   for (unsigned i = 0; i < N; ++i)
	   vec[i] = std::fabs(vec[i]);

   return vec;
}

template<typename T, unsigned N>
Vec<T, N> clamp(Vec<T, N> vec, const Vec<T, N>& low, const Vec<T, N>& high)
{
	for (unsigned i = 0; i < N; ++i)
		vec[i] = Util::clamp(vec[i], low[i], high[i]);

	return vec;
}

template<typename T, unsigned N>
Vec<T, N> min(const Vec<T, N> &min0, Vec<T, N> min1)
{
	for (unsigned i = 0; i < N; ++i)
		min1[i] = std::min(min0[i], min1[i]);

	return min1;
}

template<typename T, unsigned N>
Vec<T, N> max(const Vec<T, N> &max0, Vec<T, N> max1)
{
	for (unsigned i = 0; i < N; ++i)
		max1[i] = std::max(max0[i], max1[i]);

	return max1;
}

template<typename T, unsigned N>
void minAndMax(Vec<T, N>& vecMin, Vec<T, N>& vecMax, const Vec<T, N>& vec0, const Vec<T, N>& vec1)
{
	for (unsigned i = 0; i < N; ++i)
		Util::minAndMax(vecMin[i], vecMax[i], vec0[i], vec1[i]);
}

template<typename T, unsigned N, typename ...Args>
void minAndMax(Vec<T, N>& vecMin, Vec<T, N>& vecMax, const Args&... vecs)
{
	minAndMax(vecMin, vecMax, vecs...);
}

template<typename T, unsigned N>
void updateMinOrMax(Vec<T, N> &vecMin, Vec<T, N> &vecMax, const Vec<T, N> &vec)
{
	for (unsigned i = 0; i < N; ++i) updateMinOrMax(vec[i], vecMin[i], vecMax[i]);
}

template<typename T, unsigned N>
void updateMinAndMax(Vec<T, N> &vecMin, Vec<T, N> &vecMax, const Vec<T, N> &vec)
{
	for (unsigned i = 0; i < N; ++i) Util::updateMinAndMax(vecMin[i], vecMax[i], vec[i]);
}

#endif
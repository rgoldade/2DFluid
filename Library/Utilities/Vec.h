#ifndef LIBRARY_VEC_H
#define LIBRARY_VEC_H

#include <array>
#include <cmath>

#include "Utilities.h"

///////////////////////////////////
//
// Borrowed from Robert Bridson's
// sample code.
//
////////////////////////////////////

namespace FluidSim2D::Utilities
{

template<int N, typename T>
class Vec
{

public:

	Vec<N, T>()
	{
		static_assert(N >= 0, "Vector dimensions must be positive");
	}

	constexpr explicit Vec<N, T>(T val)
	{
		myVec.fill(val);
	}

	// TODO: Figure out a variadic method that actually works!
	constexpr Vec<N, T>(T v0, T v1) : myVec({ v0, v1 })
	{
		static_assert(N == 2, "Vector dimensions must match!");
	}
	constexpr Vec<N, T>(T v0, T v1, T v2) : myVec({ v0, v1, v2 })
	{
		static_assert(N == 3, "Vector dimensions must match!");
	}
	constexpr Vec<N, T>(T v0, T v1, T v2, T v3) : myVec({ v0, v1, v2, v3 })
	{
		static_assert(N == 4, "Vector dimensions must match!");
	}
	constexpr Vec<N, T>(T v0, T v1, T v2, T v3, T v4) : myVec({ v0, v1, v2, v3, v4 })
	{
		static_assert(N == 5, "Vector dimensions must match!");
	}

	Vec<N, T>(const Vec<N, T>&) = default;
	Vec<N, T>& operator=(const Vec<N, T>&) = default;
	Vec<N, T>(Vec<N, T>&&) = default;
	Vec<N, T>& operator=(Vec<N, T>&&) = default;

	~Vec<N, T>() = default;

	template <typename S>
	explicit Vec<N, T>(const Vec<N, S>& source)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] = static_cast<T>(source[i]);
	}

	T& operator[](int index)
	{
		return myVec[index];
	}

	const T &operator[](int index) const
	{
		return myVec[index];
	}

	decltype(auto) data() { return myVec.data(); }
	decltype(auto) data() const { return myVec.data(); }

	Vec<N, T> operator+=(const Vec<N, T>& rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] += rhs[i];

		return *this;
	}

	Vec<N, T> operator-=(const Vec<N, T>& rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] -= rhs[i];
		return *this;
	}

	Vec<N, T> operator*=(const Vec<N, T>& rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] *= rhs[i];
		return *this;
	}

	Vec<N, T> operator/=(const Vec<N, T>& rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] /= rhs[i];
		return *this;
	}

	Vec<N, T> operator+(Vec<N, T> rhs) const
	{
		rhs += *this;
		return rhs;
	}

	Vec<N, T> operator-(const Vec<N, T>& rhs) const
	{
		Vec<N, T> temp(*this);
		temp -= rhs;
		return temp;
	}

	Vec<N, T> operator*(Vec<N, T> rhs) const
	{
		rhs *= *this;
		return rhs;
	}

	Vec<N, T> operator/(const Vec<N, T>& rhs) const
	{
		Vec<N, T> temp(*this);
		temp /= rhs;
		return temp;
	}

	Vec<N, T> operator-() const // unary minus
	{
		Vec<N, T> vec;

		for (int i = 0; i < N; ++i)
			vec[i] = -myVec[i];

		return vec;
	}

	Vec<N, T> operator+=(T rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] += rhs;

		return *this;
	}

	Vec<N, T> operator-=(T rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] -= rhs;

		return *this;
	}

	Vec<N, T> operator*=(T rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] *= rhs;

		return *this;
	}

	Vec<N, T> operator/=(T rhs)
	{
		for (int i = 0; i < N; ++i)
			myVec[i] /= rhs;

		return *this;
	}

	Vec<N, T> operator*(T rhs) const
	{
		Vec<N, T> vec(*this);
		vec *= rhs;
		return vec;
	}

	Vec<N, T> operator+(T rhs) const
	{
		Vec<N, T> vec(*this);
		vec += rhs;
		return vec;
	}

	Vec<N, T> operator/(T rhs) const
	{
		Vec<N, T> vec(*this);
		vec /= rhs;
		return vec;
	}

private:
	std::array<T, N> myVec;
};

using Vec2d = Vec<2, double>;
using Vec2f = Vec<2, float>;
using Vec2i = Vec<2, int>;
using Vec2ui = Vec<2, unsigned>;
using Vec2st = Vec<2, size_t>;

using Vec3d = Vec<3, double>;
using Vec3f = Vec<3, float>;
using Vec3i = Vec<3, int>;
using Vec3ui = Vec<3, unsigned>;
using Vec3st = Vec<3, size_t>;

using Vec4d = Vec<4, double>;
using Vec4f = Vec<4, float>;
using Vec4i = Vec<4, int>;
using Vec4ui = Vec<4, unsigned>;
using Vec4st = Vec<4, size_t>;

using Vec5d = Vec<5, double>;
using Vec5f = Vec<5, float>;
using Vec5i = Vec<5, int>;
using Vec5ui = Vec<5, unsigned>;
using Vec5st = Vec<5, size_t>;

using Vec6d = Vec<6, double>;
using Vec6f = Vec<6, float>;
using Vec6i = Vec<6, int>;
using Vec6ui = Vec<6, unsigned>;
using Vec6st = Vec<6, size_t>;

template<int N, typename T>
T mag2(const Vec<N, T>& vec)
{
	T val = 0;

	for (int i = 0; i < N; ++i)
		val += sqr(vec[i]);

	return val;
}

template<int N, typename T>
T mag(const Vec<N, T>& vec)
{
	return sqrt(mag2(vec));
}

template<int N, typename T>
T dist2(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	T val = 0;

	for (int i = 0; i < N; ++i)
		val += sqr(vec0[i] - vec1[i]);

	return val;
}

template<int N, typename T>
T dist(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	return std::sqrt(dist2(vec0, vec1));
}

template<int N, typename T>
void normalizeInPlace(Vec<N, T>& vec)
{
	vec /= mag(vec);
}

template<int N, typename T>
inline Vec<N, T> normalize(const Vec<N, T>& vec)
{
	Vec<N, T> localVec(vec);
	normalizeInPlace(localVec);
	return localVec;
}

template<int N, typename T>
T infNorm(const Vec<N, T>& vec)
{
	T val = 0;

	for (int i = 0; i < N; ++i)
		val = max(std::fabs(vec[i]), val);

	return val;
}

template<int N, typename T>
bool operator==(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	for (int i = 0; i < N; ++i)
		if (vec0[i] != vec1[i]) return false;

	return true;
}

template<int N, typename T>
bool operator!=(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	return !(vec0 == vec1);
}

template<int N, typename S, typename T>
Vec<N, T> operator*(S val, const Vec<N, T>& vec)
{
	Vec<N, T> localVec(vec);
	localVec *= static_cast<T>(val);
	return localVec;
}

#ifdef _MSC_VER
#undef min
#undef max
#endif

template<int N, typename T>
T min(const Vec<N, T>& vec)
{
	T val = vec[0];

	for (int i = 1; i < N; ++i)
		if (vec[i] < val) val = vec[i];

	return val;
}

// The second vector is passed by value because we would have
// to allocate memory anyway.

template<int N, typename T>
Vec<N, T> minUnion(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = (vec0[i] < vec1[i]) ? vec0[i] : vec1[i];

	return localVec;
}

template<int N, typename T>
inline Vec<N, T> maxUnion(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = (vec0[i] > vec1[i]) ? vec0[i] : vec1[i];

	return localVec;
}

template<int N, typename T>
T max(const Vec<N, T>& vec)
{
	T val = vec[0];

	for (int i = 1; i < N; ++i)
		if (vec[i] > val) val = vec[i];

	return val;
}

template<int N, typename T>
T dot(const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	T val = 0;

	for (int i = 0; i < N; ++i)
		val += vec0[i] * vec1[i];

	return val;
}

template<typename T>
T cross(const Vec<2, T>& vec0, const Vec<2, T>& vec1)
{
	return vec0[0] * vec1[1] - vec0[1] * vec1[0];
}

template<typename T>
Vec<3, T> cross(const Vec<3, T>& vec0, const Vec<3, T>& vec1)
{
	return Vec<3, T>(vec0[1] * vec1[2] - vec0[2] * vec1[1],
		vec0[2] * vec1[0] - vec0[0] * vec1[2],
		vec0[0] * vec1[1] - vec0[1] * vec1[0]);
}

template<int N, typename T>
Vec<N, T> round(const Vec<N, T>& vec)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = std::round(vec[i]);

	return localVec;
}

template<int N, typename T>
Vec<N, T> floor(const Vec<N, T>& vec)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = std::floor(vec[i]);

	return localVec;
}

template<int N, typename T>
Vec<N, T> ceil(const Vec<N, T>& vec)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = std::ceil(vec[i]);

	return localVec;
}

template<int N, typename T>
Vec<N, T> fabs(const Vec<N, T>& vec)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = std::fabs(vec[i]);

	return localVec;
}

template<int N, typename T>
Vec<N, T> clamp(const Vec<N, T>& vec, const Vec<N, T>& low, const Vec<N, T>& high)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = clamp(vec[i], low[i], high[i]);

	return localVec;
}

template<int N, typename T>
Vec<N, T> min(const Vec<N, T>& min0, const Vec<N, T>& min1)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = std::min(min0[i], min1[i]);

	return localVec;
}

template<int N, typename T>
Vec<N, T> max(const Vec<N, T>& max0, const Vec<N, T>& max1)
{
	Vec<N, T> localVec;
	for (int i = 0; i < N; ++i)
		localVec[i] = std::max(max0[i], max1[i]);

	return localVec;
}

template<int N, typename T>
void minAndMax(Vec<N, T>& vecMin, Vec<N, T>& vecMax, const Vec<N, T>& vec0, const Vec<N, T>& vec1)
{
	for (int i = 0; i < N; ++i)
		minAndMax(vecMin[i], vecMax[i], vec0[i], vec1[i]);
}

template<int N, typename T>
void updateMinOrMax(Vec<N, T>& vecMin, Vec<N, T>& vecMax, const Vec<N, T>& vec)
{
	for (int i = 0; i < N; ++i)
		updateMinOrMax(vecMin[i], vecMax[i], vec[i]);
}

template<int N, typename T>
void updateMinAndMax(Vec<N, T>& vecMin, Vec<N, T>& vecMax, const Vec<N, T>& vec)
{
	for (int i = 0; i < N; ++i)
		updateMinAndMax(vecMin[i], vecMax[i], vec[i]);
}

}

#endif
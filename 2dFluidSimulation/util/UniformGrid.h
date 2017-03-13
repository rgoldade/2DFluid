#pragma once

#include <algorithm>
#include "vec.h"

#include "core.h"

///////////////////////////////////
//
// UniformGrid.h/cpp
// Ryan Goldade 2016
//
// Uniform grid class that stores templated values at grid centers
// Any positioned-based storage here must be accounted for by the caller.
//
////////////////////////////////////

template <typename T>
class UniformGrid
{
public:

	UniformGrid() : m_nx(Vec2st(0)) {}

	UniformGrid(const Vec2st& nx) : m_nx(nx)
	{
		m_grid.resize(m_nx[0] * m_nx[1]);
	}

	UniformGrid(const Vec2st& nx, const T& val) : m_nx(nx)
	{
		m_grid.resize(m_nx[0] * m_nx[1], val);
	}
	
	//Accessor is y-major because the inside loop for most processes is naturally y. Should give better cache coherence.
	//Clamping should only occur for interpolation. Direct index access that's outside of the grid should
	//be a sign of an error.
	T& operator()(int x, int y)
	{ 
		assert(x < m_nx[0] && y < m_nx[1] &&
				x >= 0 && y >= 0);

		return m_grid[stride(Vec2st(x, y))];
	}

	const T& operator()(int x, int y) const
	{
		assert(x < m_nx[0] && y < m_nx[1] &&
			x >= 0 && y >= 0);

		return m_grid[stride(Vec2st(x, y))];
	}

	T val(int x, int y) const
	{
		assert(x < m_nx[0] && y < m_nx[1] &&
			x >= 0 && y >= 0);

		return m_grid[stride(Vec2st(x, y))];
	}

	void clear()
	{
		m_nx = Vec2st(0);
		m_grid.clear();
	}

	bool empty() const { return m_grid.empty(); }

	void resize(const Vec2st& nx)
	{
		m_nx = nx;
		m_grid.clear();
		m_grid.resize(m_nx[0] * m_nx[1]);
	}

	void resize(const Vec2st& nx, const T& val)
	{
		m_nx = nx;
		m_grid.clear();
		m_grid.resize(m_nx[0] * m_nx[1], val);
	}

	const Vec2st& size() const { return m_nx; }
		
	inline size_t stride(const Vec2st& coord) const
	{
		return coord[1] + m_nx[1] * coord[0];
	}

	inline Vec2st unstride(size_t elem) const
	{
		Vec2st coord;
		coord[0] = elem / m_nx[1];
		coord[1] = elem % m_nx[1];
		return coord;
	}
protected:
	//Grid center container
	std::vector<T> m_grid;
	Vec2st m_nx;
};
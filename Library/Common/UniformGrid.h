#pragma once

#include <vector>

#include "Common.h"

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

	UniformGrid() : m_size(Vec2ui(0)) {}

	UniformGrid(const Vec2ui& size) : m_size(size)
	{
		m_grid.resize(m_size[0] * m_size[1]);
	}

	UniformGrid(const Vec2ui& size, const T& val) : m_size(size)
	{
		m_grid.resize(m_size[0] * m_size[1], val);
	}
	
	// Accessor is y-major because the inside loop for most processes is naturally y. Should give better cache coherence.
	// Clamping should only occur for interpolation. Direct index access that's outside of the grid should
	// be a sign of an error.
	T& operator()(unsigned i, unsigned j) { return (*this)(Vec2ui(i, j)); }

	T& operator()(const Vec2ui& coord)
	{ 
		assert(coord[0] < m_size[0] && coord[1] < m_size[1]);
		return m_grid[flatten(coord)];
	}

	const T& operator()(unsigned i, unsigned j) const { return (*this)(Vec2ui(i, j)); }

	const T& operator()(const Vec2ui& coord) const
	{
		assert(coord[0] < m_size[0] && coord[1] < m_size[1]);
		return m_grid[flatten(coord)];
	}

	void clear()
	{
		m_size = Vec2ui(0);
		m_grid.clear();
	}

	bool empty() const { return m_grid.empty(); }

	void resize(const Vec2ui& size)
	{
		m_size = size;
		m_grid.clear();
		m_grid.resize(m_size[0] * m_size[1]);
	}

	void resize(const Vec2ui& size, const T& val)
	{
		m_size = size;
		m_grid.clear();
		m_grid.resize(m_size[0] * m_size[1], val);
	}

	const Vec2ui& size() const { return m_size; }
		
	unsigned flatten(const Vec2ui& coord) const
	{
		return coord[1] + m_size[1] * coord[0];
	}

	Vec2ui unflatten(unsigned elem) const
	{
		Vec2ui coord;
		coord[0] = elem / m_size[1];
		coord[1] = elem % m_size[1];
		return coord;
	}

protected:

	//Grid center container
	std::vector<T> m_grid;
	Vec2ui m_size;
};

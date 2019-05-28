#ifndef LIBRARY_UNIFORMGRID_H
#define LIBRARY_UNIFORMGRID_H

#include <vector>

#include "Common.h"
#include "Util.h"
#include "Vec.h"

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

	UniformGrid() : mySize(Vec2i(0)) {}

	UniformGrid(const Vec2i& size) : mySize(size)
	{
		assert(size[0] >= 0 && size[1] >= 0);
		myGrid.resize(mySize[0] * mySize[1]);
	}

	UniformGrid(const Vec2i& size, const T& val) : mySize(size)
	{
		assert(size[0] >= 0 && size[1] >= 0);
		myGrid.resize(mySize[0] * mySize[1], val);
	}
	
	// Accessor is y-major because the inside loop for most processes is naturally y. Should give better cache coherence.
	// Clamping should only occur for interpolation. Direct index access that's outside of the grid should
	// be a sign of an error.
	T& operator()(int i, int j) { return (*this)(Vec2i(i, j)); }

	T& operator()(const Vec2i& coord)
	{ 
		assert(coord[0] >= 0 && coord[0] < mySize[0] &&
				coord[0] >= 0 && coord[1] < mySize[1]);
		
		return myGrid[flatten(coord)];
	}

	const T& operator()(int i, int j) const { return (*this)(Vec2i(i, j)); }

	const T& operator()(const Vec2i& coord) const
	{
		assert(coord[0] >= 0 && coord[0] < mySize[0] &&
			coord[0] >= 0 && coord[1] < mySize[1]);

		return myGrid[flatten(coord)];
	}

	void clear()
	{
		mySize = Vec2i(0);
		myGrid.clear();
	}

	bool empty() const { return myGrid.empty(); }

	void resize(const Vec2i& size)
	{
		assert(size[0] >= 0 && size[1] >= 0);

		mySize = size;
		myGrid.clear();
		myGrid.resize(mySize[0] * mySize[1]);
	}

	void resize(const Vec2i& size, const T& initialValue)
	{
		assert(size[0] >= 0 && size[1] >= 0);

		mySize = size;
		myGrid.clear();
		myGrid.resize(mySize[0] * mySize[1], initialValue);
	}

	const Vec2i& size() const { return mySize; }
		
	int flatten(const Vec2i& coord) const
	{
		assert(coord[0] >= 0 && coord[0] < mySize[0] &&
			coord[0] >= 0 && coord[1] < mySize[1]);

		return coord[1] + mySize[1] * coord[0];
	}

	Vec2i unflatten(int index) const
	{
		assert(index >= 0);

		Vec2i coord;
		coord[0] = index / mySize[1];
		coord[1] = index % mySize[1];
		return coord;
	}

protected:

	//Grid center container
	std::vector<T> myGrid;
	Vec2i mySize;
};

#endif
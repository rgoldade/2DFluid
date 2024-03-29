#ifndef FLUIDSIM2D_UNIFORM_GRID_H
#define FLUIDSIM2D_UNIFORM_GRID_H

#include <iostream>
#include <fstream>

#include "GridUtilities.h"
#include "Utilities.h"

///////////////////////////////////
//
// UniformGrid.h/cpp
// Ryan Goldade 2016
//
// Uniform grid class that stores templated values at grid centers
// Any positioned-based storage here must be accounted for by the caller.
//
////////////////////////////////////

namespace FluidSim2D
{

template <typename T>
class UniformGrid
{
public:

	UniformGrid() : mySize(Vec2i::Zero()) {}

	UniformGrid(const Vec2i& size) : mySize(size)
	{
		for (int axis : {0, 1})
			assert(size[axis] > 0);

		myGrid.setZero(mySize[0] * mySize[1]);
	}

	UniformGrid(const Vec2i& size, const T& value) : mySize(size)
	{
		for (int axis : {0, 1})
			assert(size[axis] > 0);

		myGrid.resize(mySize[0] * mySize[1]);
		myGrid.array() = value;
	}

	// Accessor is y-major because the inside loop for most processes is naturally y. Should give better cache coherence.
	// Clamping should only occur for interpolation. Direct index access that's outside of the grid should
	// be a sign of an error.
	T& operator()(int i, int j) { return (*this)(Vec2i(i, j)); }

	T& operator()(const Vec2i& coord)
	{
		for (int axis : {0, 1})
			assert(coord[axis] >= 0 && coord[axis] < mySize[axis]);

		return myGrid[flatten(coord)];
	}

	const T& operator()(int i, int j) const { return (*this)(Vec2i(i, j)); }

	const T& operator()(const Vec2i& coord) const
	{
		for (int axis : {0, 1})
			assert(coord[axis] >= 0 && coord[axis] < mySize[axis]);

		return myGrid[flatten(coord)];
	}

	void clear()
	{
		mySize.setZero();
		myGrid.resize(0);
	}

	bool empty() const { return myGrid.rows() == 0; }

	void resize(const Vec2i& size)
	{
		for (int axis : {0, 1})
			assert(size[axis] >= 0);

		mySize = size;
		myGrid.resize(mySize[0] * mySize[1]);
	}

	void resize(const Vec2i& size, const T& value)
	{
		for (int axis : {0, 1})
			assert(size[axis] >= 0);
	
		resize(size);
		myGrid.array() = value;
	}

	void reset(const T& value)
	{
		myGrid.array() = value;
	}

	const Vec2i& size() const { return mySize; }

	int voxelCount() const { return mySize[0] * mySize[1]; }

	int flatten(const Vec2i& coord) const
	{
		return coord[1] + mySize[1] * coord[0];
	}

	Vec2i unflatten(int index) const
	{
		assert(index >= 0 && index < voxelCount());

		return Vec2i(index / mySize[1], index % mySize[1]);
	}

	void printAsOBJ(const std::string& filename) const;

protected:

	//Grid center container
	VectorXt<T> myGrid;
	Vec2i mySize;
};

template<typename T>
void UniformGrid<T>::printAsOBJ(const std::string& filename) const
{
	std::ofstream writer(filename + std::string(".obj"));

	// Print the grid as a heightfield in the y-axis.
	if (writer)
	{
		// Print vertices first.
		int vertexCount = 0;
		forEachVoxelRange(Vec2i::Zero(), this->mySize, [&](const Vec2i& cell)
		{
			Vec2d position = cell.cast<double>();

			int flatIndex = this->flatten(cell);
			assert(this->unflatten(flatIndex) == cell);

			writer << "v " << position[0] << " " << myGrid[flatIndex] << " " << position[1] << "#" << vertexCount << "\n";

			assert(flatIndex == vertexCount);
			++vertexCount;
		});

		assert(vertexCount == this->mySize[0] * this->mySize[1]);
		// Print quad faces
		forEachVoxelRange(Vec2i::Ones(), this->mySize, [&](const Vec2i& node)
		{
			writer << "f";
			for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
			{
				Vec2i cell = nodeToCellCCW(node, cellIndex);

				int quadVertexIndex = this->flatten(Vec2i(cell));

				assert(this->unflatten(quadVertexIndex) == cell);

				writer << " " << quadVertexIndex + 1;
			}
			writer << "\n";

		});
	}
	else
		std::cerr << "Failed to write to file: " << filename << std::endl;
}

}

#endif
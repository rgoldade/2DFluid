#ifndef LIBRARY_VECTOR_GRID_H
#define LIBRARY_VECTOR_GRID_H

#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// VectorGrid.h
// Ryan Goldade 2016
//
// Container class of two ScalarGrids.
// Provides control for cell-centered,
// node-centered or staggered formation.
//
////////////////////////////////////

namespace FluidSim2D::Utilities
{

namespace VectorGridSettings
{
	enum class SampleType { CENTER, STAGGERED, NODE };
}

template<typename T>
class VectorGrid
{
	using ScalarSampleType = ScalarGridSettings::SampleType;
	using BorderType = ScalarGridSettings::BorderType;
	using SampleType = VectorGridSettings::SampleType;

public:

	VectorGrid() : myXform(1., Vec2f(0.)), myGridSize(0)
	{}

	VectorGrid(const Transform& xform, const Vec2i& size,
		SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: VectorGrid(xform, size, T(0), sampleType, borderType)
	{}

	VectorGrid(const Transform& xform, const Vec2i& size, T value,
		SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: myXform(xform)
		, myGridSize(size)
		, mySampleType(sampleType)
	{
		switch (sampleType)
		{
		case SampleType::CENTER:
			myGrids[0] = ScalarGrid<T>(xform, size, value, ScalarSampleType::CENTER, borderType);
			myGrids[1] = ScalarGrid<T>(xform, size, value, ScalarSampleType::CENTER, borderType);
			break;
			// If the grid is 2x2, it has 3x2 x-aligned faces and 2x3 y-aligned faces.
			// This is handled inside of the ScalarGrid
		case SampleType::STAGGERED:
			myGrids[0] = ScalarGrid<T>(xform, size, value, ScalarSampleType::XFACE, borderType);
			myGrids[1] = ScalarGrid<T>(xform, size, value, ScalarSampleType::YFACE, borderType);
			break;
			// If the grid is 2x2, it has 3x3 nodes. This is handled inside of the ScalarGrid
		case SampleType::NODE:
			myGrids[0] = ScalarGrid<T>(xform, size, value, ScalarSampleType::NODE, borderType);
			myGrids[1] = ScalarGrid<T>(xform, size, value, ScalarSampleType::NODE, borderType);
		}
	}

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	bool isGridMatched(const VectorGrid<S>& grid) const
	{
		for (int axis : {0, 1})
			if (size(axis) != grid.size(axis)) return false;

		if (myXform != grid.xform()) return false;
		if (mySampleType != grid.sampleType()) return false;

		return true;
	}

	ScalarGrid<T>& grid(int axis)
	{
		return myGrids[axis];
	}

	const ScalarGrid<T>& grid(int axis) const
	{
		return myGrids[axis];
	}

	T& operator()(int i, int j, int axis) { return (*this)(Vec2i(i, j), axis); }

	T& operator()(const Vec2i& coord, int axis)
	{
		return myGrids[axis](coord);
	}

	const T& operator()(int i, int j, int axis) const { return (*this)(Vec2i(i, j), axis); }

	const T& operator()(const Vec2i& coord, int axis) const
	{
		return myGrids[axis](coord);
	}

	T maxMagnitude() const;

	Vec<2, T> biLerp(float x, float y) const { return biLerp(Vec2f(x, y)); }
	Vec<2, T> biLerp(const Vec2f& samplePoint) const
	{
		return Vec<2, T>(biLerp(samplePoint, 0), biLerp(samplePoint, 1));
	}

	T biLerp(float x, float y, int axis) const { return biLerp(Vec2f(x, y), axis); }
	T biLerp(const Vec2f& samplePoint, int axis) const
	{
		return myGrids[axis].biLerp(samplePoint);
	}

	Vec<2, T> biCubicInterp(float x, float y) const { return biCubicInterp(Vec2f(x, y)); }
	Vec<2, T> biCubicInterp(const Vec2f& samplePoint) const { return Vec<2, T>(biCubicInterp(samplePoint, 0), biCubicInterp(samplePoint, 1)); }

	T biCubicInterp(float x, float y, int axis) const { return biCubicInterp(Vec2f(x, y), axis); }
	T biCubicInterp(const Vec2f& samplePoint, int axis) const
	{
		return myGrids[axis].biCubicInterp(samplePoint);
	}

	// World space vs. index space converters need to be done at the 
	// underlying scalar grid level because the alignment of the two 
	// grids are different depending on the SampleType.
	Vec2f indexToWorld(const Vec2f& indexPoint, int axis) const
	{
		return myGrids[axis].indexToWorld(indexPoint);
	}

	Vec2f worldToIndex(const Vec2f& worldPoint, int axis) const
	{
		return myGrids[axis].worldToIndex(worldPoint);
	}

	float dx() const { return myXform.dx(); }
	Vec2f offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }

	Vec2i size(int axis) const { return myGrids[axis].size(); }
	Vec2i gridSize() const { return myGridSize; }
	SampleType sampleType() const { return mySampleType; }

	// Rendering methods
	void drawGrid(Renderer& renderer) const;
	void drawSamplePoints(Renderer& renderer,
							const Vec3f& colour0 = Vec3f(1, 0, 0),
							const Vec3f& colour1 = Vec3f(0, 0, 1),
							const Vec2f& sampleSizes = Vec2f(1.)) const;

	void drawSupersampledValues(Renderer& renderer, float sampleRadius = .5, int samples = 5, float sampleSize = 1.) const;
	void drawSamplePointVectors(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), float length = .25) const;

private:

	// This method is private to prevent future mistakes between this transform
	// and the staggered scalar grids
	Vec2f indexToWorld(const Vec2f& indexPos) const
	{
		return myXform.indexToWorld(indexPos);
	}

	std::array<ScalarGrid<T>, 2> myGrids;

	Transform myXform;

	Vec2i myGridSize;

	SampleType mySampleType;
};

// Magnitude is useful for CFL conditions
template<typename T>
T VectorGrid<T>::maxMagnitude() const
{
	T magnitude(0);

	tbb::enumerable_thread_specific<T> parallelMax(0);

	if (mySampleType == SampleType::CENTER || mySampleType == SampleType::NODE)
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myGrids[0].voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				T& localMax = parallelMax.local();
				for (int index = range.begin(); index != range.end(); ++index)
				{
					Vec2i coord = myGrids[0].unflatten(index);
					T localMagnitude = mag2(Vec<2, T>(myGrids[0](coord), myGrids[1](coord)));

					localMax = std::max(localMax, localMagnitude);
				}
			});
	}
	else if (mySampleType == SampleType::STAGGERED)
	{
		auto blocked_range = tbb::blocked_range2d<int>(0, myGridSize[0], std::sqrt(tbbLightGrainSize), 0, myGridSize[1], std::cbrt(tbbLightGrainSize));

		tbb::parallel_for(blocked_range, [&](const tbb::blocked_range2d<int>& range)
			{
				T& localMax = parallelMax.local();

				Vec2i cell;

				for (cell[0] = range.rows().begin(); cell[0] != range.rows().end(); ++cell[0])
					for (cell[1] = range.cols().begin(); cell[1] != range.cols().end(); ++cell[1])
					{
						Vec2f averageVector(0);

						for (int axis : {0, 1})
							for (int direction : {0, 1})
							{
								Vec2i face = cellToFace(cell, axis, direction);
								averageVector[axis] += .5 * myGrids[axis](face);
							}

						T localMagnitude = mag2(averageVector);
						localMax = std::max(localMax, localMagnitude);
					}
			});
	}

	parallelMax.combine_each([&](T x) { magnitude = std::max(magnitude, x); });

	return std::sqrt(magnitude);
}

template<typename T>
void VectorGrid<T>::drawGrid(Renderer& renderer) const
{
	myGrids[0].drawGrid(renderer);
}

template<typename T>
void VectorGrid<T>::drawSamplePoints(Renderer& renderer,
										const Vec3f& colour0,
										const Vec3f& colour1,
										const Vec2f& sampleSizes) const
{
	myGrids[0].drawSamplePoints(renderer, colour0, sampleSizes[0]);
	myGrids[1].drawSamplePoints(renderer, colour1, sampleSizes[1]);
}

template<typename T>
void VectorGrid<T>::drawSupersampledValues(Renderer& renderer, float sampleRadius, int samples, float sampleSize) const
{
	myGrids[0].drawSupersampledValues(renderer, sampleRadius, samples, sampleSize);
	myGrids[1].drawSupersampledValues(renderer, sampleRadius, samples, sampleSize);
}

template<typename T>
void VectorGrid<T>::drawSamplePointVectors(Renderer& renderer, const Vec3f& colour, float length) const
{
	std::vector<Vec2f> startPoints;
	std::vector<Vec2f> endPoints;
	
	forEachVoxelRange(Vec2i(0), myGridSize, [&](const Vec2i& cell)
	{
		Vec2f worldPoint = indexToWorld(Vec2f(cell) + Vec2f(.5));
		startPoints.push_back(worldPoint);

		Vec<2, T> sampleVector = biLerp(worldPoint);
		Vec2f vectorEnd = worldPoint + length * sampleVector;
		endPoints.push_back(vectorEnd);
	});

	renderer.addLines(startPoints, endPoints, colour);
}

}

#endif
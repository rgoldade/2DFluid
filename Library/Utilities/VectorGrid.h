#ifndef FLUIDSIM2D_VECTOR_GRID_H
#define FLUIDSIM2D_VECTOR_GRID_H

#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"

#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"

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

namespace FluidSim2D
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

	VectorGrid() : myXform(1., Vec2d::Zero()), myGridSize(Vec2i::Zero())
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

	Vec2t<T> biLerp(double x, double y) const { return biLerp(Vec2d(x, y)); }
	Vec2t<T> biLerp(const Vec2d& samplePoint) const
	{
		return Vec2t<T>(biLerp(samplePoint, 0), biLerp(samplePoint, 1));
	}

	T biLerp(double x, double y, int axis) const { return biLerp(Vec2d(x, y), axis); }
	T biLerp(const Vec2d& samplePoint, int axis) const
	{
		return myGrids[axis].biLerp(samplePoint);
	}

	Vec2t<T> biCubicInterp(double x, double y) const { return biCubicInterp(Vec2d(x, y)); }
	Vec2t<T> biCubicInterp(const Vec2d& samplePoint) const { return Vec2t<T>(biCubicInterp(samplePoint, 0), biCubicInterp(samplePoint, 1)); }

	T biCubicInterp(double x, double y, int axis) const { return biCubicInterp(Vec2d(x, y), axis); }
	T biCubicInterp(const Vec2d& samplePoint, int axis) const
	{
		return myGrids[axis].biCubicInterp(samplePoint);
	}

	// World space vs. index space converters need to be done at the 
	// underlying scalar grid level because the alignment of the two 
	// grids are different depending on the SampleType.
	FORCE_INLINE Vec2d indexToWorld(const Vec2d& indexPoint, int axis) const
	{
		return myGrids[axis].indexToWorld(indexPoint);
	}

	FORCE_INLINE Vec2d worldToIndex(const Vec2d& worldPoint, int axis) const
	{
		return myGrids[axis].worldToIndex(worldPoint);
	}

	double dx() const { return myXform.dx(); }
	Vec2d offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }

	Vec2i size(int axis) const { return myGrids[axis].size(); }
	Vec2i gridSize() const { return myGridSize; }
	SampleType sampleType() const { return mySampleType; }

	// Rendering methods
	void drawGrid(Renderer& renderer) const;
	void drawSamplePoints(Renderer& renderer,
							const Vec3d& colour0 = Vec3d(1, 0, 0),
							const Vec3d& colour1 = Vec3d(0, 0, 1),
							const Vec2d& sampleSizes = Vec2d(1.)) const;

	void drawSupersampledValues(Renderer& renderer, double sampleRadius = .5, int samples = 5, double sampleSize = 1.) const;
	void drawSamplePointVectors(Renderer& renderer, const Vec3d& colour = Vec3d(0, 0, 1), double length = .25) const;

private:

	// This method is private to prevent future mistakes between this transform
	// and the staggered scalar grids
	Vec2d indexToWorld(const Vec2d& indexPos) const
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
	T sqr_mag;

	if (mySampleType == SampleType::CENTER || mySampleType == SampleType::NODE)
	{
		sqr_mag = tbb::parallel_reduce(tbb::blocked_range<int>(0, myGrids[0].voxelCount(), tbbLightGrainSize), T(0),
		[&](const tbb::blocked_range<int>& range, T maxMagnitude) -> T
		{
			for (int index = range.begin(); index != range.end(); ++index)
			{
				Vec2i coord = myGrids[0].unflatten(index);
				T localMagnitude = Vec2t<T>(myGrids[0](coord), myGrids[1](coord)).squaredNorm();

				maxMagnitude = std::max(maxMagnitude, localMagnitude);
			}

			return maxMagnitude;
		},
		[](T a, T b) -> double
		{
			return std::max(a, b);
		});
	}
	else if (mySampleType == SampleType::STAGGERED)
	{
		auto blocked_range = tbb::blocked_range2d<int>(0, myGridSize[0], 0, myGridSize[1]);

		sqr_mag = tbb::parallel_reduce(blocked_range, T(0),
		[&](const tbb::blocked_range2d<int>& range, T maxMagnitude) -> T
		{
			Vec2i cell;

			for (cell[0] = range.rows().begin(); cell[0] != range.rows().end(); ++cell[0])
				for (cell[1] = range.cols().begin(); cell[1] != range.cols().end(); ++cell[1])
				{
					Vec2d averageVector = Vec2d::Zero();

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i face = cellToFace(cell, axis, direction);
							averageVector[axis] += .5 * myGrids[axis](face);
						}

					T localMagnitude = averageVector.squaredNorm();
					maxMagnitude = std::max(maxMagnitude, localMagnitude);
				}

			return maxMagnitude;
		},
		[](T a, T b) -> T
		{
			return std::max(a, b);
		});
	}

	return std::sqrt(sqr_mag);
}

template<typename T>
void VectorGrid<T>::drawGrid(Renderer& renderer) const
{
	myGrids[0].drawGrid(renderer);
}

template<typename T>
void VectorGrid<T>::drawSamplePoints(Renderer& renderer,
										const Vec3d& colour0,
										const Vec3d& colour1,
										const Vec2d& sampleSizes) const
{
	myGrids[0].drawSamplePoints(renderer, colour0, sampleSizes[0]);
	myGrids[1].drawSamplePoints(renderer, colour1, sampleSizes[1]);
}

template<typename T>
void VectorGrid<T>::drawSupersampledValues(Renderer& renderer, double sampleRadius, int samples, double sampleSize) const
{
	myGrids[0].drawSupersampledValues(renderer, sampleRadius, samples, sampleSize);
	myGrids[1].drawSupersampledValues(renderer, sampleRadius, samples, sampleSize);
}

template<typename T>
void VectorGrid<T>::drawSamplePointVectors(Renderer& renderer, const Vec3d& colour, double length) const
{
	VecVec2d startPoints;
	VecVec2d endPoints;
	
	forEachVoxelRange(Vec2i::Zero(), myGridSize, [&](const Vec2i& cell)
	{
		Vec2d worldPoint = indexToWorld(cell.cast<double>() + Vec2d(.5f, .5f));
		startPoints.push_back(worldPoint);

		Vec2t<T> sampleVector = biLerp(worldPoint);
		Vec2d vectorEnd = worldPoint + length * sampleVector;
		endPoints.push_back(vectorEnd);
	});

	renderer.addLines(startPoints, endPoints, colour);
}

}

#endif
#ifndef LIBRARY_VECTORGRID_H
#define LIBRARY_VECTORGRID_H

#include "Common.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Util.h"
#include "Vec.h"

///////////////////////////////////
//
// VectorGrid.h
// Ryan Goldade 2016
//
// Container class of two ScalarGrids.
// Provides control for cell-centered
// or staggered formation.
//
////////////////////////////////////

namespace VectorGridSettings
{
	enum class SampleType { CENTER, STAGGERED, NODE };
}

template<typename T>
class VectorGrid
{
	using ScalarGridT = ScalarGrid<T>;
	using ScalarSampleType = ScalarGridSettings::SampleType;
	using BorderType = ScalarGridSettings::BorderType;
	using SampleType = VectorGridSettings::SampleType;

public:

	VectorGrid() : myXform(1., Vec2R(0.)), mySize(0) { myGrids.resize(2); }
	VectorGrid(const Transform& xform, const Vec2i& size,
				SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: VectorGrid(xform, size, T(0), sampleType, borderType)
	{}

	VectorGrid(const Transform& xform, const Vec2i& size, T val,
				SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: myXform(xform)
		, mySize(size)
		, mySampleType(sampleType)
	{
		myGrids.resize(2);
		switch (sampleType)
		{
		case SampleType::CENTER:
			myGrids[0] = ScalarGridT(xform, size, val, ScalarSampleType::CENTER, borderType);
			myGrids[1] = ScalarGridT(xform, size, val, ScalarSampleType::CENTER, borderType);
			break;
		// If the grid is 2x2, it has 3x2 x-aligned faces and 2x3 y-aligned faces.
		// This is handled inside of the ScalarGrid
		case SampleType::STAGGERED:
			myGrids[0] = ScalarGridT(xform, size, val, ScalarSampleType::XFACE, borderType);
			myGrids[1] = ScalarGridT(xform, size, val, ScalarSampleType::YFACE, borderType);
			break;
		// If the grid is 2x2, it has 3x3 nodes. This is handled inside of the ScalarGrid
		case SampleType::NODE:
			myGrids[0] = ScalarGridT(xform, size, val, ScalarSampleType::NODE, borderType);
			myGrids[1] = ScalarGridT(xform, size, val, ScalarSampleType::NODE, borderType);
		}
	} 

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	bool isGridMatched(const VectorGrid<S>& grid) const
	{
		if (size(0) != grid.size(0)) return false;
		if (size(1) != grid.size(1)) return false;
		if (myXform != grid.xform()) return false;
		if (mySampleType != grid.sampleType()) return false;
		return true;
	}

	ScalarGridT& grid(int axis)
	{
		assert(axis >= 0 && axis < 2);
		return myGrids[axis];
	}

	const ScalarGridT& grid(int axis) const
	{
		assert(axis >= 0 && axis < 2);
		return myGrids[axis];
	}
	
	T& operator()(int i, int j, int axis) { return (*this)(Vec2i(i, j), axis); }

	T& operator()(const Vec2i& coord, int axis)
	{
		assert(axis >= 0 && axis < 2);
		return myGrids[axis](coord);
	}

	const T& operator()(int i, int j, int axis) const { return (*this)(Vec2i(i, j), axis); }

	const T& operator()(const Vec2i& coord, int axis) const
	{
		assert(axis >= 0 && axis < 2);
		return myGrids[axis](coord);
	}

	T maxMagnitude() const;

	Vec<T, 2> interp(Real x, Real y) const { return interp(Vec2R(x, y)); }
	Vec<T, 2> interp(const Vec2R& pos) const
	{
		return Vec<T, 2>(interp(pos, 0), interp(pos, 1));
	}

	T interp(Real x, Real y, int axis) const { return interp(Vec2R(x, y), axis); }
	T interp(const Vec2R& pos, int axis) const
	{ 
		assert(axis >= 0 && axis < 2);
		return myGrids[axis].interp(pos);
	}

	Vec<T, 2> cubicInterp(Real x, Real y) const { return cubicInterp(Vec2R(x, y)); }
	Vec<T, 2> cubicInterp(const Vec2R &pos) const { return Vec<T, 2>(cubicInterp(pos, 0), cubicInterp(pos, 1)); }

	T cubicInterp(Real x, Real y, int axis) const { return cubicInterp(Vec2R(x, y), axis); }
	T cubicInterp(const Vec2R &pos, int axis) const
	{
		assert(axis < 2);
		return myGrids[axis].cubicInterp(pos);
	}

	// World space vs. index space converters need to be done at the 
	// underlying scalar grid level because the alignment of the two 
	// grids are different depending on the SampleType.

	Vec2R indexToWorld(const Vec2R& indexPos, int axis) const
	{
		assert(axis >= 0 && axis < 2);
		return myGrids[axis].indexToWorld(indexPos);
	}

	Vec2R worldToIndex(const Vec2R& worldPos, int axis) const
	{
		assert(axis < 2);
		return myGrids[axis].worldToIndex(worldPos);
	}

	Real dx() const { return myXform.dx(); }
	Real offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }

	Vec2i size(int axis) const
	{
		assert(axis >= 0 && axis < 2);
		return myGrids[axis].size();
	}

	Vec2i gridSize() const { return mySize; }
	SampleType sampleType() const { return mySampleType; }

	// Rendering methods
	void drawGrid(Renderer& renderer) const;
	void drawSamplePoints(Renderer& renderer, const Vec3f& colour0 = Vec3f(1,0,0),
								const Vec3f& colour1 = Vec3f(0, 0, 1), const Vec2R& sizes = Vec2R(1.)) const;
	void drawSupersampledValues(Renderer& renderer, Real radius = .5, int samples = 5, Real sampleSize = 1.) const;
	void drawSamplePointVectors(Renderer& renderer, const Vec3f& colour = Vec3f(0,0,1), Real length = .25) const;

private:

	// This method is private to prevent future mistakes between this transform
	// and the staggered scalar grids
	Vec2R indexToWorld(const Vec2R& indexPos) const
	{
		return myXform.indexToWorld(indexPos);
	}

	std::vector<ScalarGridT> myGrids;

	Transform myXform;

	Vec2i mySize;

	SampleType mySampleType;
};

template<typename T>
void VectorGrid<T>::drawGrid(Renderer& renderer) const
{
	std::vector<Vec2R> startPoints;
	std::vector<Vec2R> endPoints;

	for (int axis : {0, 1})
	{
		for (int line = 0; line <= mySize[axis]; ++line)
		{
			Vec2R gridStart(0);
			gridStart[axis] = line;

			// Offset backwards because we want the start of the grid cell
			Vec2R startPoint = indexToWorld(gridStart);
			startPoints.push_back(startPoint);

			Vec2R gridEnd(mySize);
			gridEnd[axis] = line;

			Vec2R endPos = indexToWorld(gridEnd);
			endPoints.push_back(endPos);
		}
	}

	renderer.addLines(startPoints, endPoints, Vec3f(0));	
}

template<typename T>
void VectorGrid<T>::drawSamplePoints(Renderer& renderer, const Vec3f& colour0,
										const Vec3f& colour1, const Vec2R& sizes) const
{
	myGrids[0].drawSamplePoints(renderer,	colour0, sizes[0]);
	myGrids[1].drawSamplePoints(renderer, colour1, sizes[1]);
}

template<typename T>
void VectorGrid<T>::drawSupersampledValues(Renderer& renderer, Real radius, int samples, Real sampleSize) const
{
	myGrids[0].drawSupersampledValues(renderer, radius, samples, size);
	myGrids[1].drawSupersampledValues(renderer, radius, samples, size);
}

template<typename T>
void VectorGrid<T>::drawSamplePointVectors(Renderer& renderer, const Vec3f& colour, Real length) const
{
	std::vector<Vec2R> startPoints;
	std::vector<Vec2R> endPoints;

	switch (mySampleType)
	{
	case SampleType::CENTER:
		
		forEachVoxelRange(Vec2i(0), mySize, [&](const Vec2i& cell)
		{
			Vec2R worldPos = indexToWorld(Vec2R(cell), 0);
			startPoints.push_back(worldPos);

			Vec2R vec(myGrids[0](cell), myGrids[1](cell));
			endPoints.push_back(worldPos + length * vec);
		});
		break;

	case SampleType::NODE:

		forEachVoxelRange(Vec2i(0), myGrids[0].size(), [&](const Vec2i& node)
		{
			Vec2R worldPos = myGrids[0].indexToWorld(Vec2R(node));
			startPoints.push_back(worldPos);

			Vec2R vec(myGrids[0](node), myGrids[1](node));
			endPoints.push_back(worldPos + length * vec);
		});

		break;

	case SampleType::STAGGERED:

		forEachVoxelRange(Vec2i(0), mySize, [&](const Vec2i& cell)
		{
			Vec2R avgWorldPos(0);
			Vec2R avgVec(0);

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);

					avgWorldPos += .25 * indexToWorld(Vec2R(face), axis);

					avgVec[axis] += .5 * myGrids[axis](face);
				}

			startPoints.push_back(avgWorldPos);
			endPoints.push_back(avgWorldPos + length * avgVec);
		});
		
		break;
	}

	renderer.addLines(startPoints, endPoints, colour);
}

// Magnitude is useful for CFL conditions
template<typename T>
T VectorGrid<T>::maxMagnitude() const
{
	Real max = std::numeric_limits<Real>::min();
	switch (mySampleType)
	{
	case SampleType::CENTER:
		
		forEachVoxelRange(Vec2i(0), mySize, [&](const Vec2i& cell)
		{
			Real tempMag2 = mag2(Vec<T, 2>(myGrids[0](cell), myGrids[1](cell)));
				
			if (max < tempMag2) max = tempMag2;
		});

		return sqrt(max);
		break;

	case SampleType::NODE:

		forEachVoxelRange(Vec2i(0), mySize, [&](const Vec2i& node)
		{
			Real tempMag2 = mag2(Vec<T, 2>(myGrids[0](node), myGrids[1](node)));

			if (max < tempMag2) max = tempMag2;
		});

		return sqrt(max);
		break;

	case SampleType::STAGGERED:

		forEachVoxelRange(Vec2i(0), mySize, [&](const Vec2i& cell)
		{
			Vec2R avgVec(0);

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i face = cellToFace(cell, axis, direction);
					avgVec[axis] += .5 * myGrids[axis](face);
				}

			Real tempMag2 = mag2(avgVec);
			if (max < tempMag2) max = tempMag2;
		});
		
		return sqrt(max);
		break;
	default:
		assert(false);
	}
	return T(0);
}

#endif
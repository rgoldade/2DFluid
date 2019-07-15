#ifndef LIBRARY_SCALARGRID_H
#define LIBRARY_SCALARGRID_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "tbb/tbb.h"

#include "Common.h"
#include "Renderer.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Util.h"
#include "Vec.h"

///////////////////////////////////
//
// ScalarGrid.h
// Ryan Goldade 2016
//
// Thin wrapper around UniformGrid
// for scalar data types. ScalarGrid
// can be non-uniform and offset
// from the origin. It also allows
// for interpolation, etc. that would
// too specific for a generic templated
// grid.
//
////////////////////////////////////

// Grid layout:
//
//  node ------ y-face ------ node
//   |                         |
//   |                         |
// x-face       center       x-face
//   |                         |
//   |                         |
//  node ------ y-face ------ node

// TODO: add harmonic borders
namespace ScalarGridSettings
{
	enum class BorderType { CLAMP, ZERO, ASSERT };
	enum class SampleType { CENTER, XFACE, YFACE, NODE };
}

template<typename T>
class ScalarGrid : public UniformGrid<T>
{
	using BorderType = ScalarGridSettings::BorderType;
	using SampleType = ScalarGridSettings::SampleType;

public:

	ScalarGrid() : myXform(1.,Vec2R(0.)), myGridSize(Vec2i(0)), UniformGrid<T>() {}

	ScalarGrid(const Transform& xform, const Vec2i& size,
				SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: ScalarGrid(xform, size, T(0), sampleType, borderType)
	{}
		
	// The grid size is the number of actual grid cells to be created. This means that a 2sqrVal grid
	// will have 3cubeVal nodes, 3sqrVal x-aligned faces, 2cubeVal y-aligned faces and 2sqrVal cell centers. The size of 
	// the underlying storage container is reflected accordingly based the sample type to give the outside
	// caller the structure of a real grid.
	ScalarGrid(const Transform& xform, const Vec2i& size, const T& val,
				SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: myXform(xform)
		, mySampleType(sampleType)
		, myBorderType(borderType)
		, myGridSize(size)
	{
		switch (sampleType)
		{
		case SampleType::CENTER:
			myCellOffset = Vec2R(.5);
			this->resize(size, val);
			break;
		case SampleType::XFACE:
			myCellOffset = Vec2R(.0, .5);
			this->resize(size + Vec2i(1, 0), val);
			break;
		case SampleType::YFACE:
			myCellOffset = Vec2R(.5, .0);
			this->resize(size + Vec2i(0, 1), val);
			break;
		case SampleType::NODE:
			myCellOffset = Vec2R(.0);
			this->resize(size + Vec2i(1), val);
		}
	}

	SampleType sampleType() const { return mySampleType; }

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	bool isGridMatched(const ScalarGrid<S>& grid) const
	{
		if (this->mySize != grid.size()) return false;
		if (myXform != grid.xform()) return false;
		if (mySampleType != grid.sampleType()) return false;
		return true;
	}

	// Global multiply operator
	void operator*(const T& s)
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), 500), [&](const tbb::blocked_range<int>& range)
		{
			for (int i = range.begin(); i != range.end(); ++i)
				this->myGrid[i] *= s;
		});
	}

	// Global add operator
	void operator+(T s)
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), 500), [&](const tbb::blocked_range<int>& range)
		{
			for (int i = range.begin(); i != range.end(); ++i)
				this->myGrid[i] += s;
		});
	}

	T maxValue() const
	{
		using namespace tbb;
		return parallel_reduce(blocked_range<int>(0, this->myGrid.size(), 500), std::numeric_limits<T>::lowest(),
								[&](const blocked_range<int>& range, T value) -> T
								{
									T localMax = value;
									for (int i = range.begin(); i != range.end(); ++i)
										localMax = std::max(localMax, this->myGrid[i]);
									return localMax;
								},
								[](T x, T y) -> T
								{
									return std::max(x, y);
								});
	}

	T minValue() const
	{
		using namespace tbb;
		return parallel_reduce(blocked_range<int>(0, this->myGrid.size(), 500), std::numeric_limits<T>::max(),
								[&](const blocked_range<int>& range, T value) -> T
								{
									T localMin = value;
									for (int i = range.begin(); i != range.end(); ++i)
										localMin = std::min(localMin, this->myGrid[i]);
									return localMin;
								},
									[](T x, T y) -> T
								{
									return std::min(x, y);
								});
	}

	void minAndMaxValue(T& min, T& max) const
	{
		using MinMaxPair = std::pair<T, T>;
		using namespace tbb;
		MinMaxPair result = parallel_reduce(blocked_range<int>(0, this->myGrid.size(), 500), MinMaxPair(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()),
			[&](const blocked_range<int>& range, MinMaxPair valuePair) -> MinMaxPair
		{
			T localMin = valuePair.first;
			T localMax = valuePair.second;
			for (int i = range.begin(); i != range.end(); ++i)
			{
				localMin = std::min(localMin, this->myGrid[i]);
				localMax = std::max(localMax, this->myGrid[i]);
			}
			return MinMaxPair(localMin, localMax);
		},
			[](MinMaxPair x, MinMaxPair y) -> MinMaxPair
		{
			return MinMaxPair(std::min(x.first, y.first), std::max(x.second, y.second));
		});
		
		min = result.first; max = result.second;
	}

	T interp(Real x, Real y, bool isIndexSpace = false) const { return interp(Vec2R(x, y), isIndexSpace); }
	T interp(const Vec2R& pos, bool isIndexSpace = false) const;
	
	T cubicInterp(Real x, Real y, bool isIndexSpace = false, bool applyClamp = false) const
	{
		return cubicInterp(Vec2R(x, y), isIndexSpace, applyClamp);
	}

	T cubicInterp(const Vec2R& pos, bool isIndexSpace = false, bool applyClamp = false) const;

	// Converters between world space and local index space
	Vec2R indexToWorld(Vec2R indexPoint) const
	{
		return myXform.indexToWorld(indexPoint + myCellOffset);
	}
	
	Vec2R worldToIndex(Vec2R worldPos) const
	{
		return myXform.worldToIndex(worldPos) - myCellOffset;
	}

	// TODO: replace with a gradient applied to a cubic interpolant.
	Vec<T, 2> gradient(const Vec2R& worldPos, bool isIndexSpace = false) const
	{
		Real offset = isIndexSpace ? 1E-1 : 1E-1 * dx();
		T dTdx = interp(worldPos + Vec2R(offset, 0.), isIndexSpace) - interp(worldPos - Vec2R(offset, 0.), isIndexSpace);
		T dTdy = interp(worldPos + Vec2R(0., offset), isIndexSpace) - interp(worldPos - Vec2R(0., offset), isIndexSpace);
		Vec<T, 2> grad(dTdx, dTdy);
		return grad / (2 * offset);
	}
	
	Real dx() const { return myXform.dx(); }
	Vec2R offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }
	
	// Render methods
	void drawGrid(Renderer& renderer) const;
	void drawGridCell(Renderer& renderer, const Vec2i& coord, const Vec3f& colour = Vec3f(0)) const;

	void drawSamplePoints(Renderer& renderer, const Vec3f& colour = Vec3f(1,0,0), Real size = 1.) const;
	void drawSupersampledValues(Renderer& renderer, Real radius = .5, int samples = 5, Real sampleSize = 1.) const;
	void drawSampleGradients(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;
	void drawVolumetric(Renderer& renderer, const Vec3f& minColour, const Vec3f& maxColour, T minVal, T maxVal) const;

	void printAsCSV(std::string filename) const;
	void printAsOBJ(std::string filename) const;

private:

	// The main interpolation call after the template specialized clamping passes
	T interpLocal(const Vec2R& pos) const;

	// Store the actual grid size. The mySize member of UniformGrid
	// represents the grid sampling. The actual grid doesn't change based on sample
	// type but the underlying array of sample points do.
	Vec2i myGridSize;

	// The transform accounts for the grid spacings and the grid offset.
	// It includes transforms to and from index space.
	// The offset specifies the location of the lowest, left-most corner
	// of the grid. The actual sample point is offset (in index space)
	// from this point based on the SampleType
	Transform myXform;
	
	SampleType mySampleType;
	BorderType myBorderType;

	// The local offset (in index space) associated with the sample type
	Vec2R myCellOffset;
};

template<typename T>
T ScalarGrid<T>::cubicInterp(const Vec2R& samplePoint, bool isIndexSpace, bool applyClamp) const
{
	Vec2R indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	Vec2R floorPoint = floor(indexPoint);

	// Revert to linear interpolation near the boundaries
	if (floorPoint[0] < 1 || floorPoint[0] >= this->mySize[0] - 2 ||
		floorPoint[1] < 1 || floorPoint[1] >= this->mySize[1] - 2)
			return interp(indexPoint, true);

	Vec2R dx = indexPoint - Vec2R(floorPoint);
	dx = clamp(dx, Vec2R(0), Vec2R(1));

	T cubicInterps[4];
	for (int yOffset = -1; yOffset <= 2; ++yOffset)
	{
		Real y = floorPoint[1] + Real(yOffset);

		T p_1 = (*this)(Vec2i(floorPoint[0] - 1, y));
		T p0 =	(*this)(Vec2i(floorPoint[0],	  y));
		T p1 =	(*this)(Vec2i(floorPoint[0] + 1, y));
		T p2 =	(*this)(Vec2i(floorPoint[0] + 2, y));

		cubicInterps[yOffset + 1] = Util::cubicInterp(p_1, p0, p1, p2, dx[0]);
	}

	T cubicValue = Util::cubicInterp(cubicInterps[0],
								cubicInterps[1],
								cubicInterps[2],
								cubicInterps[3], dx[1]);

	if (applyClamp)
	{
		T v00 = (*this)(Vec2i(floorPoint[0], floorPoint[1]));
		T v10 = (*this)(Vec2i(floorPoint[0] + 1, floorPoint[1]));

		T v01 = (*this)(Vec2i(floorPoint[0], floorPoint[1] + 1));
		T v11 = (*this)(Vec2i(floorPoint[0] + 1, floorPoint[1] + 1));

		T clampMin = std::min(std::min(v00, v10), std::min(v01, v11));
		T clampMax = std::max(std::max(v00, v10), std::max(v01, v11));
		
		cubicValue = Util::clamp(cubicValue, clampMin, clampMax);
	}

	return cubicValue;
}

template<typename T>
T ScalarGrid<T>::interp(const Vec2R& samplePoint, bool isIndexSpace) const
{
	Vec2R indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	switch (myBorderType)
	{
	case BorderType::ZERO:
		if ((indexPoint[0] < 0.) || (indexPoint[1] < 0.) ||
			(indexPoint[0] > Real(this->mySize[0] - 1)) ||
			(indexPoint[1] > Real(this->mySize[1] - 1))) return T(0.);
	case BorderType::CLAMP:
		indexPoint[0] = (indexPoint[0] < 0.) ? 0. : ((indexPoint[0] > Real(this->mySize[0] - 1)) ? Real(this->mySize[0] - 1) : indexPoint[0]);
		indexPoint[1] = (indexPoint[1] < 0.) ? 0. : ((indexPoint[1] > Real(this->mySize[1] - 1)) ? Real(this->mySize[1] - 1) : indexPoint[1]);
		break;
	// Useful debug check. Equivalent to "NONE" for release mode
	case BorderType::ASSERT:
		assert((indexPoint[0] >= 0.) && (indexPoint[1] >= 0.) &&
					(indexPoint[0] <= Real(this->mySize[0] - 1)) &&
					(indexPoint[1] <= Real(this->mySize[1] - 1)));
		break;
	}

	return interpLocal(indexPoint);
}

// The local interp applies bi-linear interpolation on the UniformGrid. The 
// templated type must have operators for basic add/mult arithmetic.
template<typename T>
T ScalarGrid<T>::interpLocal(const Vec2R& indexPoint) const
{
	Vec2R floorPoint = floor(indexPoint);

	// This should always pass since the clamp test in the public interp
	// will not allow anything outside of the border to get this far.
	assert(floorPoint[0] >= 0 && floorPoint[1] >= 0);
	assert(floorPoint[0] <= Real(this->mySize[0] - 1) &&
			floorPoint[1] <= Real(this->mySize[1] - 1));

	if (floorPoint[0] == Real(this->mySize[0] - 1)) --floorPoint[0];
	if (floorPoint[1] == Real(this->mySize[1] - 1)) --floorPoint[1];

	Vec2R dx = indexPoint - Vec2R(floorPoint);
	dx = clamp(dx, Vec2R(0), Vec2R(1));

	// Use base grid class operator
	T v00 = (*this)(Vec2i(floorPoint[0], floorPoint[1]));
	T v10 = (*this)(Vec2i(floorPoint[0] + 1, floorPoint[1]));

	T v01 = (*this)(Vec2i(floorPoint[0], floorPoint[1] + 1));
	T v11 = (*this)(Vec2i(floorPoint[0] + 1, floorPoint[1] + 1));

	return Util::bilerp(v00, v10, v01, v11, dx[0], dx[1]);
}

template<typename T>
void ScalarGrid<T>::drawGridCell(Renderer& renderer, const Vec2i& cell, const Vec3f& colour) const
{
	std::vector<Vec2R> startPoints;
	std::vector<Vec2R> endPoints;

	const Vec2R edgeToNodeOffset[4][2] = { {Vec2R(0), Vec2R(1, 0) },
											{ Vec2R(0, 1), Vec2R(1, 1) },
											{ Vec2R(0), Vec2R(0,1) },
											{ Vec2R(1, 0), Vec2R(1,1)} };

	for (int edge = 0; edge < 4; ++edge)
	{
		Vec2R nodePoint0 = indexToWorld(Vec2R(cell) - myCellOffset + edgeToNodeOffset[edge][0]);
		Vec2R nodePoint1 = indexToWorld(Vec2R(cell) - myCellOffset + edgeToNodeOffset[edge][1]);
		
		startPoints.push_back(nodePoint0);
		endPoints.push_back(nodePoint1);
	}

	renderer.addLines(startPoints, endPoints, colour);
}

template<typename T>
void ScalarGrid<T>::drawGrid(Renderer& renderer) const
{
	std::vector<Vec2R> startPoints;
	std::vector<Vec2R> endPoints;

	for (int axis : {0, 1})
	{
		for (int line = 0; line <= myGridSize[axis]; ++line)
		{
			// Offset backwards because we want the start of the grid cell
			Vec2R gridStart(0);
			gridStart[axis] = line;

			Vec2R startPoint = indexToWorld(gridStart - myCellOffset);
			startPoints.push_back(startPoint);

			Vec2R gridEnd(myGridSize);
			gridEnd[axis] = line;

			Vec2R endPoint = indexToWorld(gridEnd - myCellOffset);
			endPoints.push_back(endPoint);
		}
	}

	renderer.addLines(startPoints, endPoints, Vec3f(0));
}

template<typename T>
void ScalarGrid<T>::drawSamplePoints(Renderer& renderer, const Vec3f& colour, Real size) const
{
	std::vector<Vec2R> samplePoints;

	Vec2i gridSize = this->mySize;

	forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
	{
		Vec2R indexPoint(cell);
		Vec2R worldPoint = indexToWorld(indexPoint);

		samplePoints.push_back(worldPoint);
	});

	renderer.addPoints(samplePoints, colour, size);
}

// Warning: there is no protection here for ASSERT border types
template<typename T>
void ScalarGrid<T>::drawSupersampledValues(Renderer& renderer, Real radius, int samples, Real sampleSize) const
{
	assert(radius >= 0 && samples >= 0 && sampleSize >= 0);
	std::vector<Vec2R> samplePoints;

	T minSample, maxSample;
	minAndMaxValue(minSample, maxSample);

	Vec2i gridSize = this->mySize;

	forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
	{
		Vec2R indexPoint(cell);

		// Supersample
		Real dx = 2. * radius / Real(samples);
		for (Real x = -radius; x <= radius; x += dx)
			for (Real y = -radius; y <= radius; y += dx)
			{
				Vec2R samplePoint = indexPoint + Vec2R(x, y);
				Vec2R worldPoint = indexToWorld(samplePoint);

				T val = (interp(worldPoint) - minSample) / (maxSample - minSample);
				renderer.addPoint(worldPoint, Vec3f(float(val), float(val), 0), sampleSize);
			}
	});
}

template<typename T>
void ScalarGrid<T>::drawSampleGradients(Renderer& renderer, const Vec3f& colour, Real length) const
{
	std::vector<Vec2R> sample_points;
	std::vector<Vec2R> gradient_points;

	Vec2i size = this->mySize;
	forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& cell)
	{
		Vec2R indexPoint = Vec2R(cell);
		Vec2R start_pos = indexToWorld(indexPoint);
		sample_points.push_back(start_pos);

		Vec<T, 2> grad = gradient(start_pos);
		Vec2R end_pos = start_pos + length * grad;
		gradient_points.push_back(end_pos);
	});

	renderer.addLines(sample_points, gradient_points, colour);
}

template<typename T>
void ScalarGrid<T>::drawVolumetric(Renderer& renderer, const Vec3f& minColour, const Vec3f& maxColour, T minVal, T maxVal) const
{
	ScalarGrid<Real> nodes(xform(), this->mySize, SampleType::NODE);

	std::vector<Vec2R> quadVertices(nodes.size()[0] * nodes.size()[1]);
	std::vector<Vec4i> pixelQuads(this->mySize[0] * this->mySize[1]);
	std::vector<Vec3f> colours(this->mySize[0] * this->mySize[1]);

	// Set node points
	{
		Vec2i size = nodes.size();

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& node)
		{
			int index = nodes.flatten(node);
			quadVertices[index] = nodes.indexToWorld(Vec2R(node));
		});
	}

	{
		Vec2i size = this->mySize;

		int quadCount = 0;

		forEachVoxelRange(Vec2i(0), size, [&](const Vec2i& cell)
		{
			Vec4i quad;

			for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
			{
				Vec2i node = cellToNodeCCW(cell, nodeIndex);
				quad[nodeIndex] = nodes.flatten(node);				
			}

			T pixelValue = (*this)(cell);

			pixelQuads[quadCount] = quad;

			Vec3f colour;
			if (pixelValue > maxVal) colour = maxColour;
			else if (pixelValue < minVal) colour = minColour;
			else
			{
				float s = pixelValue / (maxVal - minVal);
				colour = minColour * (1. - s) + maxColour * s;
			}

			colours[quadCount++] = colour;
		});
	}

	renderer.addQuads(quadVertices, pixelQuads, colours);
}

template<typename T>
void ScalarGrid<T>::printAsCSV(std::string filename) const
{
	std::ofstream writer(filename);

	if (writer)
	{
		for (int i = 0; i < this->mySize[0]; ++i)
		{
			int j = 0;
			for (; j < this->mySize[1] - 1; ++j)
			{
				writer << this->myGrid[this->flatten(Vec2i(i, j))] << ", ";
			}
			writer << this->myGrid[this->flatten(Vec2i(i, j))];
			writer << "\n";
		}
	}
	else
		std::cerr << "Failed to write to file: " << filename << std::endl;
}

template<typename T>
void ScalarGrid<T>::printAsOBJ(std::string filename) const
{
	std::ofstream writer(filename + std::string(".obj"));

	// Print the grid as a heightfield in the y-axis.
	if (writer)
	{
		// Print vertices first.
		int vertexCount = 0;
		forEachVoxelRange(Vec2i(0), this->mySize, [&](const Vec2i& cell)
		{
			Vec2R position = indexToWorld(Vec2R(cell));

			int flatIndex = this->flatten(cell);
			assert(this->unflatten(flatIndex) == cell);

			writer << "v " << position[0] << " " << (*this)(cell) << " " << position[1] << "#" << vertexCount << "\n";

			assert(flatIndex == vertexCount);
			++vertexCount;
		});

		assert(vertexCount == Real(this->mySize[0] * this->mySize[1]));
		// Print quad faces
		forEachVoxelRange(Vec2i(1), this->mySize, [&](const Vec2i& node)
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

#endif
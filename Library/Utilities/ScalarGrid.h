#ifndef LIBRARY_SCALAR_GRID_H
#define LIBRARY_SCALAR_GRID_H

#include "GridUtilities.h"
#include "Renderer.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"
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


namespace FluidSim2D::Utilities
{

using namespace RenderTools;

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

	ScalarGrid() : myXform(1., Vec2f(0.)), myGridSize(Vec2i(0)), UniformGrid<T>() {}

	ScalarGrid(const Transform& xform, const Vec2i& size,
		SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: ScalarGrid(xform, size, T(0), sampleType, borderType)
	{}

	// The grid size is the number of actual grid cells to be created. This means that a 2x2 grid
	// will have 3x3 nodes, 3x2 x-aligned faces, 3x2 y-aligned faces and 2x2 cell centers. The size of 
	// the underlying storage container is reflected accordingly based the sample type to give the outside
	// caller the structure of a real grid.
	ScalarGrid(const Transform& xform, const Vec2i& size, const T& initialValue,
		SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
		: myXform(xform)
		, mySampleType(sampleType)
		, myBorderType(borderType)
		, myGridSize(size)
	{
		switch (sampleType)
		{
		case SampleType::CENTER:
			myCellOffset = Vec2f(.5);
			this->resize(size, initialValue);
			break;
		case SampleType::XFACE:
			myCellOffset = Vec2f(.0, .5);
			this->resize(size + Vec2i(1, 0), initialValue);
			break;
		case SampleType::YFACE:
			myCellOffset = Vec2f(.5, .0);
			this->resize(size + Vec2i(0, 1), initialValue);
			break;
		case SampleType::NODE:
			myCellOffset = Vec2f(.0);
			this->resize(size + Vec2i(1), initialValue);
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
		if (this->myXform != grid.xform()) return false;
		if (this->mySampleType != grid.sampleType()) return false;
		return true;
	}

	// Global multiply operator
	void operator*(const T& scalar)
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int index = range.begin(); index != range.end(); ++index)
				this->myGrid[index] *= scalar;
		});
	}

	// Global add operator
	void operator+(const T& scalar)
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int index = range.begin(); index != range.end(); ++index)
				this->myGrid[index] += scalar;
		});
	}

	T maxValue() const
	{
		return tbb::parallel_reduce(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize), std::numeric_limits<T>::lowest(),
			[&](const tbb::blocked_range<int>& range, T maxValue) -> T
		{
			for (int index = range.begin(); index != range.end(); ++index)
				maxValue = std::max(maxValue, this->myGrid[index]);
			return maxValue;
		},
			[](T x, T y) -> T
		{
			return std::max(x, y);
		});
	}

	T minValue() const
	{
		return tbb::parallel_reduce(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize),
			std::numeric_limits<T>::max(),
			[&](const tbb::blocked_range<int>& range, T minValue) -> T
		{
			for (int index = range.begin(); index != range.end(); ++index)
				minValue = std::min(minValue, this->myGrid[index]);
			return minValue;
		},
			[](T x, T y) -> T
		{
			return std::min(x, y);
		});
	}

	std::pair<T, T> minAndMaxValue() const
	{
		using MinMaxPair = std::pair<T, T>;

		MinMaxPair result = tbb::parallel_reduce(tbb::blocked_range<int>(0, this->voxelCount(), tbbLightGrainSize),
			MinMaxPair(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()),
			[&](const tbb::blocked_range<int>& range, MinMaxPair valuePair) -> MinMaxPair
		{
			T localMin = valuePair.first;
			T localMax = valuePair.second;

			for (int index = range.begin(); index != range.end(); ++index)
			{
				localMin = std::min(localMin, this->myGrid[index]);
				localMax = std::max(localMax, this->myGrid[index]);
			}
			return MinMaxPair(localMin, localMax);
		},
			[](MinMaxPair x, MinMaxPair y) -> MinMaxPair
		{
			return MinMaxPair(std::min(x.first, y.first), std::max(x.second, y.second));
		});

		return std::pair<T, T>(result.first, result.second);
	}

	T biLerp(float x, float y, bool isIndexSpace = false) const { return biLerp(Vec2f(x, y), isIndexSpace); }
	T biLerp(const Vec2f& samplePoint, bool isIndexSpace = false) const;

	T biCubicInterp(float x, float y, bool isIndexSpace = false, bool applyClamp = false) const
	{
		return biCubicInterp(Vec2f(x, y), isIndexSpace, applyClamp);
	}

	T biCubicInterp(const Vec2f& pos, bool isIndexSpace = false, bool applyClamp = false) const;

	// Converters between world space and local index space
	Vec2f indexToWorld(const Vec2f& indexPoint) const
	{
		return myXform.indexToWorld(indexPoint + myCellOffset);
	}

	Vec2f worldToIndex(const Vec2f& worldPoint) const
	{
		return myXform.worldToIndex(worldPoint) - myCellOffset;
	}

	Vec<2, T> gradient(const Vec2f& worldPos, bool isIndexSpace = false) const
	{
		float offset = isIndexSpace ? 1E-1 : 1E-1 * dx();
		T dTdx = biLerp(worldPos + Vec2f(offset, 0.), isIndexSpace) - biLerp(worldPos - Vec2f(offset, 0.), isIndexSpace);
		T dTdy = biLerp(worldPos + Vec2f(0., offset), isIndexSpace) - biLerp(worldPos - Vec2f(0., offset), isIndexSpace);
		Vec<2, T> grad(dTdx, dTdy);
		return grad / (2 * offset);
	}

	float dx() const { return myXform.dx(); }
	Vec2f offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }

	// Render methods
	void drawGrid(Renderer& renderer) const;
	void drawGridCell(Renderer& renderer, const Vec2i& cell, const Vec3f& colour = Vec3f(0)) const;

	void drawSamplePoints(Renderer& renderer, const Vec3f& colour = Vec3f(1, 0, 0), float sampleSize = 1.) const;
	void drawSupersampledValues(Renderer& renderer, float sampleRadius = .5, int samples = 5, float sampleSize = 1.) const;

	void drawSampleGradients(Renderer& renderer, const Vec3f& colour = Vec3f(.5), float length = .25) const;
	void drawVolumetric(Renderer& renderer, const Vec3f& minColour, const Vec3f& maxColour, T minVal, T maxVal) const;

	void printAsCSV(std::string filename) const;
	void printAsOBJ(std::string filename) const;

private:

	// The main interpolation call after the template specialized clamping passes
	T biLerpLocal(const Vec2f& pos) const;

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
	Vec2f myCellOffset;
};

template<typename T>
T ScalarGrid<T>::biCubicInterp(const Vec2f& samplePoint, bool isIndexSpace, bool applyClamp) const
{
	Vec2f indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	Vec2i floorPoint = Vec2i(floor(indexPoint));

	// Revert to linear interpolation near the boundaries
	if (floorPoint[0] < 1 || floorPoint[0] >= this->mySize[0] - 2 ||
		floorPoint[1] < 1 || floorPoint[1] >= this->mySize[1] - 2)
		return biLerp(indexPoint, true);

	Vec2f dx = indexPoint - Vec2f(floorPoint);
	dx = clamp(dx, Vec2f(0), Vec2f(1));

	std::array<T, 4> biCubicInterp;
	for (int yOffset = -1; yOffset <= 2; ++yOffset)
	{
		int y = floorPoint[1] + yOffset;

		T p_1 = (*this)(floorPoint[0] - 1, y);
		T p0 = (*this)(floorPoint[0], y);
		T p1 = (*this)(floorPoint[0] + 1, y);
		T p2 = (*this)(floorPoint[0] + 2, y);

		biCubicInterp[yOffset + 1] = cubicInterp(p_1, p0, p1, p2, dx[0]);
	}

	T cubicValue = cubicInterp(biCubicInterp[0],
								biCubicInterp[1],
								biCubicInterp[2],
								biCubicInterp[3],
								dx[1]);

	if (applyClamp)
	{
		T v00 = (*this)(floorPoint[0], floorPoint[1]);
		T v10 = (*this)(floorPoint[0] + 1, floorPoint[1]);

		T v01 = (*this)(floorPoint[0], floorPoint[1] + 1);
		T v11 = (*this)(floorPoint[0] + 1, floorPoint[1] + 1);

		T clampMin = std::min(std::min(v00, v10), std::min(v01, v11));
		T clampMax = std::max(std::max(v00, v10), std::max(v01, v11));

		cubicValue = clamp(cubicValue, clampMin, clampMax);
	}

	return cubicValue;
}

template<typename T>
T ScalarGrid<T>::biLerp(const Vec2f& samplePoint, bool isIndexSpace) const
{
	Vec2f indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	switch (myBorderType)
	{
	case BorderType::ZERO:

		for (int axis : {0, 1})
		{
			if (indexPoint[axis] < 0 || indexPoint[axis] > float(this->mySize[axis] - 1))
				return T(0);
		}

		break;
	case BorderType::CLAMP:

		for (int axis : {0, 1})
			indexPoint[axis] = clamp(indexPoint[axis], float(0), float(this->mySize[axis] - 1));

		break;
		// Useful debug check. Equivalent to "NONE" for release mode
	case BorderType::ASSERT:

		for (int axis : {0, 1})
			assert(indexPoint[axis] >= 0 && indexPoint[axis] <= float(this->mySize[axis] - 1));

		break;
	}

	return biLerpLocal(indexPoint);
}

// The local interp applies bi-linear interpolation on the UniformGrid. The 
// templated type must have operators for basic add/mult arithmetic.
template<typename T>
T ScalarGrid<T>::biLerpLocal(const Vec2f& indexPoint) const
{
	Vec2f floorPoint = floor(indexPoint);
	Vec2i baseSampleCell = Vec2i(floorPoint);

	for (int axis : {0, 1})
	{
		if (baseSampleCell[axis] == this->mySize[axis] - 1)
			--baseSampleCell[axis];
	}

	// Use base grid class operator
	T v00 = (*this)(Vec2i(baseSampleCell[0], baseSampleCell[1]));
	T v10 = (*this)(Vec2i(baseSampleCell[0] + 1, baseSampleCell[1]));

	T v01 = (*this)(Vec2i(baseSampleCell[0], baseSampleCell[1] + 1));
	T v11 = (*this)(Vec2i(baseSampleCell[0] + 1, baseSampleCell[1] + 1));

	Vec2f dx = indexPoint - floorPoint;

	for (int axis : {0, 1})
		assert(dx[axis] >= 0 && dx[axis] <= 1);

	return bilerp(v00, v10, v01, v11, dx[0], dx[1]);
}

template<typename T>
void ScalarGrid<T>::drawGrid(Renderer& renderer) const
{
	std::vector<Vec2f> startPoints;
	std::vector<Vec2f> endPoints;

	for (int axis : {0, 1})
	{
		Vec2i start(0);
		Vec2i end = myGridSize + Vec2i(1);
		end[axis] = 1;

		forEachVoxelRange(start, end, [&](const Vec2i& cell)
		{
			Vec2f gridStart(cell);

			Vec2f startPoint = indexToWorld(gridStart - myCellOffset);
			startPoints.push_back(startPoint);

			Vec2f gridEnd(cell);
			gridEnd[axis] = myGridSize[axis];

			Vec2f endPoint = indexToWorld(gridEnd - myCellOffset);
			endPoints.push_back(endPoint);
		});
	}

	renderer.addLines(startPoints, endPoints, Vec3f(0));
}

template<typename T>
void ScalarGrid<T>::drawGridCell(Renderer& renderer, const Vec2i& cell, const Vec3f& colour) const
{
	std::vector<Vec2f> startPoints;
	std::vector<Vec2f> endPoints;

	const Vec2f edgeToNodeOffset[4][2] = { {Vec2f(0), Vec2f(1, 0) },
											{ Vec2f(0, 1), Vec2f(1, 1) },
											{ Vec2f(0), Vec2f(0,1) },
											{ Vec2f(1, 0), Vec2f(1,1)} };

	for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
	{
		Vec2f startNode = indexToWorld(Vec2f(cell) - myCellOffset + edgeToNodeOffset[edgeIndex][0]);
		Vec2f endNode = indexToWorld(Vec2f(cell) - myCellOffset + edgeToNodeOffset[edgeIndex][1]);

		startPoints.push_back(startNode);
		endPoints.push_back(endNode);
	}

	renderer.addLines(startPoints, endPoints, colour);
}

template<typename T>
void ScalarGrid<T>::drawSamplePoints(Renderer& renderer, const Vec3f& colour, float sampleSize) const
{
	tbb::enumerable_thread_specific<std::vector<Vec2f>> parallelSamplePoints;

	tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localSamplePoints = parallelSamplePoints.local();

		for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
		{
			Vec2i sampleCoord = this->unflatten(sampleIndex);

			Vec2f worldPoint = indexToWorld(Vec2f(sampleCoord));

			localSamplePoints.push_back(worldPoint);
		}
	});

	std::vector<Vec2f> samplePoints;
	mergeLocalThreadVectors(samplePoints, parallelSamplePoints);

	renderer.addPoints(samplePoints, colour, sampleSize);
}

// Warning: there is no protection here for ASSERT border types
template<typename T>
void ScalarGrid<T>::drawSupersampledValues(Renderer& renderer, float sampleRadius, int samples, float sampleSize) const
{
	std::pair<T, T> minMaxPair = minAndMaxValue();
	T minSample = minMaxPair.first;
	T maxSample = minMaxPair.second;

	forEachVoxelRange(Vec2i(0), this->mySize, [&](const Vec2i& cell)
	{
		// Supersample
		float dx = 2. * sampleRadius / float(samples);
		Vec2f indexPoint(cell);
		Vec2f sampleOffset;
		for (sampleOffset[0] = -sampleRadius; sampleOffset[0] <= sampleRadius; sampleOffset[0] += dx)
			for (sampleOffset[1] = -sampleRadius; sampleOffset[1] <= sampleRadius; sampleOffset[1] += dx)
			{
				Vec2f samplePoint = indexPoint + sampleOffset;
				Vec2f worldPoint = indexToWorld(samplePoint);

				T value = (biLerp(worldPoint) - minSample) / (maxSample - minSample);

				renderer.addPoint(worldPoint, Vec3f(value, value, 0), sampleSize);
			}
	});
}

template<typename T>
void ScalarGrid<T>::drawSampleGradients(Renderer& renderer, const Vec3f& colour, float length) const
{
	std::vector<Vec2f> samplePoints;
	std::vector<Vec2f> gradientPoints;

	forEachVoxelRange(Vec2i(0), this->mySize, [&](const Vec2i& cell)
	{
		Vec2f worldPoint = indexToWorld(Vec2f(cell));
		samplePoints.push_back(worldPoint);

		Vec<2, T> gradVector = gradient(worldPoint);
		Vec2f vectorEnd = worldPoint + length * gradVector;
		gradientPoints.push_back(vectorEnd);
	});

	renderer.addLines(samplePoints, gradientPoints, colour);
}

template<typename T>
void ScalarGrid<T>::drawVolumetric(Renderer& renderer, const Vec3f& minColour, const Vec3f& maxColour, T minVal, T maxVal) const
{
	ScalarGrid<float> nodes(xform(), this->mySize, SampleType::NODE);

	std::vector<Vec2f> quadVertices(nodes.size()[0] * nodes.size()[1]);
	std::vector<Vec4i> pixelQuads(this->mySize[0] * this->mySize[1]);
	std::vector<Vec3f> colours(this->mySize[0] * this->mySize[1]);

	// Set node points
	forEachVoxelRange(Vec2i(0), nodes.size(), [&](const Vec2i& node)
	{
		int index = nodes.flatten(node);
		quadVertices[index] = nodes.indexToWorld(Vec2f(node));
	});

	{
		int quadCount = 0;

		forEachVoxelRange(Vec2i(0), this->mySize, [&](const Vec2i& cell)
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

	renderer.addQuadFaces(quadVertices, pixelQuads, colours);
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
			Vec2f position = indexToWorld(Vec2f(cell));

			int flatIndex = this->flatten(cell);
			assert(this->unflatten(flatIndex) == cell);

			writer << "v " << position[0] << " " << (*this)(cell) << " " << position[1] << "#" << vertexCount << "\n";

			assert(flatIndex == vertexCount);
			++vertexCount;
		});

		assert(vertexCount == float(this->mySize[0] * this->mySize[1]));
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

		writer.close();
	}
	else
		std::cerr << "Failed to write to file: " << filename << std::endl;
}

}
#endif
#ifndef FLUIDSIM2D_SCALAR_GRID_H
#define FLUIDSIM2D_SCALAR_GRID_H

#include <array>

#include "GridUtilities.h"
#include "Renderer.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"

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


namespace FluidSim2D
{

namespace ScalarGridSettings
{
	enum class BorderType { CLAMP, ZERO };
	enum class SampleType { CENTER, XFACE, YFACE, NODE };
}

template<typename T>
class ScalarGrid : public UniformGrid<T>
{
	using BorderType = ScalarGridSettings::BorderType;
	using SampleType = ScalarGridSettings::SampleType;

public:

	ScalarGrid() : myXform(1., Vec2d::Zero()), myGridSize(Vec2i::Zero()), UniformGrid<T>() {}

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
			myCellOffset = Vec2d(.5, .5);
			this->resize(size, initialValue);
			break;
		case SampleType::XFACE:
			myCellOffset = Vec2d(.0, .5);
			this->resize(size + Vec2i(1, 0), initialValue);
			break;
		case SampleType::YFACE:
			myCellOffset = Vec2d(.5, .0);
			this->resize(size + Vec2i(0, 1), initialValue);
			break;
		case SampleType::NODE:
			myCellOffset = Vec2d(.0, .0);
			this->resize(size + Vec2i::Ones(), initialValue);
		}
	}

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling scheme
	template<typename S>
	bool isGridMatched(const ScalarGrid<S>& grid) const
	{
		if (this->mySize != grid.size()) return false;
		if (this->myXform != grid.xform()) return false;
		if (this->mySampleType != grid.sampleType()) return false;
		return true;
	}

	T maxValue() const
	{
		return this->myGrid.maxCoeff();
	}

	T minValue() const
	{
		return this->myGrid.minCoeff();
	}

	std::pair<T, T> minAndMaxValue() const
	{
		return std::make_pair<T, T>(minValue(), maxValue());
	}

	FORCE_INLINE T biLerp(double x, double y, bool isIndexSpace = false) const { return biLerp(Vec2d(x, y), isIndexSpace); }
	T biLerp(const Vec2d& samplePoint, bool isIndexSpace = false) const;

	FORCE_INLINE T biCubicInterp(double x, double y, bool isIndexSpace = false, bool applyClamp = false) const
	{
		return biCubicInterp(Vec2d(x, y), isIndexSpace, applyClamp);
	}

	T biCubicInterp(const Vec2d& pos, bool isIndexSpace = false, bool applyClamp = false) const;

	// Converters between world space and local index space
	FORCE_INLINE Vec2d indexToWorld(const Vec2d& indexPoint) const
	{
		return myXform.indexToWorld(indexPoint + myCellOffset);
	}

	FORCE_INLINE Vec2d worldToIndex(const Vec2d& worldPoint) const
	{
		return myXform.worldToIndex(worldPoint) - myCellOffset;
	}

	Vec2t<T> biLerpGradient(const Vec2d& worldPos, bool isIndexSpace = false) const;
	Vec2t<T> biCubicGradient(const Vec2d& worldPos, bool isIndexSpace = false) const;

	double dx() const { return myXform.dx(); }
	Vec2d offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }

	SampleType sampleType() const { return mySampleType; }

	// Render methods
	void drawGrid(Renderer& renderer) const;
	void drawGridCell(Renderer& renderer, const Vec2i& cell, const Vec3d& colour = Vec3d::Zero()) const;

	void drawSamplePoints(Renderer& renderer, const Vec3d& colour = Vec3d(1, 0, 0), double sampleSize = 1.) const;
	void drawSupersampledValues(Renderer& renderer, double sampleRadius = .5, int samples = 5, double sampleSize = 1.) const;

	void drawSampleGradients(Renderer& renderer, const Vec3d& colour = Vec3d(.5, .5, .5), double length = .25) const;
	void drawVolumetric(Renderer& renderer, const Vec3d& minColour, const Vec3d& maxColour, T minVal, T maxVal) const;

	void printAsCSV(std::string filename) const;
	void printAsOBJ(std::string filename) const;

private:

	// The main interpolation call after the template specialized clamping passes
	T biLerpLocal(const Vec2d& pos) const;

	Vec2t<T> biLerpGradientLocal(const Vec2d& samplePoint) const;

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
	Vec2d myCellOffset;
};

template<typename T>
T ScalarGrid<T>::biCubicInterp(const Vec2d& samplePoint, bool isIndexSpace, bool applyClamp) const
{
	Vec2d indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	Vec2i floorPoint = indexPoint.cast<int>();

	// Revert to linear interpolation near the boundaries
	if (floorPoint[0] < 1 || floorPoint[0] >= this->mySize[0] - 2 ||
		floorPoint[1] < 1 || floorPoint[1] >= this->mySize[1] - 2)
		return biLerp(indexPoint, true);

	Vec2d dx = indexPoint - floorPoint.cast<double>();
	dx = clamp(dx, Vec2d::Zero().eval(), Vec2d::Ones().eval());

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

		cubicValue = std::clamp(cubicValue, clampMin, clampMax);
	}

	return cubicValue;
}

template<typename T>
T ScalarGrid<T>::biLerp(const Vec2d& samplePoint, bool isIndexSpace) const
{
	Vec2d indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	switch (myBorderType)
	{
		case BorderType::ZERO:
		{
			for (int axis : {0, 1})
			{
				if (indexPoint[axis] < 0 || indexPoint[axis] > double(this->mySize[axis] - 1))
					return T(0);
			}

			break;
		}
		case BorderType::CLAMP:
		{
			for (int axis : {0, 1})
			{
				if (indexPoint[axis] < 0 || indexPoint[axis] > double(this->mySize[axis] - 1))
					indexPoint[axis] = std::clamp(indexPoint[axis], double(0), double(this->mySize[axis] - 1));
			}
		}
	}

	return biLerpLocal(indexPoint);
}

// The local interp applies bi-linear interpolation on the UniformGrid. The 
// templated type must have operators for basic add/mult arithmetic.
template<typename T>
T ScalarGrid<T>::biLerpLocal(const Vec2d& indexPoint) const
{
	Vec2d floorPoint = indexPoint.array().floor();
	Vec2i baseSampleCell = floorPoint.cast<int>();

	for (int axis : {0, 1})
	{
		if (baseSampleCell[axis] == this->mySize[axis] - 1)
			--baseSampleCell[axis];
	}

	// Use base grid class operator
	T v00 = (*this)(baseSampleCell[0], baseSampleCell[1]);
	T v10 = (*this)(baseSampleCell[0] + 1, baseSampleCell[1]);

	T v01 = (*this)(baseSampleCell[0], baseSampleCell[1] + 1);
	T v11 = (*this)(baseSampleCell[0] + 1, baseSampleCell[1] + 1);

	Vec2d dx = indexPoint - floorPoint;

	for (int axis : {0, 1})
		assert(dx[axis] >= 0 && dx[axis] <= 1);

	return bilerp<T, double>(v00, v10, v01, v11, dx[0], dx[1]);
}

template<typename T>
Vec2t<T> ScalarGrid<T>::biLerpGradient(const Vec2d& samplePoint, bool isIndexSpace) const
{
	Vec2d indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	switch (myBorderType)
	{
		case BorderType::ZERO:
		{
			for (int axis : {0, 1})
			{
				if (indexPoint[axis] < 0 || indexPoint[axis] > double(this->mySize[axis] - 1))
					return Vec2t<T>::Zero();
			}

			break;
		}
		case BorderType::CLAMP:
		{
			for (int axis : {0, 1})
				indexPoint[axis] = std::clamp(indexPoint[axis], double(0), double(this->mySize[axis] - 1));

			break;
		}
	}

	return biLerpGradientLocal(indexPoint);
}

template<typename T>
Vec2t<T> ScalarGrid<T>::biLerpGradientLocal(const Vec2d& samplePoint) const
{
	Vec2d floorPoint = samplePoint.array().floor();
	Vec2i baseSampleCell = floorPoint.cast<int>();

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

	Vec2d deltaX = samplePoint - floorPoint;

	for (int axis : {0, 1})
		assert(deltaX[axis] >= 0 && deltaX[axis] <= 1);
	
	return bilerpGradient<T, double>(v00, v10, v01, v11, deltaX[0], deltaX[1]) / dx();
}

template<typename T>
Vec2t<T> ScalarGrid<T>::biCubicGradient(const Vec2d& samplePoint, bool isIndexSpace) const
{
	Vec2d indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	Vec2i floorPoint = indexPoint.cast<int>();

	// Revert to linear interpolation near the boundaries
	if (floorPoint[0] < 1 || floorPoint[0] >= this->mySize[0] - 2 ||
		floorPoint[1] < 1 || floorPoint[1] >= this->mySize[1] - 2)
		return biLerpGradient(indexPoint, true);

	Vec2d deltaX = indexPoint - floorPoint.cast<double>();

	for (int axis : {0, 1})
		assert(deltaX[axis] >= 0 && deltaX[axis] <= 1);

	Vec2d grad;
	// Compute X-gradient
	{
		std::array<T, 4> biCubicGradient;
		for (int yOffset = -1; yOffset <= 2; ++yOffset)
		{
			int y = floorPoint[1] + yOffset;

			T p_1 = (*this)(floorPoint[0] - 1, y);
			T p0 = (*this)(floorPoint[0], y);
			T p1 = (*this)(floorPoint[0] + 1, y);
			T p2 = (*this)(floorPoint[0] + 2, y);

			biCubicGradient[yOffset + 1] = cubicInterpGradient(p_1, p0, p1, p2, deltaX[0]);
		}

		grad[0] = cubicInterp(biCubicGradient[0],
								biCubicGradient[1],
								biCubicGradient[2],
								biCubicGradient[3],
								deltaX[1]);
	}

	// Compute Y-gradient
	{
		std::array<T, 4> biCubicGradient;
		for (int xOffset = -1; xOffset <= 2; ++xOffset)
		{
			int x = floorPoint[0] + xOffset;

			T p_1 = (*this)(x, floorPoint[1] - 1);
			T p0 = (*this)(x, floorPoint[1]);
			T p1 = (*this)(x, floorPoint[1] + 1);
			T p2 = (*this)(x, floorPoint[1] + 2);

			biCubicGradient[xOffset + 1] = cubicInterpGradient(p_1, p0, p1, p2, deltaX[1]);
		}

		grad[1] = cubicInterp(biCubicGradient[0],
								biCubicGradient[1],
								biCubicGradient[2],
								biCubicGradient[3],
								deltaX[0]);
	}

	return grad / dx();
}

template<typename T>
void ScalarGrid<T>::drawGrid(Renderer& renderer) const
{
	VecVec2d startPoints;
	VecVec2d endPoints;

	for (int axis : {0, 1})
	{
		Vec2i start(0);
		Vec2i end = myGridSize + Vec2i::Ones();
		end[axis] = 1;

		forEachVoxelRange(start, end, [&](const Vec2i& cell)
		{
			Vec2d gridStart = cell.cast<double>();

			Vec2d startPoint = indexToWorld(gridStart - myCellOffset);
			startPoints.push_back(startPoint);

			Vec2d gridEnd = cell.cast<double>();
			gridEnd[axis] = double(myGridSize[axis]);

			Vec2d endPoint = indexToWorld(gridEnd - myCellOffset);
			endPoints.push_back(endPoint);
		});
	}

	renderer.addLines(startPoints, endPoints, Vec3d(0));
}

template<typename T>
void ScalarGrid<T>::drawGridCell(Renderer& renderer, const Vec2i& cell, const Vec3d& colour) const
{
	VecVec2d startPoints;
	VecVec2d endPoints;

	const Vec2d edgeToNodeOffset[4][2] = { {Vec2d::Zero(), Vec2d(1.f, 0.f) },
											{ Vec2d(0.f, 1.f), Vec2d::Ones() },
											{ Vec2d::Zero(), Vec2d(0.f, 1.f) },
											{ Vec2d(1.f, 0.f), Vec2d::Ones()} };

	for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
	{
		Vec2d startNode = indexToWorld(cell.cast<double>() - myCellOffset + edgeToNodeOffset[edgeIndex][0]);
		Vec2d endNode = indexToWorld(cell.cast<double>() - myCellOffset + edgeToNodeOffset[edgeIndex][1]);

		startPoints.push_back(startNode);
		endPoints.push_back(endNode);
	}

	renderer.addLines(startPoints, endPoints, colour);
}

template<typename T>
void ScalarGrid<T>::drawSamplePoints(Renderer& renderer, const Vec3d& colour, double sampleSize) const
{
	VecVec2d samplePoints;
	samplePoints.reserve(this->voxelCount());

	forEachVoxelRange(Vec2i::Zero(), this->size(), [&](const Vec2i& cell)
	{
		Vec2d worldPoint = indexToWorld(cell.cast<double>());
		samplePoints.push_back(worldPoint);
	});

	renderer.addPoints(samplePoints, colour, sampleSize);
}

template<typename T>
void ScalarGrid<T>::drawSupersampledValues(Renderer& renderer, double sampleRadius, int samples, double sampleSize) const
{
	std::pair<T, T> minMaxPair = minAndMaxValue();
	T minSample = minMaxPair.first;
	T maxSample = minMaxPair.second;

	forEachVoxelRange(Vec2i::Zero(), this->mySize, [&](const Vec2i& cell)
	{
		// Supersample
		double dx = 2. * sampleRadius / double(samples);
		Vec2d indexPoint = cell.cast<double>();
		Vec2d sampleOffset;
		for (sampleOffset[0] = -sampleRadius; sampleOffset[0] <= sampleRadius; sampleOffset[0] += dx)
			for (sampleOffset[1] = -sampleRadius; sampleOffset[1] <= sampleRadius; sampleOffset[1] += dx)
			{
				Vec2d samplePoint = indexPoint + sampleOffset;
				Vec2d worldPoint = indexToWorld(samplePoint);

				T value = (biLerp(worldPoint) - minSample) / (maxSample - minSample);

				renderer.addPoint(worldPoint, Vec3d(value, value, 0), sampleSize);
			}
	});
}

template<typename T>
void ScalarGrid<T>::drawSampleGradients(Renderer& renderer, const Vec3d& colour, double length) const
{
	VecVec2d samplePoints;
	VecVec2d gradientPoints;

	forEachVoxelRange(Vec2i::Zero(), this->mySize, [&](const Vec2i& cell)
	{
		Vec2d worldPoint = indexToWorld(cell.cast<double>());
		samplePoints.push_back(worldPoint);

		Vec2t<T> gradVector = biLerpGradient(worldPoint);
		Vec2d vectorEnd = worldPoint + length * gradVector;
		gradientPoints.push_back(vectorEnd);
	});

	renderer.addLines(samplePoints, gradientPoints, colour);
}

template<typename T>
void ScalarGrid<T>::drawVolumetric(Renderer& renderer, const Vec3d& minColour, const Vec3d& maxColour, T minVal, T maxVal) const
{
	ScalarGrid<double> nodes(xform(), this->mySize, SampleType::NODE);

	VecVec2d quadVertices(nodes.size()[0] * nodes.size()[1]);
	VecVec4i pixelQuads(this->mySize[0] * this->mySize[1]);
	VecVec3d colours(this->mySize[0] * this->mySize[1]);

	// Set node points
	forEachVoxelRange(Vec2i::Zero(), nodes.size(), [&](const Vec2i& node)
	{
		int index = nodes.flatten(node);
		quadVertices[index] = nodes.indexToWorld(node.cast<double>());
	});

	{
		int quadCount = 0;

		forEachVoxelRange(Vec2i::Zero(), this->mySize, [&](const Vec2i& cell)
		{
			Vec4i quad;

			for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
			{
				Vec2i node = cellToNodeCCW(cell, nodeIndex);
				quad[nodeIndex] = nodes.flatten(node);
			}

			T pixelValue = (*this)(cell);

			pixelQuads[quadCount] = quad;

			Vec3d colour;
			if (pixelValue > maxVal) colour = maxColour;
			else if (pixelValue < minVal) colour = minColour;
			else
			{
				double s = pixelValue / (maxVal - minVal);
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
		forEachVoxelRange(Vec2i::Zero(), this->mySize, [&](const Vec2i& cell)
		{
			Vec2d position = indexToWorld(cell.cast<double>());

			int flatIndex = this->flatten(cell);
			assert(this->unflatten(flatIndex) == cell);

			writer << "v " << position[0] << " " << (*this)(cell) << " " << position[1] << "#" << vertexCount << "\n";

			assert(flatIndex == vertexCount);
			++vertexCount;
		});

		assert(vertexCount == double(this->mySize[0] * this->mySize[1]));
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

		writer.close();
	}
	else
		std::cerr << "Failed to write to file: " << filename << std::endl;
}

}
#endif
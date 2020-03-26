#ifndef LIBRARY_LEVELSET_H
#define LIBRARY_LEVELSET_H

#include "EdgeMesh.h"
#include "FieldAdvector.h"
#include "Integrator.h"
#include "Predicates.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// LevelSet.h/cpp
// Ryan Goldade 2016
//
// 2d level set surface tracker.
// Uses a simple dense grid but with
// a narrow band for most purposes.
// Redistancing performs an interface
// search for nodes near the zero crossing
// and then fast marching to update the
// remaining grid (w.r.t. narrow band).
//
////////////////////////////////////

namespace FluidSim2D::SurfaceTrackers
{

static const int marchingSquaresTemplate[16][4] = { { -1,-1,-1,-1 },
													{ 3, 0,-1,-1 },
													{ 0, 1,-1,-1 },
													{ 3, 1,-1,-1 },
	
													{ 1, 2,-1,-1 },
													{ 3, 0, 1, 2 },
													{ 0, 2,-1,-1 },
													{ 3, 2,-1,-1 },
	
													{ 2, 3,-1,-1 },
													{ 2, 0,-1,-1 },
													{ 0, 1, 2, 3 },
													{ 2, 1,-1,-1 },
	
													{ 1, 3,-1,-1 },
													{ 1, 0,-1,-1 },
													{ 0, 3,-1,-1 },
													{ -1,-1,-1,-1 } };
	

using namespace RenderTools;
using namespace SimTools;
using namespace Utilities;

class LevelSet
{
public:
	LevelSet()
		: myNarrowBand(0)
		, myPhiGrid(Transform(0, Vec2f(0)), Vec2i(0))
		, myIsBackgroundNegative(false)
	{ exactinit(); }

	LevelSet(const Transform& xform, const Vec2i& size) : LevelSet(xform, size, int(size[0] * size[1])) {}
	LevelSet(const Transform& xform, const Vec2i& size, int bandwidth, bool isBoundaryNegative = false)
		: myNarrowBand(float(bandwidth) * xform.dx())
		, myPhiGrid(xform, size, isBoundaryNegative ? -float(bandwidth) * xform.dx() : float(bandwidth) * xform.dx())
		, myIsBackgroundNegative(isBoundaryNegative)
	{
		assert(size[0] >= 0 && size[1] >= 0);
		// In order to deal with triangle meshes, we need to initialize
		// the geometric predicate library.
		exactinit();
	}

	void initFromMesh(const EdgeMesh& initialMesh, bool doResizeGrid = true);

	void reinit(bool rebuildWithFIM = true);

	void reinitMesh(bool useMarchingSquares = false)
	{
		EdgeMesh tempMesh;
		if (useMarchingSquares) tempMesh = buildMSMesh();
		else tempMesh = buildDCMesh();
		initFromMesh(tempMesh, false);
	}

	bool isGridMatched(const LevelSet& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	bool isGridMatched(const ScalarGrid<float>& grid) const
	{
		assert(grid.sampleType() == ScalarGridSettings::SampleType::CENTER);
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	void unionSurface(const LevelSet& unionPhi);

	bool isBackgroundNegative() const { return myIsBackgroundNegative; }
	void setBackgroundNegative() { myIsBackgroundNegative = true; }

	EdgeMesh buildMSMesh() const;
	EdgeMesh buildDCMesh() const;

	template<typename VelocityField>
	void advect(float dt, const VelocityField& velocity, IntegrationOrder order);

	Vec2f normal(const Vec2f& worldPoint) const
	{
		Vec2f normal = myPhiGrid.gradient(worldPoint);

		if (normal == Vec2f(0)) return Vec2f(0);

		return normalize(normal);
	}

	void clear() { myPhiGrid.clear(); }
	void resize(const Vec2i& size) { myPhiGrid.resize(size); }

	float narrowBand() { return myNarrowBand / dx(); }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	float dx() const { return myPhiGrid.dx(); }
	Vec2f offset() const { return myPhiGrid.offset(); }
	Transform xform() const { return myPhiGrid.xform(); }
	Vec2i size() const { return myPhiGrid.size(); }

	Vec2f indexToWorld(const Vec2f& indexPoint) const { return myPhiGrid.indexToWorld(indexPoint); }
	Vec2f worldToIndex(const Vec2f& worldPoint) const { return myPhiGrid.worldToIndex(worldPoint); }

	float biLerp(const Vec2f& worldPoint) const { return myPhiGrid.biLerp(worldPoint); }

	float& operator()(int i, int j) { return myPhiGrid(i, j); }
	float& operator()(const Vec2i& cell) { return myPhiGrid(cell); }

	const float& operator()(int i, int j) const { return myPhiGrid(i, j); }
	const float& operator()(const Vec2i& cell) const { return myPhiGrid(cell); }

	int voxelCount() const { return myPhiGrid.voxelCount(); }
	Vec2i unflatten(int cellIndex) const { return myPhiGrid.unflatten(cellIndex); }

	Vec2f findSurface(const Vec2f& worldPoint, int iterationLimit) const;

	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec2f interpolateInterface(const Vec2i& startPoint, const Vec2i& endPoint) const;

	void drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const;
	void drawMeshGrid(Renderer& renderer) const;
	void drawSupersampledValues(Renderer& renderer, float radius = .5, int samples = 1, float sampleSize = 1.) const;
	void drawNormals(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), float length = .25) const;
	void drawSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.), float lineWidth = 1) const;
	void drawDCSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.), float lineWidth = 1) const;

private:

	void reinitFastMarching(UniformGrid<VisitedCellLabels>& interfaceCells);
	void reinitFastIterative(UniformGrid<VisitedCellLabels>& interfaceCells);

	Vec2f findSurfaceIndex(const Vec2f& indexPoint, int iterationLimit = 10) const;

	ScalarGrid<float> myPhiGrid;

	// The narrow band of signed distances around the interface
	float myNarrowBand;

	bool myIsBackgroundNegative;
};

template<typename VelocityField>
void LevelSet::advect(float dt, const VelocityField& velocity, IntegrationOrder order)
{
	ScalarGrid<float> tempPhiGrid = myPhiGrid;
	advectField(dt, tempPhiGrid, myPhiGrid, velocity, order, InterpolationOrder::CUBIC);

	//std::swap(tempPhiGrid, myPhiGrid);

	myPhiGrid = tempPhiGrid;
}

}
#endif
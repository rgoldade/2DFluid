#ifndef FLUIDSIM2D_LEVELSET_H
#define FLUIDSIM2D_LEVELSET_H

#include "EdgeMesh.h"
#include "FieldAdvector.h"
#include "Integrator.h"
#include "Predicates.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
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

namespace FluidSim2D
{

class LevelSet
{
public:
	LevelSet()
		: myNarrowBand(0)
		, myPhiGrid(Transform(0, Vec2d::Zero()), Vec2i::Zero())
		, myIsBackgroundNegative(false)
	{ exactinit(); }

	LevelSet(const Transform& xform, const Vec2i& size) : LevelSet(xform, size, double(size[0] * size[1])) {}
	LevelSet(const Transform& xform, const Vec2i& size, double bandwidth, bool isBoundaryNegative = false)
		: myNarrowBand(bandwidth * xform.dx())
		, myPhiGrid(xform, size, isBoundaryNegative ? -myNarrowBand : myNarrowBand)
		, myIsBackgroundNegative(isBoundaryNegative)
	{
		assert(size[0] >= 0 && size[1] >= 0);
		// In order to deal with triangle meshes, we need to initialize
		// the geometric predicate library.
		exactinit();
	}

	void initFromMesh(const EdgeMesh& initialMesh, bool doResizeGrid = true);

	void reinit(bool useMarchingSquares = false)
	{
		EdgeMesh tempMesh;
		if (useMarchingSquares) tempMesh = buildMSMesh();
		else tempMesh = buildDCMesh();
		initFromMeshImpl(tempMesh, false);
	}

	bool isGridMatched(const LevelSet& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	bool isGridMatched(const ScalarGrid<double>& grid) const
	{
		if (grid.sampleType() != ScalarGridSettings::SampleType::CENTER) return false;
		if ((size().array() != grid.size().array()).all()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	void unionSurface(const LevelSet& unionPhi);

	bool isBackgroundNegative() const { return myIsBackgroundNegative; }
	void setBackgroundNegative() { myIsBackgroundNegative = true; }

	EdgeMesh buildMSMesh() const;
	EdgeMesh buildDCMesh() const;

	template<typename VelocityField>
	void advect(double dt, const VelocityField& velocity, IntegrationOrder order);

	FORCE_INLINE Vec2d normal(const Vec2d& worldPoint, bool useLinearIntep = true) const
	{
		Vec2d normal;
		
		if (useLinearIntep)
		{
			normal = myPhiGrid.biLerpGradient(worldPoint);
		}
		else
		{
			normal = myPhiGrid.biCubicGradient(worldPoint);
		}

		if ((normal.array() == Vec2d::Zero().array()).all()) return Vec2d::Zero();

		return normal.normalized();
	}

	void clear() { myPhiGrid.clear(); }
	void resize(const Vec2i& size) { myPhiGrid.resize(size); }

	double narrowBand() { return myNarrowBand; }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	double dx() const { return myPhiGrid.dx(); }
	Vec2d offset() const { return myPhiGrid.offset(); }
	Transform xform() const { return myPhiGrid.xform(); }
	Vec2i size() const { return myPhiGrid.size(); }

	Vec2d indexToWorld(const Vec2d& indexPoint) const { return myPhiGrid.indexToWorld(indexPoint); }
	Vec2d worldToIndex(const Vec2d& worldPoint) const { return myPhiGrid.worldToIndex(worldPoint); }

	FORCE_INLINE double biLerp(const Vec2d& worldPoint) const
	{
		return myPhiGrid.biLerp(worldPoint);
	}
	
	FORCE_INLINE double biCubicInterp(const Vec2d& worldPoint) const
	{
		return myPhiGrid.biCubicInterp(worldPoint);
	}

	double& operator()(int i, int j) { return myPhiGrid(i, j); }
	double& operator()(const Vec2i& cell) { return myPhiGrid(cell); }

	const double& operator()(int i, int j) const { return myPhiGrid(i, j); }
	const double& operator()(const Vec2i& cell) const { return myPhiGrid(cell); }

	int voxelCount() const { return myPhiGrid.voxelCount(); }
	Vec2i unflatten(int cellIndex) const { return myPhiGrid.unflatten(cellIndex); }

	Vec2d findSurface(const Vec2d& worldPoint, int iterationLimit, double tolerance) const;

	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec2d interpolateInterface(const Vec2i& startPoint, const Vec2i& endPoint) const;

	void drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const;
	void drawMeshGrid(Renderer& renderer) const;
	void drawSupersampledValues(Renderer& renderer, double radius = .5, int samples = 1, double sampleSize = 1) const;
	void drawNormals(Renderer& renderer, const Vec3d& colour = Vec3d(0, 0, 1), double length = .25) const;
	void drawSurface(Renderer& renderer, const Vec3d& colour = Vec3d::Zero(), double lineWidth = 1) const;
	void drawDCSurface(Renderer& renderer, const Vec3d& colour = Vec3d::Zero(), double lineWidth = 1) const;

private:

	void initFromMeshImpl(const EdgeMesh& initialMesh, bool doResizeGrid);

	void reinitFastMarching(UniformGrid<VisitedCellLabels>& interfaceCells);

	Vec2d findSurfaceIndex(const Vec2d& indexPoint, int iterationLimit, double tolerance) const;

	// The narrow band of signed distances around the interface
	double myNarrowBand;

	bool myIsBackgroundNegative;

	ScalarGrid<double> myPhiGrid;
};

template<typename VelocityField>
void LevelSet::advect(double dt, const VelocityField& velocity, IntegrationOrder order)
{
	ScalarGrid<double> tempPhiGrid = myPhiGrid;
	advectField(dt, tempPhiGrid, myPhiGrid, velocity, order, InterpolationOrder::CUBIC);

	std::swap(tempPhiGrid, myPhiGrid);
}

}
#endif
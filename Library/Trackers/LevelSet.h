#ifndef LIBRARY_LEVELSET_H
#define LIBRARY_LEVELSET_H

#include "AdvectField.h"
#include "Common.h"
#include "EdgeMesh.h"
#include "Integrator.h"
#include "Predicates.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
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

class LevelSet
{
public:
	LevelSet() : myPhiGrid() {}

	LevelSet(const Transform& xform, const Vec2i& size) : LevelSet(xform, size, int(size[0] * size[1])) {}
	LevelSet(const Transform& xform, const Vec2i& size, int bandwidth, bool isBoundaryNegative = false)
		: myNarrowBand(Real(bandwidth) * xform.dx())
		, myPhiGrid(xform, size, isBoundaryNegative ? -Real(bandwidth) * xform.dx() : Real(bandwidth) * xform.dx())
		, myIsBackgroundNegative(isBoundaryNegative)
	{
		assert(size[0] >= 0 && size[1] >= 0);
		// In order to deal with triangle meshes, we need to initialize
		// the geometric predicate library.
		exactinit();
	}

	void initFromMesh(const EdgeMesh& initialMesh, bool resizeGrid = true);
	
	void reinit();
	void reinitFIM();
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

	bool isGridMatched(const ScalarGrid<Real>& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	void unionSurface(const LevelSet& unionPhi);

	// Assume negative ambient outside distance
	bool isBackgroundNegative() const { return myIsBackgroundNegative; }
	void setBackgroundNegative() { myIsBackgroundNegative = true; }

	EdgeMesh buildMSMesh() const;
	EdgeMesh buildDCMesh() const;

	template<typename VelocityField>
	void advect(Real dt, const VelocityField& velocity, IntegrationOrder order);

	Vec2R normal(const Vec2R& worldPoint) const
	{
		Vec2R normal = myPhiGrid.gradient(worldPoint);
		
		if (normal == Vec2R(0)) return Vec2R(0);

		return normalize(normal);
	}

	void clear() { myPhiGrid.clear(); }
	void resize(const Vec2i& size) { myPhiGrid.resize(size); }

	Real narrowBand() { return myNarrowBand / dx(); }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	Real dx() const { return myPhiGrid.dx(); }
	Vec2R offset() const { return myPhiGrid.offset(); }
	Transform xform() const { return myPhiGrid.xform(); }
	Vec2i size() const { return myPhiGrid.size(); }

	Vec2R indexToWorld(const Vec2R& indexPoint) const { return myPhiGrid.indexToWorld(indexPoint); }
	Vec2R worldToIndex(const Vec2R& worldPoint) const { return myPhiGrid.worldToIndex(worldPoint); }
	
	Real interp(const Vec2R& worldPoint) const { return myPhiGrid.interp(worldPoint); }

	Real& operator()(int i, int j) { return myPhiGrid(i, j); }
	Real& operator()(const Vec2i& cell) { return myPhiGrid(cell); }

	const Real& operator()(int i, int j) const { return myPhiGrid(i, j); }
	const Real& operator()(const Vec2i& cell) const { return myPhiGrid(cell); }

	Vec2i unflatten(int cellIndex) const { return myPhiGrid.unflatten(cellIndex); }

	Vec2R findSurface(const Vec2R& worldPoint, int iterationLimit) const;
	
	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec2R interpolateInterface(const Vec2i& startPoint, const Vec2i& endPoint) const;

	void drawGrid(Renderer& renderer) const;
	void drawMeshGrid(Renderer& renderer) const;
	void drawSupersampledValues(Renderer& renderer, Real radius = .5, 
									int samples = 1, Real sampleSize = 1.) const;
	void drawNormals(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;
	void drawSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.), Real lineWidth = 1) const;
	void drawDCSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.), Real lineWidth = 1) const;

private:

	void reinitFastMarching(UniformGrid<MarkedCells>& interfaceCells);
	void reinitFastIterative(UniformGrid<MarkedCells>& interfaceCells);

	Vec2R findSurfaceIndex(const Vec2R& indexPoint, int iterationLimit = 10) const;

	ScalarGrid<Real> myPhiGrid;

	// The narrow band of signed distances around the interface
	Real myNarrowBand;

	bool myIsBackgroundNegative;
};

template<typename VelocityField>
void LevelSet::advect(Real dt, const VelocityField& velocity, IntegrationOrder order)
{
	AdvectField<ScalarGrid<Real>> advector(myPhiGrid);

	ScalarGrid<Real> tempPhiGrid = myPhiGrid;
	advector.advectField(dt, tempPhiGrid, velocity, order, InterpolationOrder::CUBIC);

	std::swap(tempPhiGrid, myPhiGrid);
}

#endif
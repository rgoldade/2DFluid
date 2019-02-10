#ifndef LIBRARY_LEVELSET2D_H
#define LIBRARY_LEVELSET2D_H

#include "AdvectField.h"
#include "Common.h"

#include "Integrator.h"

#include "Mesh2D.h"
#include "Predicates.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// LevelSet2d.h/cpp
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

class LevelSet2D
{
public:
	LevelSet2D() : myPhiGrid() {}

	LevelSet2D(const Transform& xform, const Vec2ui& size) : LevelSet2D(xform, size, unsigned(size[0] * size[1])) {}
	LevelSet2D(const Transform& xform, const Vec2ui& size, unsigned bandwidth, bool inverted = false)
		: myNarrowBand(Real(bandwidth) * xform.dx())
		, myPhiGrid(xform, size, inverted ? -Real(bandwidth) * xform.dx() : Real(bandwidth) * xform.dx())
		, myIsInverted(inverted)
	{
		// In order to deal with triangle meshes, we need to initialize
		// the geometric predicate library.
		exactinit();
	}

	void init(const Mesh2D& init_mesh, bool resize = true);
	
	void reinit();
	void reinitFIM();
	void reinitMesh(bool useMarchingSquares = false)
	{
		Mesh2D tempMesh;
		if (useMarchingSquares) tempMesh = buildMSMesh();
		else tempMesh = buildDCMesh();
		init(tempMesh, false);
	}

	bool isMatched(const LevelSet2D& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	bool isMatched(const ScalarGrid<Real>& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	void unionSurface(const LevelSet2D& unionPhi);

	// Assume negative ambient outside distance
	bool inverted() const { return myIsInverted; }
	void setInverted() { myIsInverted = true; }

	Mesh2D buildMSMesh() const;
	Mesh2D buildDCMesh() const;

	template<typename VelocityField>
	void advect(Real dt, const VelocityField& vel, IntegrationOrder order);

	Vec2R normal(const Vec2R& worldPoint) const
	{
		Vec2R normal = myPhiGrid.gradient(worldPoint);
		
		if (normal == Vec2R(0)) return Vec2R(0);

		return normalize(normal);
	}

	void clear() { myPhiGrid.clear(); }
	void resize(const Vec2ui& size) { myPhiGrid.resize(size); }

	Real narrowBand() { return myNarrowBand / dx(); }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	Real dx() const { return myPhiGrid.dx(); }
	Vec2R offset() const { return myPhiGrid.offset(); }
	Transform xform() const { return myPhiGrid.xform(); }
	Vec2ui size() const { return myPhiGrid.size(); }

	Vec2R indexToWorld(const Vec2R& indexPoint) const { return myPhiGrid.indexToWorld(indexPoint); }
	Vec2R worldToIndex(const Vec2R& worldPoint) const { return myPhiGrid.worldToIndex(worldPoint); }
	
	Real interp(const Vec2R& worldPoint) const { return myPhiGrid.interp(worldPoint); }

	Real& operator()(unsigned i, unsigned j) { return myPhiGrid(i, j); }
	Real& operator()(const Vec2ui& coord) { return myPhiGrid(coord); }

	const Real& operator()(unsigned i, unsigned j) const { return myPhiGrid(i, j); }
	const Real& operator()(const Vec2ui& coord) const { return myPhiGrid(coord); }

	Vec2R findSurface(const Vec2R& worldPoint, unsigned iterationLimit) const;
	
	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec2R interpolateInterface(const Vec2ui& startPoint, const Vec2ui& endPoint) const;

	void drawGrid(Renderer& renderer) const;
	void drawMeshGrid(Renderer& renderer) const;
	void drawSupersampledValues(Renderer& renderer, Real radius = .5, 
									unsigned samples = 1, unsigned size = 1) const;
	void drawNormals(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;
	void drawSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.));
	void drawDCSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.));

private:

	void reinitFastMarching(UniformGrid<MarkedCells>& interfaceCells);
	void reinitFastIterative(UniformGrid<MarkedCells>& interfaceCells);

	Vec2R findSurfaceIndex(const Vec2R& indexPoint, unsigned iterationLimit = 10) const;

	ScalarGrid<Real> myPhiGrid;

	// The narrow band of signed distances around the interface
	Real myNarrowBand;

	bool myIsInverted;
};

template<typename VelocityField>
void LevelSet2D::advect(Real dt, const VelocityField& vel, IntegrationOrder order)
{
	AdvectField<ScalarGrid<Real>> advector(myPhiGrid);

	ScalarGrid<Real> tempPhiGrid = myPhiGrid;
	advector.advectField(dt, tempPhiGrid, vel, order, InterpolationOrder::CUBIC);

	std::swap(tempPhiGrid, myPhiGrid);
}

#endif
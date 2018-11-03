#pragma once

#include "Common.h"

#include "Predicates.h"

#include "Mesh2D.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "Renderer.h"
#include "Integrator.h"
#include "Transform.h"

#include "AdvectField.h"

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
	LevelSet2D() : m_phi() {}

	LevelSet2D(const Transform& xform, const Vec2ui& size) : LevelSet2D(xform, size, Real(size[0] * size[1])) {}
	LevelSet2D(const Transform& xform, const Vec2ui& size, Real bandwidth, bool inverted = false)
		: m_narrow_band((Real)bandwidth * xform.dx())
		, m_phi(xform, size, inverted ? (Real)bandwidth * xform.dx() : -(Real)bandwidth * xform.dx())
		, m_inverted(inverted)
	{
		// In order to deal with triangle meshes, we need to initialize
		// the geometric predicate library.
		exactinit();
	}

	void init(const Mesh2D& init_mesh, bool resize = true);
	
	void reinit();
	void reinitFIM();

	bool is_matched(const LevelSet2D& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	bool is_matched(const ScalarGrid<Real>& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	void surface_union(const LevelSet2D& inputVolume);

	// Assume negative ambient outside distance
	bool inverted() const { return m_inverted; }
	void set_inverted() { m_inverted = true; }

	void extract_mesh(Mesh2D& surf) const;
	void extract_dc_mesh(Mesh2D& surf) const;

	template<typename VelocityField>
	void advect(Real dt, const VelocityField& vel, IntegrationOrder order);

	Vec2R normal(const Vec2R& world_pos) const
	{
		Vec2R normal = m_phi.gradient(world_pos);
		
		if (normal == Vec2R(0)) return Vec2R(0);

		return normalized(m_phi.gradient(world_pos));
	}

	void clear() { m_phi.clear(); }
	void resize(const Vec2ui& size) { m_phi.resize(size); }

	Real narrow_band() { return (Real)(m_narrow_band / dx()); }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	Real dx() const { return m_phi.dx(); }
	Vec2R offset() const { return m_phi.offset(); }
	Transform xform() const { return m_phi.xform(); }
	Vec2ui size() const { return m_phi.size(); }

	Vec2R idx_to_ws(const Vec2R& idx_pos) const { return m_phi.idx_to_ws(idx_pos); }
	Vec2R ws_to_idx(const Vec2R& world_pos) const { return m_phi.ws_to_idx(world_pos); }
	
	Real interp(const Vec2R& world_pos) const { return m_phi.interp(world_pos); }

	Real& operator()(unsigned i, unsigned j) { return m_phi(i, j); }
	Real& operator()(const Vec2ui& coord) { return m_phi(coord); }

	const Real& operator()(unsigned i, unsigned j) const { return m_phi(i, j); }
	const Real& operator()(const Vec2ui& coord) const { return m_phi(coord); }

	Vec2R interface_search(const Vec2R& world_pos, unsigned iterLimit) const;
	
	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec2R interp_interface(const Vec2ui& i0, const Vec2ui& i1) const;

	void draw_grid(Renderer& renderer) const;
	void draw_mesh_grid(Renderer& renderer) const;
	void draw_supersampled_values(Renderer& renderer, Real radius = .5, 
									unsigned samples = 1, unsigned size = 1) const;
	void draw_normals(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;
	void draw_surface(Renderer& renderer, const Vec3f& colour = Vec3f(0.));
	void draw_dc_surface(Renderer& renderer, const Vec3f& colour = Vec3f(0.));

private:

	void fast_marching(UniformGrid<MarkedCells>& markedCells);
	void fast_iterative(UniformGrid<MarkedCells>& markedCells);

	Vec2R interface_search_idx(const Vec2R& ipos, unsigned iterLimit = 10) const;

	ScalarGrid<Real> m_phi;

	// The narrow band of signed distances around the interface
	Real m_narrow_band;

	bool m_inverted;
};

template<typename VelocityField>
void LevelSet2D::advect(Real dt, const VelocityField& vel, IntegrationOrder order)
{
	AdvectField<ScalarGrid<Real>> advector(m_phi);

	ScalarGrid<Real> temp_phi = m_phi;
	advector.advect_field(dt, temp_phi, vel, order);

	std::swap(temp_phi, m_phi);
}
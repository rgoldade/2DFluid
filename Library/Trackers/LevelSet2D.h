#pragma once

#include "Core.h"
#include "Vec.h"

#include "Predicates.h"

#include "Mesh2D.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "Renderer.h"

#include "Transform.h"

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
	LevelSet2D()
		: m_phi()
		, m_dirty_surface(false)
		, m_mesh_set(false)
		, m_gradient_set(false)
		, m_curvature_set(false) {}

	LevelSet2D(const Transform& xform, const Vec2st& nx) : LevelSet2D(xform, nx, nx[0] * nx[1]) {}
	LevelSet2D(const Transform& xform, const Vec2st& nx, size_t narrow_band, bool inverted = false, Real background = 0)
		: m_phi(xform, nx, background)
		, m_nb((Real)narrow_band * xform.dx())
		, m_dirty_surface(false)
		, m_mesh_set(false)
		, m_gradient_set(false)
		, m_curvature_set(false)
		, m_inverted(inverted)
	{
		exactinit();
	}

	LevelSet2D(const LevelSet2D &g)
	{
		m_phi = g.m_phi;
		m_nb = g.m_nb;

		m_dirty_surface = g.m_dirty_surface;
		m_mesh_set = g.m_mesh_set;
		m_gradient_set = g.m_gradient_set;
		m_curvature_set = g.m_curvature_set;

		m_surface = g.m_surface;
		m_gradient = g.m_gradient;
		m_curvature = g.m_curvature;
		m_inverted = g.m_inverted;
	}

	LevelSet2D& operator=(const LevelSet2D& g)
	{
		m_phi = g.m_phi;
		m_nb = g.m_nb;

		m_dirty_surface = g.m_dirty_surface;
		m_mesh_set = g.m_mesh_set;
		m_gradient_set = g.m_gradient_set;
		m_curvature_set = g.m_curvature_set;
		
		m_surface = g.m_surface;
		m_gradient = g.m_gradient;
		m_curvature = g.m_curvature;
		m_inverted = g.m_inverted;

		return *this;
	}

	void init(const Mesh2D& init_mesh, bool resize = true);
	
	inline bool is_matched(const LevelSet2D& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	inline bool is_matched(const ScalarGrid<Real>& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	// Each loop reconstructs the gradient, leading to more accurate SDFs
	void reinit(size_t iters = 2);
	void reinitFIM(bool force_rebuild = false);

	bool is_dirtied() { return m_dirty_surface; }

	void surface_union(const LevelSet2D& input_volume);

	// Assume negative ambient outside distance
	bool inverted() const { return m_inverted; }
	void set_inverted() { m_inverted = true; }
	//void sample_grid(const LevelSet2D& sphi);
	
	void extract_mesh(Mesh2D& surf) const;

	void extract_dc_mesh(Mesh2D& surf);

	template<typename VelField, typename Integrator>
	void backtrace_advect(Real dt, const VelField& vel, const Integrator& f);

	// Normals need to be in world space at the moment due to the complexity of index spaces
	// with staggered grids. This needs to be fixed in the vector grid class
	inline Vec2R normal(const Vec2R& wpos)
	{
		if (!m_gradient_set) build_gradient();
		return normal_const(wpos);
	}

	inline Vec2R normal_const(const Vec2R& wpos) const
	{
		Vec2R grad;
		if (m_gradient_set)
			grad = m_gradient.interp(wpos);
		else
			grad = m_phi.gradient(wpos);

		if (fabs(grad[0]) < 1E-6 && fabs(grad[1]) < 1E-6) return Vec2R(0.);
		else normalize(grad);
		return grad;
	}

	void gradient_field(VectorGrid<Real>& grad);
	
	void build_gradient();

	bool curvature_field(ScalarGrid<Real>& grad);

	void build_curvature();

	const ScalarGrid<Real>& get_curvature()
	{
		build_curvature();
		return m_curvature;
	}

	inline void clear() { m_phi.clear(); m_dirty_surface = true; }

	// TODO: I can resize.. but not really do much else
	inline void resize(const Vec2st& nx) { m_phi.resize(nx); m_dirty_surface = true; }

	inline size_t narrow_band() { return (size_t)(m_nb / dx()); }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	inline Real dx() const { return m_phi.dx(); }
	inline Vec2R offset() const { return m_phi.offset(); }
	inline Transform xform() const { return m_phi.xform(); }
	inline Vec2st size() const { return m_phi.size(); }

	inline Vec2R idx_to_ws(const Vec2R& ipos) const { return m_phi.idx_to_ws(ipos); }
	inline Vec2R ws_to_idx(const Vec2R& ipos) const { return m_phi.ws_to_idx(ipos); }

	inline Real phi(size_t x, size_t y) const { return phi(Vec2st(x, y)); }
	inline Real phi(const Vec2st& coord) const { return m_phi(coord[0], coord[1]); }
	inline Real phi(const Vec2R& wpos) const { return m_phi.interp(wpos); }
	inline void set_phi(const Vec2st& coord, Real val) { m_phi(coord[0], coord[1]) = val; m_dirty_surface = true; }
	
	inline Real interp(const Vec2R& wpos) const { return phi(wpos); } // TODO: make calls uniform

	// Operator overloading to be consistent with ScalarGrid and VectorGrid
	//inline Real& operator()(int x, int y) { return m_phi(x, y); }
	inline const Real& operator()(int x, int y) const { return m_phi(x, y); }

	Vec2R interface_search(const Vec2R wpos, size_t iter_limit = 10) const;
	
	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec2R interp_interface(const Vec2st& i0, const Vec2st& i1) const;

	void draw_grid(Renderer& renderer) const;
	void draw_mesh_grid(Renderer& renderer) const;
	void draw_supersampled_values(Renderer& renderer, Real radius = .5, size_t samples = 1, size_t size = 1) const;
	void draw_normals(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;
	void draw_surface(Renderer& renderer, const Vec3f& colour = Vec3f(0.));
	void draw_dc_surface(Renderer& renderer, const Vec3f& colour = Vec3f(0.));

	void fast_marching(UniformGrid<marked>& marked_cells);
	void fast_iterative(UniformGrid<marked>& marked_cells, std::vector<Vec2st> &active_list);

private:

	Vec2R interface_search_idx(const Vec2R& ipos, size_t iter_limit = 10) const;

	ScalarGrid<Real> m_phi;
	VectorGrid<Real> m_gradient;
	ScalarGrid<Real> m_curvature;

	// The narrow band of signed distances around the interface
	Real m_nb;

	bool m_dirty_surface, m_mesh_set, m_gradient_set, m_curvature_set;
	Mesh2D m_surface;
	bool m_inverted;
};

// Backtracing assumes it's provided with a positive timestep and handle backtracing itself.
// Backtracing will only be performed within the narrow band
template<typename VelField, typename Integrator>
void LevelSet2D::backtrace_advect(Real dt, const VelField& vel, const Integrator& f)
{
	assert(dt >= 0);

	Real length = 0;
	for (size_t x = 0; x < size()[0]; ++x)
		for (size_t y = 0; y < size()[1]; ++y)
		{
			if (fabs(m_phi(x, y)) < m_nb)
			{
				Vec2R pos = idx_to_ws(Vec2R(x, y));
				
				Vec2R velvec = vel(dt, pos);
				Real velmag = mag(velvec);
				if (velmag > length) length = velmag;
			}
		}

	Real time = 0;
	Real cfl = (m_nb / 2.) / (length + 1E-5);
	if (cfl > dt) cfl = dt;
	while (time < dt)
	{
		ScalarGrid<Real> temp_phi = m_phi;

		for (size_t x = 0; x < size()[0]; ++x)
			for (size_t y = 0; y < size()[1]; ++y)
			{
				if (fabs(m_phi(x, y)) < m_nb)
				{
					Vec2R pos = idx_to_ws(Vec2R(x, y));
					pos = f(pos, -cfl, vel);
					temp_phi(x, y) = m_phi.interp(pos);
				}
			}
		time += cfl;
		m_phi = temp_phi;
	}

	m_mesh_set = false;
	m_dirty_surface = true;
	m_gradient_set = false;
	m_curvature_set = false;
}

#pragma once

#include "core.h"
#include "util.h"

#include "UniformGrid.h"
#include "Transform.h"

#include "Renderer.h"

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

// TODO: add harmonic borders
namespace ScalarGridSettings
{
	enum BorderType { CLAMP, ZERO, ASSERT };
	enum ScalarSampleType { CENTER, XFACE, YFACE, NODE };
}

using namespace ScalarGridSettings;
template<typename T>
class ScalarGrid : public UniformGrid<T>
{
	typedef ScalarGridSettings::BorderType BorderType;
	typedef ScalarGridSettings::ScalarSampleType SampleType;

public:

	ScalarGrid() : m_xform(1.,Vec2R(0.)), m_full_nx(Vec2st(0)), UniformGrid<T>() {}

	ScalarGrid(const Transform& xform, const Vec2st& nx,
				SampleType stype = CENTER, BorderType btype = CLAMP)
		: ScalarGrid(xform, nx, T(0), stype, btype)
	{}
		
	// The grid size (nx) is the number of actual grid cells to be created. This means that a 2x2 grid
	// will have 3x3 nodes, 3x2 x-aligned faces, 2x3 y-aligned faces and 2x2 cell centers. The size of 
	// the underlying storage container is reflected accordingly based the sample type to give the outside
	// caller the structure of a real grid.
	ScalarGrid(const Transform& xform, const Vec2st& nx, const T& val,
				SampleType stype = CENTER, BorderType btype = CLAMP)
		: m_xform(xform)
		, m_stype(stype)
		, m_btype(btype)
		, m_full_nx(nx)
	{
		switch (stype)
		{
		case CENTER:
			m_cell_offset = Vec2R(.5);
			resize(nx, val);
			break;
		case XFACE:
			m_cell_offset = Vec2R(.0, .5);
			resize(nx + Vec2st(1, 0), val);
			break;
		case YFACE:
			m_cell_offset = Vec2R(.5, .0);
			resize(nx + Vec2st(0, 1), val);
			break;
		case NODE:
			m_cell_offset = Vec2R(.0);
			resize(nx + Vec2st(1), val);
		}
	}

	// TODO: write a method to copy in grids with different sample types

	// TODO: add ability to use different bordertypes for copy constructor
	ScalarGrid(const ScalarGrid& g) : UniformGrid<T>(g)
	{
		m_xform = g.m_xform;
		m_cell_offset = g.m_cell_offset;
		m_stype = g.m_stype;
		m_btype = g.m_btype;
		m_full_nx = g.m_full_nx;
	}
	
	ScalarGrid& operator=(const ScalarGrid& g)
	{
		UniformGrid<T>::operator=(g);

		m_xform = g.m_xform;
		m_cell_offset = g.m_cell_offset;
		m_stype = g.m_stype;
		m_btype = g.m_btype;
		m_full_nx = g.m_full_nx;
		
		return *this;
	}

	SampleType sample_type() { return m_stype; }

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	inline bool is_matched(const ScalarGrid<S>& grid) const
	{
		if (size() != grid.size()) return false;
		if (m_xform != grid.xform()) return false;
		if (m_stype != grid.sample_type()) return false;
		return true;
	}

	// Global mult operator
	void operator*(const T& s)
	{
		for (auto& i : m_grid) i *= s;
	}

	// Global add operator
	void operator+(T s)
	{
		for (auto& i : m_grid) i += s;
	}

	T maxval() const { auto result = std::max_element(m_grid.begin(), m_grid.end()); return *result; }
	T minval() const { auto result = std::min_element(m_grid.begin(), m_grid.end()); return *result; }

	void minmax(T& min, T& max) const
	{
		auto result = std::minmax_element(m_grid.begin(), m_grid.end());
		min = *(result.first); max = *(result.second);
	}

	T interp(Real x, Real y, bool idx_space = false) const { return interp(Vec2R(x, y), idx_space); }
	T interp(const Vec2R& pos, bool idx_space = false) const;
	
	T cubic_interp(Real x, Real y, bool idx_space = false) const { return cubic_interp(Vec2R(x, y), idx_space); }
	T cubic_interp(const Vec2R& pos, bool idx_space = false) const;

	// Converters between world space and local index space
	inline Vec2R idx_to_ws(const Vec2R& ipos) const
	{
		return m_xform.idx_to_ws(ipos + m_cell_offset);
	}
	
	inline Vec2R ws_to_idx(const Vec2R& wpos) const
	{
		return m_xform.ws_to_idx(wpos) - m_cell_offset;
	}

	// Gradient operators
	Vec<2, T> gradient(const Vec2R& wpos, bool idx_space = false) const
	{
		Real offset = idx_space ? 1E-1 : 1E-1 * dx();
		T dTdx = interp(wpos + Vec2R(offset, 0.), idx_space) - interp(wpos - Vec2R(offset, 0.), idx_space);
		T dTdy = interp(wpos + Vec2R(0., offset), idx_space) - interp(wpos - Vec2R(0., offset), idx_space);
		Vec<2, T> grad(dTdx, dTdy);
		return grad / (2 * offset);
	}
	
	inline Real dx() const { return m_xform.dx(); }
	inline Vec2R offset() const { return m_xform.offset(); }
	inline Transform xform() const { return m_xform; }

	inline SampleType sample_type() const { return m_stype; }
	
	// Render methods
	void draw_grid(Renderer& renderer) const;
	void draw_sample_points(Renderer& renderer, const Vec3f& colour = Vec3f(1,0,0), Real size = 1.) const;
	void draw_supersampled_values(Renderer& renderer, Real radius = .5, size_t samples = 5, size_t size = 1) const;
	void draw_sample_gradients(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;

private:

	// The main interpolation call after the template specialized clamping passes
	inline T interp_local(const Vec2R& pos) const;

	// Store the actual grid size. The m_nx member variable represents the 
	// underlying sample grid. The actual grid doesn't change based on sample
	// type but the underlying array of sample points do.
	Vec2st m_full_nx;

	// The transform accounts for the grid spacings and the grid offset.
	// It includes transforms to and from index space.
	// The offset specifies the location of the lowest, left-most corner
	// of the grid. The actual sample point is offset (in index space)
	// from this point based on the SampleType
	Transform m_xform;
	
	SampleType m_stype;
	BorderType m_btype;

	// The local offset (in index space) associated with the sample type
	Vec2R m_cell_offset;
};

// Catmull-Rom cubic interpolation. Shamelessly lifted from MantaFlow (mantaflow.com)
template<typename T>
static inline T CINT(T p_1, T p0, T p1, T p2, T x)
{
	T x2 = x * x, x3 = x2 * x;
	return 0.5 *((-x3 + 2.0 * x2 - x) * p_1 + (3.0 * x3 - 5.0 * x2 + 2.0) * p0 + (-3.0 * x3 + 4.0 * x2 + x) * p1 + (x3 - x2) * p2);
};

template<typename T>
T ScalarGrid<T>::cubic_interp(const Vec2R& pos, bool idx_space) const
{
	Vec2R ipos = idx_space ? pos : ws_to_idx(pos);

	Vec2i p11 = floor(ipos);
	// Revert to basic interpolation near the boundaries
	if (p11[0] < 1 || p11[0] > size()[0] - 1 ||
		p11[1] < 1 || p11[1] > size()[1] - 1)
			return interp(pos, true);

	Vec2R u = ipos - Vec2R(p11);
	
	T interp_x[4];
	for (int off_y = -1; off_y <= 2; ++off_y)
	{
		Real yi = p11[1] + off_y;
		T p_1 = (*this)(p11[0] - 1,	yi);
		T p0 =	(*this)(p11[0],		yi);
		T p1 =	(*this)(p11[0] + 1,	yi);
		T p2 =	(*this)(p11[0] + 2,	yi);
		interp_x[off_y + 1] = CINT(p_1, p0, p1, p2, u[0]);
	}

	T p_1 = interp_x[0];
	T p0 = interp_x[1];
	T p1 = interp_x[2];
	T p2 = interp_x[3];

	return CINT(p_1, p0, p1, p2, u[1]);
}

template<typename T>
T ScalarGrid<T>::interp(const Vec2R& pos, bool idx_space) const
{
	Vec2R ipos = idx_space ? pos : ws_to_idx(pos);

	switch (m_btype)
	{
	case ZERO:
		if ((ipos[0] < 0.) || (ipos[1] < 0.) ||
			(ipos[0] > (Real)(m_nx[0] - 1)) ||
			(ipos[1] > (Real)(m_nx[1] - 1))) return T(0.);
	case CLAMP:
		ipos[0] = (ipos[0] < 0.) ? 0. : ((ipos[0] > (Real)(m_nx[0] - 1)) ? (Real)(m_nx[0] - 1) : ipos[0]);
		ipos[1] = (ipos[1] < 0.) ? 0. : ((ipos[1] > (Real)(m_nx[1] - 1)) ? (Real)(m_nx[1] - 1) : ipos[1]);
		break;
	// Useful debug check. Equivalent to "NONE" for release mode
	case ASSERT:
		assert((ipos[0] >= 0.) && (ipos[1] >= 0.) &&
					(ipos[0] <= (Real)(m_nx[0] - 1)) &&
					(ipos[1] <= (Real)(m_nx[1] - 1)));
		break;
	}

	return interp_local(ipos);
}

// The local interp applies bi-linear interpolation on the UniformGrid. The 
// templated type must have operators for basic add/mult arithmetic.
template<typename T>
T ScalarGrid<T>::interp_local(const Vec2R& ipos) const
{
	int xf = floor(ipos[0]);
	int yf = floor(ipos[1]);

	// TODO: should this actually be floor(ipos[0]+1) ?
	int xc = ceil(ipos[0]);
	int yc = ceil(ipos[1]);

	// This should always pass since the clamp test in the public interp
	// will not allow anything outside of the border to get this far.
	assert(xf >= 0 && yf >= 0);
	assert(xc < m_nx[0] && yf < m_nx[1]);

	// Use base grid class operator
	T v00 = (*this)(xf, yf);
	T v10 = (*this)(xc, yf);

	T v01 = (*this)(xf, yc);
	T v11 = (*this)(xc, yc);

	Real sx = ipos[0] - (Real)xf;
	Real sy = ipos[1] - (Real)yf;

	auto lerp = [](const T& val0, const T& val1, Real f){ return val0 + f * (val1 - val0); };

	T interp_val = lerp(lerp(v00, v10, sx), lerp(v01, v11, sx), sy);
	return interp_val;
}

template<typename T>
void ScalarGrid<T>::draw_grid(Renderer& renderer) const
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (int i = 0; i <= m_full_nx[0]; ++i)
	{
		// Offset backwards because we want the start of the grid cell
		Vec2R pos = idx_to_ws(Vec2R(i, 0) - m_cell_offset);
		start_points.push_back(pos);

		pos = idx_to_ws(Vec2R(i, m_full_nx[1]) - m_cell_offset);
		end_points.push_back(pos);
	}

	for (int j = 0; j <= m_full_nx[1]; ++j)
	{
		Vec2R pos = idx_to_ws(Vec2R(0, j) - m_cell_offset);
		start_points.push_back(pos);

		pos = idx_to_ws(Vec2R(m_full_nx[0], j) - m_cell_offset);
		end_points.push_back(pos);
	}

	renderer.add_lines(start_points, end_points, Vec3f(0));
}

template<typename T>
void ScalarGrid<T>::draw_sample_points(Renderer& renderer, const Vec3f& colour, Real size) const
{
	std::vector<Vec2R> sample_points;

	for (int i = 0; i < m_nx[0]; ++i)
		for (int j = 0; j < m_nx[1]; ++j)
		{
			Vec2R ipos = Vec2R(i, j);
			Vec2R wpos = idx_to_ws(ipos);

			sample_points.push_back(wpos);
		}

	renderer.add_points(sample_points, colour, size);
}

// Warning: there is no protection here for ASSERT border types
template<typename T>
void ScalarGrid<T>::draw_supersampled_values(Renderer& renderer, Real radius, size_t samples, size_t size) const
{
	std::vector<Vec2R> sample_points;

	T max_sample = maxval();
	T min_sample = minval();

	for (int i = 0; i < m_nx[0]; ++i)
		for (int j = 0; j < m_nx[1]; ++j)
		{
			Vec2R ipos = Vec2R(i, j);

			// Supersample
			Real dx = 2 * radius / (Real)samples;
			for (Real x = -radius; x <= radius; x += dx)
				for (Real y = -radius; y <= radius; y += dx)
				{
					Vec2R ss_pos = ipos + Vec2R(x, y);
					Vec2R wpos = idx_to_ws(ss_pos);

					T sampleval = (interp(wpos) - min_sample) / max_sample;
					renderer.add_point(wpos, Vec3f((float)sampleval, (float)sampleval, 0), size);
				}	
		}
}

template<typename T>
void ScalarGrid<T>::draw_sample_gradients(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1),
											Real length = .25) const
{
	std::vector<Vec2R> sample_points;
	std::vector<Vec2R> gradient_points;
	for (int i = 0; i < m_nx[0]; ++i)
		for (int j = 0; j < m_nx[1]; ++j)
		{
			Vec2R ipos = Vec2R(i, j);
			Vec2R wpos = idx_to_ws(ipos);
			sample_points.push_back(wpos);

			Vec<2, T> grad = gradient(wpos);
			Vec2R gpos = wpos + length * grad;
			gradient_points.push_back(gpos);
		}
	renderer.add_lines(sample_points, gradient_points, colour);
}


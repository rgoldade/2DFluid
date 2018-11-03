#pragma once

#include <algorithm>

#include "Common.h"

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

	ScalarGrid() : m_xform(1.,Vec2R(0.)), m_full_size(Vec2ui(0)), UniformGrid<T>() {}

	ScalarGrid(const Transform& xform, const Vec2ui& size,
				SampleType stype = SampleType::CENTER, BorderType btype = BorderType::CLAMP)
		: ScalarGrid(xform, size, T(0), stype, btype)
	{}
		
	// The grid size is the number of actual grid cells to be created. This means that a 2x2 grid
	// will have 3x3 nodes, 3x2 x-aligned faces, 2x3 y-aligned faces and 2x2 cell centers. The size of 
	// the underlying storage container is reflected accordingly based the sample type to give the outside
	// caller the structure of a real grid.
	ScalarGrid(const Transform& xform, const Vec2ui& size, const T& val,
				SampleType stype = SampleType::CENTER, BorderType btype = BorderType::CLAMP)
		: m_xform(xform)
		, m_stype(stype)
		, m_btype(btype)
		, m_full_size(size)
	{
		switch (stype)
		{
		case SampleType::CENTER:
			m_cell_offset = Vec2R(.5);
			this->resize(size, val);
			break;
		case SampleType::XFACE:
			m_cell_offset = Vec2R(.0, .5);
			this->resize(size + Vec2ui(1, 0), val);
			break;
		case SampleType::YFACE:
			m_cell_offset = Vec2R(.5, .0);
			this->resize(size + Vec2ui(0, 1), val);
			break;
		case SampleType::NODE:
			m_cell_offset = Vec2R(.0);
			this->resize(size + Vec2ui(1), val);
		}
	}

	SampleType sample_type() { return m_stype; }

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	bool is_matched(const ScalarGrid<S>& grid) const
	{
		if (this->m_size != grid.size()) return false;
		if (m_xform != grid.xform()) return false;
		if (m_stype != grid.sample_type()) return false;
		return true;
	}

	// Global multiply operator
	void operator*(const T& s)
	{
		for (auto& i : this->m_grid) i *= s;
	}

	// Global add operator
	void operator+(T s)
	{
		for (auto& i : this->m_grid) i += s;
	}

	T max_val() const { auto result = std::max_element(this->m_grid.begin(), this->m_grid.end()); return *result; }
	T min_val() const { auto result = std::min_element(this->m_grid.begin(), this->m_grid.end()); return *result; }

	void minmax(T& min, T& max) const
	{
		auto result = std::minmax_element(this->m_grid.begin(), this->m_grid.end());
		min = *(result.first); max = *(result.second);
	}

	T interp(Real x, Real y, bool idx_space = false) const { return interp(Vec2R(x, y), idx_space); }
	T interp(const Vec2R& pos, bool idx_space = false) const;
	
	T cubic_interp(Real x, Real y, bool idx_space = false, bool clamp = false) const { return cubic_interp(Vec2R(x, y), idx_space, clamp); }
	T cubic_interp(const Vec2R& pos, bool idx_space = false, bool clamp = false) const;

	// Converters between world space and local index space
	Vec2R idx_to_ws(Vec2R index_pos) const
	{
		return m_xform.idx_to_ws(index_pos + m_cell_offset);
	}
	
	Vec2R ws_to_idx(Vec2R world_pos) const
	{
		return m_xform.ws_to_idx(world_pos) - m_cell_offset;
	}

	// Gradient operators
	Vec<T, 2> gradient(const Vec2R& world_pos, bool idx_space = false) const
	{
		Real offset = idx_space ? 1E-1 : 1E-1 * dx();
		T dTdx = interp(world_pos + Vec2R(offset, 0.), idx_space) - interp(world_pos - Vec2R(offset, 0.), idx_space);
		T dTdy = interp(world_pos + Vec2R(0., offset), idx_space) - interp(world_pos - Vec2R(0., offset), idx_space);
		Vec<T, 2> grad(dTdx, dTdy);
		return grad / (2 * offset);
	}
	
	Real dx() const { return m_xform.dx(); }
	Vec2R offset() const { return m_xform.offset(); }
	Transform xform() const { return m_xform; }

	SampleType sample_type() const { return m_stype; }
	
	// Render methods
	void draw_grid(Renderer& renderer) const;
	void draw_grid_cell(Renderer& renderer, const Vec2ui& coord, const Vec3f& colour = Vec3f(0)) const;

	void draw_sample_points(Renderer& renderer, const Vec3f& colour = Vec3f(1,0,0), Real size = 1.) const;
	void draw_supersampled_values(Renderer& renderer, Real radius = .5, unsigned samples = 5, unsigned size = 1) const;
	void draw_sample_gradients(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1), Real length = .25) const;
	void draw_volumetric(Renderer& renderer, const Vec3f& mincolour, const Vec3f& maxcolour, T minval, T maxval) const;

private:

	// The main interpolation call after the template specialized clamping passes
	T interp_local(const Vec2R& pos) const;

	// Store the actual grid size. The m_nx member variable represents the 
	// underlying sample grid. The actual grid doesn't change based on sample
	// type but the underlying array of sample points do.
	Vec2ui m_full_size;

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
T ScalarGrid<T>::cubic_interp(const Vec2R& pos, bool idx_space, bool clamp) const
{
	Vec2R index_pos = idx_space ? pos : ws_to_idx(pos);

	Vec2R p11 = floor(index_pos);

	// Revert to linear interpolation near the boundaries
	if (p11[0] < 1 || p11[0] >= this->size()[0] - 2 ||
		p11[1] < 1 || p11[1] >= this->size()[1] - 2)
			return interp(pos, true);

	Vec2R u = index_pos - Vec2R(p11);
	
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

	T val = CINT(p_1, p0, p1, p2, u[1]);

	if (clamp)
	{
		Real xf = floor(index_pos[0]);
		Real yf = floor(index_pos[1]);

		// TODO: should this actually be floor(index_pos[0]+1) ?
		Real xc = ceil(index_pos[0]);
		Real yc = ceil(index_pos[1]);

		T v00 = (*this)(Vec2ui(xf, yf));
		T v10 = (*this)(Vec2ui(xc, yf));

		T v01 = (*this)(Vec2ui(xf, yc));
		T v11 = (*this)(Vec2ui(xc, yc));

		val = max(min(min(min(v00, v10), v01), v11), val);
		val = min(max(max(max(v00, v10), v01), v11), val);
	}

	return val;
}

template<typename T>
T ScalarGrid<T>::interp(const Vec2R& pos, bool idx_space) const
{
	Vec2R index_pos = idx_space ? pos : ws_to_idx(pos);

	switch (m_btype)
	{
	case BorderType::ZERO:
		if ((index_pos[0] < 0.) || (index_pos[1] < 0.) ||
			(index_pos[0] > Real(this->m_size[0] - 1)) ||
			(index_pos[1] > Real(this->m_size[1] - 1))) return T(0.);
	case BorderType::CLAMP:
		index_pos[0] = (index_pos[0] < 0.) ? 0. : ((index_pos[0] > Real(this->m_size[0] - 1)) ? Real(this->m_size[0] - 1) : index_pos[0]);
		index_pos[1] = (index_pos[1] < 0.) ? 0. : ((index_pos[1] > Real(this->m_size[1] - 1)) ? Real(this->m_size[1] - 1) : index_pos[1]);
		break;
	// Useful debug check. Equivalent to "NONE" for release mode
	case BorderType::ASSERT:
		assert((index_pos[0] >= 0.) && (index_pos[1] >= 0.) &&
					(index_pos[0] <= Real(this->m_size[0] - 1)) &&
					(index_pos[1] <= Real(this->m_size[1] - 1)));
		break;
	}

	return interp_local(index_pos);
}

// The local interp applies bi-linear interpolation on the UniformGrid. The 
// templated type must have operators for basic add/mult arithmetic.
template<typename T>
T ScalarGrid<T>::interp_local(const Vec2R& index_pos) const
{
	Real xf = floor(index_pos[0]);
	Real yf = floor(index_pos[1]);

	// TODO: should this actually be floor(index_pos[0]+1) ?
	Real xc = ceil(index_pos[0]);
	Real yc = ceil(index_pos[1]);

	// This should always pass since the clamp test in the public interp
	// will not allow anything outside of the border to get this far.
	assert(xf >= 0 && yf >= 0);
	assert(xc <= Real(this->m_size[0] - 1) &&
			yc <= Real(this->m_size[1] - 1));

	// Use base grid class operator
	T v00 = (*this)(Vec2ui(xf, yf));
	T v10 = (*this)(Vec2ui(xc, yf));

	T v01 = (*this)(Vec2ui(xf, yc));
	T v11 = (*this)(Vec2ui(xc, yc));

	Real sx = index_pos[0] - xf;
	Real sy = index_pos[1] - yf;

	auto lerp = [](const T& val0, const T& val1, Real f){ return val0 + f * (val1 - val0); };

	T interp_val = lerp(lerp(v00, v10, sx), lerp(v01, v11, sx), sy);
	return interp_val;
}

static Vec2R edge_to_node[4][2] = { {Vec2R(0), Vec2R(1,0) },
									{ Vec2R(1,0), Vec2R(1,1) },
									{ Vec2R(1,1), Vec2R(0,1) },
									{ Vec2R(0,1), Vec2R(0,0) } };

template<typename T>
void ScalarGrid<T>::draw_grid_cell(Renderer& renderer, const Vec2ui& coord, const Vec3f& colour) const
{
	std::vector<Vec2R> startpoints;
	std::vector<Vec2R> endpoints;

	for (int e = 0; e < 4; ++e)
	{
		Vec2R pos1 = idx_to_ws(Vec2R(coord) - m_cell_offset + edge_to_node[e][0]);
		Vec2R pos2 = idx_to_ws(Vec2R(coord) - m_cell_offset + edge_to_node[e][1]);
		
		startpoints.push_back(pos1);
		endpoints.push_back(pos2);
	}

	renderer.add_lines(startpoints, endpoints, colour);
}

template<typename T>
void ScalarGrid<T>::draw_grid(Renderer& renderer) const
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		for (unsigned i = 0; i <= m_full_size[axis]; ++i)
		{
			// Offset backwards because we want the start of the grid cell
			Vec2R grid_start(0);
			grid_start[axis] = i;

			Vec2R pos = idx_to_ws(grid_start - m_cell_offset);
			start_points.push_back(pos);

			Vec2R grid_end(m_full_size);
			grid_end[axis] = i;

			pos = idx_to_ws(grid_end - m_cell_offset);
			end_points.push_back(pos);
		}
	}

	renderer.add_lines(start_points, end_points, Vec3f(0));
}

template<typename T>
void ScalarGrid<T>::draw_sample_points(Renderer& renderer, const Vec3f& colour, Real size) const
{
	std::vector<Vec2R> sample_points;

	Vec2ui grid_size = this->m_size;

	for_each_voxel_range(Vec2ui(0), grid_size, [&](const Vec2ui& coord)
	{
		Vec2R index_pos = Vec2R(coord);
		Vec2R wpos = idx_to_ws(index_pos);

		sample_points.push_back(wpos);
	});

	renderer.add_points(sample_points, colour, size);
}

// Warning: there is no protection here for ASSERT border types
template<typename T>
void ScalarGrid<T>::draw_supersampled_values(Renderer& renderer, Real radius, unsigned samples, unsigned size) const
{
	std::vector<Vec2R> sample_points;

	T max_sample = max_val();
	T min_sample = min_val();

	Vec2ui grid_size = this->m_size;

	for_each_voxel_range(Vec2ui(0), grid_size, [&](const Vec2ui& coord)
	{
		Vec2R index_pos = Vec2R(coord);

		// Supersample
		Real dx = 2 * radius / Real(samples);
		for (Real x = -radius; x <= radius; x += dx)
			for (Real y = -radius; y <= radius; y += dx)
			{
				Vec2R sample_pos = index_pos + Vec2R(x, y);
				Vec2R world_pos = idx_to_ws(sample_pos);

				T sampleval = (interp(world_pos) - min_sample) / max_sample;
				renderer.add_point(world_pos, Vec3f(float(sampleval), float(sampleval), 0), size);
			}
	});
}

template<typename T>
void ScalarGrid<T>::draw_sample_gradients(Renderer& renderer, const Vec3f& colour, Real length) const
{
	std::vector<Vec2R> sample_points;
	std::vector<Vec2R> gradient_points;

	Vec2ui size = this->m_size;
	for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& coord)
	{
		Vec2R index_pos = Vec2R(coord);
		Vec2R start_pos = idx_to_ws(index_pos);
		sample_points.push_back(start_pos);

		Vec<T, 2> grad = gradient(start_pos);
		Vec2R end_pos = start_pos + length * grad;
		gradient_points.push_back(end_pos);
	});

	renderer.add_lines(sample_points, gradient_points, colour);
}

template<typename T>
void ScalarGrid<T>::draw_volumetric(Renderer& renderer, const Vec3f& mincolour, const Vec3f& maxcolour, T minval, T maxval) const
{
	ScalarGrid<Real> nodes(xform(), this->size(), SampleType::NODE);

	std::vector<Vec2R> verts(nodes.size()[0] * nodes.size()[1]);
	std::vector<Vec4ui> pixels(this->m_size[0] * this->m_size[1]);
	std::vector<Vec3f> colours(this->m_size[0] * this->m_size[1]);

	// Set node points
	{
		Vec2ui size = nodes.size();

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& coord)
		{
			unsigned vert = nodes.flatten(Vec2ui(coord));
			verts[vert] = nodes.idx_to_ws(Vec2R(coord));
		});
	}

	Vec2ui cell_to_node_cw[] = { Vec2ui(0,0), Vec2ui(1,0), Vec2ui(1,1), Vec2ui(0,1) };

	{
		Vec2ui size = this->m_size;

		unsigned pixelcount = 0;

		for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& coord)
		{
			Vec2ui idx(coord);
			Vec4ui quad;

			for (int c = 0; c < 4; ++c)
			{
				Vec2ui point = idx + cell_to_node_cw[c];
				quad[c] = nodes.flatten(Vec2ui(point));
			}

			T val = (*this)(coord);

			pixels[pixelcount] = quad;

			Vec3f colour;
			if (val > maxval) colour = maxcolour;
			else if (val < minval) colour = mincolour;
			else
			{
				float s = val / (maxval - minval);

				colour = mincolour * (1. - s) + maxcolour * s;
			}

			colours[pixelcount++] = colour;
		});
	}

	renderer.add_quads(verts, pixels, colours);
}

#pragma once

#include "core.h"
#include "ScalarGrid.h"

#include "Transform.h"

///////////////////////////////////
//
// VectorGrid.h
// Ryan Goldade 2016
//
// Container class of two ScalarGrids.
// Provides control for cell-centered
// or staggered formation.
//
////////////////////////////////////

namespace VectorGridSettings
{
	enum VectorSampleType { CENTER, STAGGERED, NODE };
}

template<typename T>
class VectorGrid
{
	typedef ScalarGrid<T> Scalar;
	typedef ScalarGridSettings::BorderType Border;
	typedef VectorGridSettings::VectorSampleType SampleType;
public:

	// TODO: fix up the scope stuff here with using

	VectorGrid() : m_xform(1., Vec2R(0.)), m_nx(0) { m_grids.resize(2); }
	VectorGrid(const Transform& xform, const Vec2st& nx,
				SampleType stype = SampleType::CENTER, Border btype = CLAMP)
		: VectorGrid(xform, nx, T(0), stype, btype)
	{}

	VectorGrid(const Transform& xform, const Vec2st& nx, T val,
				SampleType stype = SampleType::CENTER, Border btype = CLAMP)
		: m_xform(xform)
		, m_nx(nx)
		, m_stype(stype)
	{
		m_grids.resize(2);
		switch (stype)
		{
		case SampleType::CENTER:
			m_grids[0] = Scalar(xform, nx, val, ScalarGridSettings::CENTER, btype);
			m_grids[1] = Scalar(xform, nx, val, ScalarGridSettings::CENTER, btype);
			break;
		// If the grid is 2x2, it has 3x2 x-aligned faces and 2x3 y-aligned faces.
		// This is handled inside of the ScalarGrid
		case SampleType::STAGGERED:
			m_grids[0] = Scalar(xform, nx, val, ScalarGridSettings::XFACE, btype);
			m_grids[1] = Scalar(xform, nx, val, ScalarGridSettings::YFACE, btype);
			break;
		// If the grid is 2x2, it has 3x3 nodes. This is handled inside of the ScalarGrid
		case SampleType::NODE:
			m_grids[0] = Scalar(xform, nx, val, ScalarGridSettings::NODE, btype);
			m_grids[1] = Scalar(xform, nx, val, ScalarGridSettings::NODE, btype);
		}
	}


	SampleType sample_type() { return m_stype; }

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	inline bool is_matched(const VectorGrid<S>& grid) const
	{
		if (size() != grid.size()) return false;
		if (m_xform != grid.xform()) return false;
		if (m_stype != grid.sample_type()) return false;
		return true;
	}

	ScalarGrid<T>& grid(size_t axis)
	{
		assert(axis < 2);
		return m_grids[axis];
	}
	
	const ScalarGrid<T>& grid(size_t axis) const
	{
		assert(axis < 2);
		return m_grids[axis];
	}
	// TODO: write accessors to put values into the grid
	// write renderers to see the sample points, vector values (averaged and offset for staggered)
	
	T& operator()(int x, int y, size_t axis)
	{
		assert(axis < 2);
		return m_grids[axis](x, y);
	}

	const T& operator()(int x, int y, size_t axis) const
	{
		assert(axis < 2);
		return m_grids[axis](x, y);
	}	

	T max_magnitude() const;

	Vec<2, T> interp(Real x, Real y) const { return interp(Vec2R(x, y)); }
	Vec<2, T> interp(const Vec2R& pos) const
	{
		return Vec<2,T>(interp(pos, 0), interp(pos, 1));
	}

	T interp(Real x, Real y, size_t axis) const { return interp(Vec2R(x, y), axis); }
	T interp(const Vec2R& pos, size_t axis) const
	{ 
		assert(axis < 2);
		return m_grids[axis].interp(pos);
	}

	void set(size_t x, size_t y, size_t axis, T val)
	{
		assert(axis < 2);
		m_grids[axis](x, y) = val;
	}

	// World space vs. index space converters need to be done at the 
	// underlying scalar grid level because the alignment of the two 
	// grids are different depending on the SampleType.
	inline Vec2R idx_to_ws(const Vec2R& ipos, size_t axis) const
	{
		assert(axis < 2);
		return m_grids[axis].idx_to_ws(ipos);
	}
	inline Vec2R ws_to_idx(const Vec2R& wpos, size_t axis) const
	{
		assert(axis < 2);
		return m_grids[axis].ws_to_idx(wpos);
	}

	Real dx() const { return m_xform.dx(); }
	Real offset() const { return m_xform.offset(); }
	inline Transform xform() const { return m_xform; }
	Vec2st size(size_t axis) const { return m_grids[axis].size(); }

	// Rendering methods
	void draw_grid(Renderer& renderer) const;
	void draw_sample_points(Renderer& renderer, const Vec3f& colour0 = Vec3f(1,0,0),
								const Vec3f& colour1 = Vec3f(0, 0, 1), const Vec2R& sizes = Vec2R(1.)) const;
	void draw_supersampled_values(Renderer& renderer, Real radius = .5, size_t samples = 10) const;
	void draw_sample_point_vectors(Renderer& renderer, const Vec3f& colour = Vec3f(0,0,1), Real length = .25) const;

private:

	// This method is private to prevent future mistakes between this transform
	// and the staggered scalar grids
	Vec2R idx_to_ws(const Vec2R& idx) const
	{
		return m_xform.idx_to_ws(idx);
	}

	std::vector<Scalar> m_grids;

	Transform m_xform;

	Vec2st m_nx;

	SampleType m_stype;
};

template<typename T>
void VectorGrid<T>::draw_grid(Renderer& renderer) const
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (int i = 0; i <= m_nx[0]; ++i)
	{
		// Offset backwards because we want the start of the grid cell
		Vec2R pos = idx_to_ws(Vec2R(i, 0));
		start_points.push_back(pos);

		pos = idx_to_ws(Vec2R(i, m_nx[1]));
		end_points.push_back(pos);
	}

	for (int j = 0; j <= m_nx[1]; ++j)
	{
		Vec2R pos = idx_to_ws(Vec2R(0, j));
		start_points.push_back(pos);

		pos = idx_to_ws(Vec2R(m_nx[0], j));
		end_points.push_back(pos);
	}

	renderer.add_lines(start_points, end_points, Vec3f(0));	
}

template<typename T>
void VectorGrid<T>::draw_sample_points(Renderer& renderer, const Vec3f& colour0,
										const Vec3f& colour1, const Vec2R& sizes) const
{
	m_grids[0].draw_sample_points(renderer,	colour0, sizes[0]);
	m_grids[1].draw_sample_points(renderer, colour1, sizes[1]);
}

template<typename T>
void VectorGrid<T>::draw_supersampled_values(Renderer& renderer, Real radius, size_t samples) const
{
	m_grids[0].draw_supersampled_values(renderer, radius, samples);
	m_grids[1].draw_supersampled_values(renderer, radius, samples);
}

template<typename T>
void VectorGrid<T>::draw_sample_point_vectors(Renderer& renderer, const Vec3f& colour = Vec3f(0, 0, 1),
												Real length = .25) const
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	switch (m_stype)
	{
	case SampleType::CENTER:
		for (size_t i = 0; i < m_nx[0]; ++i)
			for (size_t j = 0; j < m_nx[1]; ++j)
			{
				Vec2R wpos =  idx_to_ws(Vec2R(i, j), 0);
				start_points.push_back(wpos);

				Vec2R vec(m_grids[0].val(i, j), m_grids[1].val(i, j));
				end_points.push_back(wpos + length * vec);
			}
		break;
	case SampleType::NODE:
		for (size_t i = 0; i <= m_grids[0].size()[0]; ++i)
			for (size_t j = 0; j <= m_grids[0].size()[1]; ++j)
			{
				Vec2R wpos = m_grids[0].idx_to_ws(Vec2R(i, j));
				start_points.push_back(wpos);

				Vec2R vec(m_grids[0].val(i, j), m_grids[1].val(i, j));
				end_points.push_back(wpos + length * vec);
			}
		break;
	case SampleType::STAGGERED:
		for (size_t i = 0; i < m_nx[0]; ++i)
			for (size_t j = 0; j < m_nx[1]; ++j)
			{
				Vec2R wpos1 = idx_to_ws(Vec2R(i, j), 0);
				Vec2R wpos2 = idx_to_ws(Vec2R(i + 1, j), 0);
				Vec2R wpos3 = idx_to_ws(Vec2R(i, j), 1);
				Vec2R wpos4 = idx_to_ws(Vec2R(i, j + 1), 1);
				Vec2R avg_wpos = .25 * (wpos1 + wpos2 + wpos3 + wpos4);
				start_points.push_back(avg_wpos);

				Vec2R vec1(m_grids[0].val(i, j), m_grids[1].val(i, j));
				Vec2R vec2(m_grids[0].val(i+1, j), m_grids[1].val(i, j+1));
				Vec2R avg_vec = .5 * (vec1 + vec2);
				end_points.push_back(avg_wpos + length * avg_vec);
			}
		break;
	}

	renderer.add_lines(start_points, end_points, colour);
}

// Magnitude is useful for CFL conditions
template<typename T>
T VectorGrid<T>::max_magnitude() const
{
	Real max = std::numeric_limits<Real>::min();
	switch (m_stype)
	{
	case SampleType::CENTER:

		for (size_t i = 0; i < m_nx[0]; ++i)
			for (size_t j = 0; j < m_nx[1]; ++j)
			{
				Real tempmag2 = mag2(Vec<2,T>(m_grids[0](i, j), m_grids[1](i, j)));
				
				if (max < tempmag2) max = tempmag2;
			}
		return sqrt(max);
		break;

	case SampleType::NODE:

		for (size_t i = 0; i < m_grids[0].size()[0]; ++i)
			for (size_t j = 0; j < m_grids[0].size()[1]; ++j)
			{
				Real tempmag2 = mag2(Vec<2, T>(m_grids[0](i, j), m_grids[1](i, j)));

				if (max < tempmag2) max = tempmag2;
			}

		return sqrt(max);
		break;

	case SampleType::STAGGERED:

		for (size_t i = 0; i < m_nx[0]; ++i)
			for (size_t j = 0; j < m_nx[1]; ++j)
			{
				Vec2R vec1(m_grids[0].val(i, j), m_grids[1].val(i, j));
				Vec2R vec2(m_grids[0].val(i + 1, j), m_grids[1].val(i, j + 1));
				Vec2R avg_vec = .5 * (vec1 + vec2);

				Real tempmag2 = mag2(avg_vec);
				if (max < tempmag2) max = tempmag2;
			}
		
		return sqrt(max);
		break;
	default:
		assert(false);
	}
	return T(0);
}
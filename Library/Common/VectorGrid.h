#pragma once

#include "Common.h"
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
	enum class SampleType { CENTER, STAGGERED, NODE };
}

template<typename T>
class VectorGrid
{
	using ScalarGridT = ScalarGrid<T>;
	using ScalarSampleType = ScalarGridSettings::SampleType;
	using BorderType = ScalarGridSettings::BorderType;
	using SampleType = VectorGridSettings::SampleType;

public:

	VectorGrid() : m_xform(1., Vec2R(0.)), m_size(0) { m_grids.resize(2); }
	VectorGrid(const Transform& xform, const Vec2ui& size,
				SampleType stype = SampleType::CENTER, BorderType btype = BorderType::CLAMP)
		: VectorGrid(xform, size, T(0), stype, btype)
	{}

	VectorGrid(const Transform& xform, const Vec2ui& size, T val,
				SampleType stype = SampleType::CENTER, BorderType btype = BorderType::CLAMP)
		: m_xform(xform)
		, m_size(size)
		, m_stype(stype)
	{
		m_grids.resize(2);
		switch (stype)
		{
		case SampleType::CENTER:
			m_grids[0] = ScalarGridT(xform, size, val, ScalarSampleType::CENTER, btype);
			m_grids[1] = ScalarGridT(xform, size, val, ScalarSampleType::CENTER, btype);
			break;
		// If the grid is 2x2, it has 3x2 x-aligned faces and 2x3 y-aligned faces.
		// This is handled inside of the ScalarGrid
		case SampleType::STAGGERED:
			m_grids[0] = ScalarGridT(xform, size, val, ScalarSampleType::XFACE, btype);
			m_grids[1] = ScalarGridT(xform, size, val, ScalarSampleType::YFACE, btype);
			break;
		// If the grid is 2x2, it has 3x3 nodes. This is handled inside of the ScalarGrid
		case SampleType::NODE:
			m_grids[0] = ScalarGridT(xform, size, val, ScalarSampleType::NODE, btype);
			m_grids[1] = ScalarGridT(xform, size, val, ScalarSampleType::NODE, btype);
		}
	} 

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	bool is_matched(const VectorGrid<S>& grid) const
	{
		if (size(0) != grid.size(0)) return false;
		if (size(1) != grid.size(1)) return false;
		if (m_xform != grid.xform()) return false;
		if (m_stype != grid.sample_type()) return false;
		return true;
	}

	ScalarGridT& grid(unsigned axis)
	{
		assert(axis < 2);
		return m_grids[axis];
	}
	
	const ScalarGridT& grid(unsigned axis) const
	{
		assert(axis < 2);
		return m_grids[axis];
	}
	// TODO: write accessors to put values into the grid
	// write renderers to see the sample points, vector values (averaged and offset for staggered)
	
	T& operator()(unsigned i, unsigned j, unsigned axis) { return (*this)(Vec2ui(i, j), axis); }

	T& operator()(const Vec2ui& coord, unsigned axis)
	{
		assert(axis < 2);
		return m_grids[axis](coord);
	}

	const T& operator()(unsigned i, unsigned j, unsigned axis) const { return (*this)(Vec2ui(i, j), axis); }

	const T& operator()(const Vec2ui& coord, unsigned axis) const
	{
		assert(axis < 2);
		return m_grids[axis](coord);
	}

	T max_magnitude() const;

	Vec<T, 2> interp(Real x, Real y) const { return interp(Vec2R(x, y)); }
	Vec<T, 2> interp(const Vec2R& pos) const
	{
		return Vec<T, 2>(interp(pos, 0), interp(pos, 1));
	}

	T interp(Real x, Real y, unsigned axis) const { return interp(Vec2R(x, y), axis); }
	T interp(const Vec2R& pos, unsigned axis) const
	{ 
		assert(axis < 2);
		return m_grids[axis].interp(pos);
	}

	// World space vs. index space converters need to be done at the 
	// underlying scalar grid level because the alignment of the two 
	// grids are different depending on the SampleType.
	Vec2R idx_to_ws(const Vec2R& index_pos, unsigned axis) const
	{
		assert(axis >= 0 && axis < 2);
		return m_grids[axis].idx_to_ws(index_pos);
	}

	Vec2R ws_to_idx(const Vec2R& world_pos, unsigned axis) const
	{
		assert(axis < 2);
		return m_grids[axis].ws_to_idx(world_pos);
	}

	Real dx() const { return m_xform.dx(); }
	Real offset() const { return m_xform.offset(); }
	Transform xform() const { return m_xform; }
	Vec2ui size(unsigned axis) const { return m_grids[axis].size(); }
	Vec2ui grid_size() const { return m_size; }
	SampleType sample_type() const { return m_stype; }

	// Rendering methods
	void draw_grid(Renderer& renderer) const;
	void draw_sample_points(Renderer& renderer, const Vec3f& colour0 = Vec3f(1,0,0),
								const Vec3f& colour1 = Vec3f(0, 0, 1), const Vec2R& sizes = Vec2R(1.)) const;
	void draw_supersampled_values(Renderer& renderer, Real radius = .5, unsigned samples = 10) const;
	void draw_sample_point_vectors(Renderer& renderer, const Vec3f& colour = Vec3f(0,0,1), Real length = .25) const;

private:

	// This method is private to prevent future mistakes between this transform
	// and the staggered scalar grids
	Vec2R idx_to_ws(const Vec2R& idx) const
	{
		return m_xform.idx_to_ws(idx);
	}

	std::vector<ScalarGridT> m_grids;

	Transform m_xform;

	Vec2ui m_size;

	SampleType m_stype;
};

template<typename T>
void VectorGrid<T>::draw_grid(Renderer& renderer) const
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		for (unsigned i = 0; i <= m_size[axis]; ++i)
		{
			Vec2R grid_start(0);
			grid_start[axis] = i;

			// Offset backwards because we want the start of the grid cell
			Vec2R pos = idx_to_ws(grid_start);
			start_points.push_back(pos);

			Vec2R grid_end(m_size);
			grid_end[axis] = i;

			pos = idx_to_ws(grid_end);
			end_points.push_back(pos);
		}
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
void VectorGrid<T>::draw_supersampled_values(Renderer& renderer, Real radius, unsigned samples) const
{
	m_grids[0].draw_supersampled_values(renderer, radius, samples);
	m_grids[1].draw_supersampled_values(renderer, radius, samples);
}

template<typename T>
void VectorGrid<T>::draw_sample_point_vectors(Renderer& renderer, const Vec3f& colour, Real length) const
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	switch (m_stype)
	{
	case SampleType::CENTER:
		
		for_each_voxel_range(Vec2ui(0), m_size, [&](const Vec2ui& cell)
		{
			Vec2R world_pos = idx_to_ws(Vec2R(cell), 0);
			start_points.push_back(world_pos);

			Vec2R vec(m_grids[0](cell), m_grids[1](cell));
			end_points.push_back(world_pos + length * vec);
		});
		break;

	case SampleType::NODE:

		for_each_voxel_range(Vec2ui(0), m_grids[0].size(), [&](const Vec2ui& node)
		{
			Vec2R world_pos = m_grids[0].idx_to_ws(Vec2R(node));
			start_points.push_back(world_pos);

			Vec2R vec(m_grids[0](node), m_grids[1](node));
			end_points.push_back(world_pos + length * vec);
		});

		break;

	case SampleType::STAGGERED:

		for_each_voxel_range(Vec2ui(0), m_size, [&](const Vec2ui& cell)
		{
			Vec2R avg_world_pos(0);
			Vec2R avg_vec(0);

			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2ui face = cell + cell_to_face[dir];

				avg_world_pos += .25 * idx_to_ws(Vec2R(face));

				unsigned axis = dir / 2;
				unsigned forward = dir % 2;

				avg_vec[forward] += .5 * m_grids[axis](face);
			}

			start_points.push_back(avg_world_pos);
			end_points.push_back(avg_world_pos + length * avg_vec);
		});
		
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
		
		for_each_voxel_range(Vec2ui(0), m_size, [&](const Vec2ui& cell)
		{
			Real tempmag2 = mag2(Vec<T, 2>(m_grids[0](cell), m_grids[1](cell)));
				
			if (max < tempmag2) max = tempmag2;
		});

		return sqrt(max);
		break;

	case SampleType::NODE:

		for_each_voxel_range(Vec2ui(0), m_size, [&](const Vec2ui& node)
		{
			Real tempmag2 = mag2(Vec<T, 2>(m_grids[0](node), m_grids[1](node)));

			if (max < tempmag2) max = tempmag2;
		});

		return sqrt(max);
		break;

	case SampleType::STAGGERED:

		for_each_voxel_range(Vec2ui(0), m_size, [&](const Vec2ui& cell)
		{
			Vec2R avg_vec(0);

			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2ui face = cell + cell_to_face[dir];

				unsigned axis = dir / 2;

				avg_vec[axis] += .5 * m_grids[axis](face);
			}

			Real tempmag2 = mag2(avg_vec);
			if (max < tempmag2) max = tempmag2;
		});
		
		return sqrt(max);
		break;
	default:
		assert(false);
	}
	return T(0);
}

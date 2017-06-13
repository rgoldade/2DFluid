#pragma once

#include "core.h"
#include "vec.h"
#include "predicates.h"
#include "HashGrid2D.h"
#include "Mesh2D.h"

#include "Renderer.h"

///////////////////////////////////
//
// MeshRebuilder.h/cpp
// Ryan Goldade 2017
//
// 2D mesh rebuilder using an approach
// similar to "Fast and Robust Tracking
// of Fluid Surfaces" by Muller 2009.
//
////////////////////////////////////

class MeshRebuilder
{
public:
	MeshRebuilder(Real dx)
		: m_dx(dx)
	{
		exactinit();
	}

	void rebuild(Mesh2D& mesh, bool dualcontouring);

	void draw_intersections(Renderer &renderer);

private:

	// World space conversions
	inline Vec2R ws_to_idx(const Vec2R& pos) const
	{
		return pos / m_dx;
	}
	inline Vec2R idx_to_ws(const Vec2R& pos) const
	{
		return pos * m_dx;
	}
	
	Real m_dx;

	// Mesh-grid intersection points
	HashGrid2D<Vec2R> m_intersections;

	// Intersection normals
	HashGrid2D<Vec2R> m_normals;

	// Direction count
	HashGrid2D<int> m_directions;
	HashGrid2D<Real> m_count;

	// Inside/outside parity for nodes
	HashGrid2D<bool> m_nodes;

	// Debug list
	std::vector<Vec2R> m_intersection_list;
};
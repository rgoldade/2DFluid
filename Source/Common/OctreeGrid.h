#pragma once

#include "Core.h"
#include "Util.h"

#include "ScalarGrid.h"
#include "Transform.h"

#include "Renderer.h"

///////////////////////////////////
//
// OctreeGrid.h
// Ryan Goldade 2017
//
////////////////////////////////////

enum ActiveCell {INACTIVE, ACTIVE, UP, DOWN};

static Vec2st cell_children[] = { Vec2st(0, 0), Vec2st(1, 0),
									Vec2st(1, 1), Vec2st(0, 1) };

// Offset for small cells embedded in the large cell. Each set is
// build based on a similar aligned face.
static size_t adjacent_offset[4][2] = { {1, 2}, {0, 3}, {3, 2}, {0, 1} };

class OctreeGrid
{
public:

	OctreeGrid(const OctreeGrid& sample_tree)
	{
		m_markers = sample_tree.get_grids();
	}

	OctreeGrid(const Transform& xform, const Vec2st& fine_nx, size_t max_levels = 2)
	{
		Vec2st nx = fine_nx;

		for (size_t i = 0; i < 2; ++i)
		{
			Real logx = log2(nx[i]);
			logx = ceil(logx);
			nx[i] = exp2(logx);
		}

		if (nx[0] > 0 && nx[1] > 0)
		{
			if (log2(nx[0]) < max_levels) max_levels = log2(nx[0]);
			if (log2(nx[1]) < max_levels) max_levels = log2(nx[1]);
		}

		m_markers.resize(max_levels);
		m_markers[0] = ScalarGrid<ActiveCell>(xform, nx, UP);

		Real dx = xform.dx();

		for (size_t i = 1; i < max_levels; ++i)
		{
			dx *= 2.;
			nx = nx / 2;

			Transform tmp_xform(dx, xform.offset());

			m_markers[i] = ScalarGrid<ActiveCell>(tmp_xform, nx, UP);
		}
	}

	void refine_grid();

	// Use a refinement functor to build the grading of the tree. The functor
	// should take a Vec2R in world space and return a bool. A true value indicates
	// that the sample position should be at the finest resolution. A false value
	// indicates it's free to grade as needed.
	template<typename Refinement>
	void build_tree(Refinement refiner);

	inline Vec2st get_parent(const Vec2st& coord) const
	{
		return coord / 2;
	}

	inline Vec2st get_child(const Vec2st& coord, size_t child) const
	{
		assert(child < 4);
		return coord * 2 + cell_children[child];
	}

	inline Vec2st get_child_node(const Vec2st& coord) const
	{
		return coord * 2;
	}

	inline Vec2st get_inset_node(const Vec2st& coord, size_t axis) const
	{
		Vec2st offset(0); offset[(axis + 1) % 2] += 1;
		return coord * 2 + offset;
	}

	inline Vec2st get_child_face(const Vec2st& coord, size_t axis, size_t child) const
	{
		assert(axis < 2 && child < 2);
		Vec2st offset(0); offset[axis == 0 ? 1 : 0] += child;
		return coord * 2 + offset;
	}

	size_t levels() const { return m_markers.size(); }
	Vec2st size(size_t level) const { return m_markers[level].size(); }

	const std::vector<ScalarGrid<ActiveCell>>& get_grids() const { return m_markers; }

	inline bool is_active(const Vec2st& coord, size_t level) const { return m_markers[level](coord[0], coord[1]) == ACTIVE; }
	inline ActiveCell activity(const Vec2st& coord, size_t level) const { return m_markers[level](coord[0], coord[1]); }
	std::vector<Vec3st> get_face_adjacent_cells(const Vec2st& coord, size_t level, size_t dir) const;

	template<typename Refinement>
	bool unit_test(Refinement &refinement) const;

	void draw_grid(Renderer &renderer, const Vec3f& colour = Vec3f(0)) const;
	void draw_cell_connections(Renderer& renderer) const;

	inline Transform xform(size_t level) const
	{
		return m_markers[level].xform();
	}


private:
	// Markers to label active cells, if the cell is "above"
	// or "below" the leaf (aka active cell).
	std::vector<ScalarGrid<ActiveCell>> m_markers;

};

template<typename Refinement>
void OctreeGrid::build_tree(Refinement refiner)
{
	// Pass over the finest resolution and activate cells near the isosurface.
	// Then activate its siblings.

	for (size_t i = 0; i < m_markers[0].size()[0]; ++i)
		for (size_t j = 0; j < m_markers[0].size()[1]; ++j)
		{
			if (m_markers[0](i, j) != ACTIVE)
			{
				int type = refiner(m_markers[0].idx_to_ws(Vec2R(i, j)));
				if (type == 0)
					m_markers[0](i, j) = ACTIVE;
				else if(type < 0)
					m_markers[0](i, j) = UP;
				else
					m_markers[0](i, j) = INACTIVE;
			}
		}

	for (int level = 0; level < m_markers.size() - 1; ++level)
	{
		// First pass:
		// If cell is ACTIVE, set all siblings to ACTIVE and parent to DOWN.

		Vec2st size = m_markers[level].size();

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				ActiveCell activity = m_markers[level](i, j);

				if (activity == ACTIVE)
				{
					// Label all sibling cells as active
					Vec2st parent = get_parent(Vec2st(i, j));

					for (size_t c = 0; c < 4; ++c)
					{
						Vec2st sibling = get_child(parent, c);
						if (m_markers[level](sibling[0], sibling[1]) == UP)
							m_markers[level](sibling[0], sibling[1]) = ACTIVE;
					}

					m_markers[level + 1](parent[0], parent[1]) = DOWN;
				}
			}

		// Second pass: 
		// If cell is ACTIVE, any face-adjacent cell that are labelled up will have their parents listed as active.
		// If cell is UP, make sure child is UP. Otherwise set to DOWN. The "otherwise" should never happen given the other two ifs
		// If cell is DOWN, set parent to DOWN.
		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				ActiveCell activity = m_markers[level](i, j);

				if (activity == ACTIVE)
				{
					Vec2i cell(i, j);
					// Loop over adjacent cells
					for (size_t c = 0; c < 4; ++c)
					{
						Vec2i adjcell = cell + cell_offset[c];

						if (adjcell[0] < 0 || adjcell[1] < 0
							|| adjcell[0] >= size[0]
							|| adjcell[1] >= size[1]) continue;

						if (m_markers[level](adjcell[0], adjcell[1]) == UP)
						{
							Vec2st parent = get_parent(Vec2st(adjcell));
							assert(parent != get_parent(Vec2st(cell)));

							m_markers[level + 1](parent[0], parent[1]) = ACTIVE;
						}
					}
				}
				else if (activity == DOWN || activity == INACTIVE)
				{
					Vec2st parent = get_parent(Vec2st(i, j));
					m_markers[level + 1](parent[0], parent[1]) = DOWN;
				}
				else if (level > 0) // Implied that activity == UP
				{
					for (size_t c = 0; c < 4; ++c)
					{
						Vec2st child = get_child(Vec2st(i, j), c);
						assert(m_markers[level - 1](child[0], child[1]) == UP);
					}
				}
			}
	}

	// Clean up loop. Label all UP cells at the coarsest level to ACTIVE
	{
		size_t level = m_markers.size() - 1;
		Vec2st size = m_markers[level].size();

		for (int i = 0; i < size[0]; ++i)
			for (int j = 0; j < size[1]; ++j)
			{
				if (m_markers[level](i, j) == UP) m_markers[level](i, j) = ACTIVE;
			}
	}
}

template<typename Refinement>
bool OctreeGrid::unit_test(Refinement &refinement) const
{
	size_t maxlevels = m_markers.size();

	// Make sure that there's only one active cell in each path from the finest to coarsest
	// stack of cells.
	{
		Vec2st size = m_markers[0].size();

		for (size_t i = 0; i < size[0]; ++i)
			for (size_t j = 0; j < size[1]; ++j)
			{
				Vec2st coord(i, j);

				size_t count = 0;
				for (size_t level = 0; level < maxlevels; ++level)
				{
					if (m_markers[level](coord[0], coord[1]) == ACTIVE) ++count;
					coord = get_parent(coord);
				}

				if (count == 0)
				{
					Vec2R pos = m_markers[0].idx_to_ws(Vec2R(coord));
					if (refinement(pos) <= 0) return false;
				}
				if (count > 1) return false;
			}
	}

	// Make sure that an active cell's adjacent active cells reciprocate
	for (size_t level = 0; level < maxlevels; ++level)
	{
		Vec2st size = m_markers[level].size();

		for (size_t i = 0; i < size[0]; ++i)
			for (size_t j = 0; j < size[1]; ++j)
			{
				if (m_markers[level](i, j) == ACTIVE || m_markers[level](i, j) == INACTIVE)
				{
					for (size_t c = 0; c < 4; ++c)
					{
						Vec2i adjcell = Vec2i(i, j) + cell_offset[c];

						if (adjcell[0] < 0 || adjcell[1] < 0 ||
							adjcell[0] >= size[0] || adjcell[1] >= size[1])
							continue;

						std::vector<Vec3st> adjcells = get_face_adjacent_cells(Vec2st(i, j), level, c);

						for (size_t adj = 0; adj < adjcells.size(); ++adj)
						{
							Vec2st cell(adjcells[adj][0], adjcells[adj][1]);
							int offset = (c % 2 == 0) ? 1 : -1;
							std::vector<Vec3st> returnadjcells = get_face_adjacent_cells(cell, adjcells[adj][2], c + offset);

							auto result = std::find(returnadjcells.begin(), returnadjcells.end(), Vec3st(i, j, level));
							if (result == returnadjcells.end())
							{
								return false;
							}
						}
					}
				}
			}
	}

	return true;
}

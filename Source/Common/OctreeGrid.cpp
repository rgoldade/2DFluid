#include "OctreeGrid.h"

void OctreeGrid::draw_cell_connections(Renderer& renderer) const
{
	std::vector<Vec2R> startpoints;
	std::vector<Vec2R> endpoints;

	size_t maxlevels = m_markers.size();
	for (size_t level = 0; level < maxlevels; ++level)
	{
		Vec2st size = m_markers[level].size();

		for (size_t i = 0; i < size[0]; ++i)
			for (size_t j = 0; j < size[1]; ++j)
			{
				if (m_markers[level](i, j) == ACTIVE)
				{
					for (size_t c = 0; c < 4; ++c)
					{
						Vec2i adjcell = Vec2i(i, j) + cell_offset[c];

						if (adjcell[0] < 0 || adjcell[1] < 0 ||
							adjcell[0] >= size[0] || adjcell[1] >= size[1])
							continue;

						ActiveCell celltype = m_markers[level](adjcell[0], adjcell[1]);

						Vec2R cpos = m_markers[level].idx_to_ws(Vec2R(i, j));
						
						if (celltype == ACTIVE)
						{
							Vec2R apos = m_markers[level].idx_to_ws(Vec2R(adjcell));
							startpoints.push_back(cpos);
							endpoints.push_back(apos);
						}
						else if (celltype == UP)
						{
							Vec2st parent = get_parent(Vec2st(adjcell));
							assert(m_markers[level + 1](parent[0], parent[1]) == ACTIVE);

							Vec2R apos = m_markers[level + 1].idx_to_ws(Vec2R(parent));
							startpoints.push_back(cpos);
							endpoints.push_back(apos);
						}
						else if (celltype == DOWN)
						{
							for (size_t cc = 0; cc < 2; ++cc)
							{
								Vec2st child = get_child(Vec2st(adjcell), adjacent_offset[c][cc]);
								assert(m_markers[level - 1](child[0], child[1]) == ACTIVE);
								Vec2R apos = m_markers[level - 1].idx_to_ws(Vec2R(child));

								startpoints.push_back(cpos);
								endpoints.push_back(apos);
							}
						}
					}
				}
			}
	}

	renderer.add_lines(startpoints, endpoints, Vec3f(0, 0, 1));
}

void OctreeGrid::draw_grid(Renderer& renderer, const Vec3f& colour) const
{
	size_t count = 0;
	size_t maxlevels = m_markers.size();
	for (size_t level = 0; level < maxlevels; ++level)
	{
		Vec2st size = m_markers[level].size();

		for (size_t i = 0; i < size[0]; ++i)
			for (size_t j = 0; j < size[1]; ++j)
			{
				if (m_markers[level](i, j) == ACTIVE)
				{
					m_markers[level].draw_grid_cell(renderer, Vec2st(i, j), colour);
					/*Vec2R pos = m_markers[level].idx_to_ws(Vec2R(i, j));
					renderer.add_point(pos, colours[level], 5);*/

					++count;
				}
			}
	}

	std::cout << "Grid count " << count << std::endl;
}

std::vector<Vec3st> OctreeGrid::get_face_adjacent_cells(const Vec2st& coord, size_t level, size_t dir) const
{
	assert(dir < 4);

	std::vector<Vec3st> adjcells;

	Vec2i acell = Vec2i(coord[0], coord[1]) + cell_offset[dir];

	ActiveCell celltype = m_markers[level](acell[0], acell[1]);
	if (celltype == ACTIVE || celltype == INACTIVE)
	{
		adjcells.push_back(Vec3st(acell[0], acell[1], level));
	}
	else if (celltype == UP)
	{
		Vec2st parent = get_parent(Vec2st(acell[0], acell[1]));
		assert(m_markers[level + 1](parent[0], parent[1]) == ACTIVE);

		adjcells.push_back(Vec3st(parent[0], parent[1], level + 1));
	}
	else if (celltype == DOWN)
	{
		for (size_t cc = 0; cc < 2; ++cc)
		{
			Vec2st child = get_child(Vec2st(acell[0], acell[1]), adjacent_offset[dir][cc]);

			assert(m_markers[level - 1](child[0], child[1]) == ACTIVE || m_markers[level - 1](child[0], child[1]) == INACTIVE);

			adjcells.push_back(Vec3st(child[0], child[1], level - 1));
		}
	}

	return std::move(adjcells);
}

void OctreeGrid::refine_grid()
{
	Transform tempxform(xform(0).dx() / 2., xform(0).offset());
	Vec2st tempnx = size(0) * 2;
	size_t maxlevel = m_markers.size();

	// Set up new marker grids
	std::vector<ScalarGrid<ActiveCell>> newmarkers;
	newmarkers.resize(maxlevel);
	newmarkers[0] = ScalarGrid<ActiveCell>(tempxform, tempnx, UP);

	Real dx = tempxform.dx();
	Vec2st nx = tempnx;
	for (size_t i = 1; i < maxlevel; ++i)
	{
		dx *= 2.;
		nx = nx / 2;

		Transform tmpxform(dx, xform(0).offset());

		newmarkers[i] = ScalarGrid<ActiveCell>(tmpxform, nx, UP);
	}
	
	// Fill new market grids based on existing but one octave lower
	for (size_t level = 0; level < maxlevel; ++level)
	{
		Vec2st size = newmarkers[level].size();
		
		for (size_t i = 0; i < size[0]; ++i)
			for (size_t j = 0; j < size[1]; ++j)
			{
				Vec2st parent = get_parent(Vec2st(i, j));

				newmarkers[level](i, j) = m_markers[level](parent[0], parent[1]);
			}
	}

	m_markers = std::move(newmarkers);
}
#include <limits>
#include <utility>

#include "tbb/tbb.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "LevelSet2D.h"
#include "VectorGrid.h"
#include "Predicates.h"

void LevelSet2D::draw_grid(Renderer& renderer) const
{
	m_phi.draw_grid(renderer);
}

void LevelSet2D::draw_mesh_grid(Renderer& renderer) const
{
	Transform xform(m_phi.dx(), m_phi.offset() + Vec2R(0.5) * m_phi.dx());
	ScalarGrid<int> temp_grid(xform, m_phi.size());
	temp_grid.draw_grid(renderer);
}

void LevelSet2D::draw_supersampled_values(Renderer& renderer, Real radius, unsigned samples, unsigned size) const
{
	m_phi.draw_supersampled_values(renderer, radius, samples, size);
}
void LevelSet2D::draw_normals(Renderer& renderer, const Vec3f& colour, Real length) const
{
	m_phi.draw_sample_gradients(renderer, colour, length);
}

void LevelSet2D::draw_surface(Renderer& renderer, const Vec3f& colour)
{
	Mesh2D surface;
	
	extract_mesh(surface);
	surface.draw_mesh(renderer, colour);
}

void LevelSet2D::draw_dc_surface(Renderer& renderer, const Vec3f& colour)
{
	Mesh2D surface;
	extract_dc_mesh(surface);
	surface.draw_mesh(renderer, colour, true, true, Vec3f(1.,0,0));
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.

Vec2R LevelSet2D::interface_search(const Vec2R& world_pos, unsigned iter_limit) const
{
	Real dist = m_phi.interp(world_pos);
	unsigned count = 0;
	Real epsilon = 1E-2 * dx();
	Vec2R temp_pos = world_pos;

	if (fabs(dist) < m_narrow_band)
	{
		while (fabs(dist) > epsilon && count < iter_limit)
		{
			temp_pos -= dist * normal(temp_pos);
			dist = m_phi.interp(temp_pos);
			++count;
		}
	}

	return temp_pos;
}

Vec2R LevelSet2D::interface_search_idx(const Vec2R& index_pos, unsigned iter_limit) const
{
	Vec2R world_pos = idx_to_ws(index_pos);
	world_pos = interface_search(world_pos, iter_limit);
	return ws_to_idx(world_pos);
}

auto vec_sort = [](const Vec2ui &a, const Vec2ui &b) -> bool
{
	if (a[0] < b[0]) return true;
	else if (a[0] == b[0] && a[1] < b[1]) return true;
	return false;
};

void LevelSet2D::reinitFIM()
{
	UniformGrid<MarkedCells> marked_cells(size(), MarkedCells::UNVISITED);
	
	// Find the zero crossings, update their distances and flag as source cells
	ScalarGrid<Real> temp_phi = m_phi;

	unsigned voxels = size()[0] * size()[1];

	tbb::parallel_for(tbb::blocked_range<unsigned>(0, voxels), [&](const tbb::blocked_range<unsigned> &range)
	{
		for (unsigned r = range.begin(); r != range.end(); ++r)
		{
			Vec2ui idx = marked_cells.unflatten(r);

			// Check for a zero crossing
			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2i cidx;

				cidx = Vec2i(idx) + cell_to_cell[dir];

				unsigned axis = dir / 2;

				if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;

				if (m_phi(idx) * m_phi(Vec2ui(cidx)) <= 0.)
				{
					// Update distance
					Vec2R index_pos = idx_to_ws(Vec2R(idx));
					Vec2R int_pos = interface_search(index_pos, 5);
					
					Real udf = dist(int_pos, index_pos);

					// If the cell has not be updated yet OR the update is lower than a previous
					// update, assign the SDF to the cell
					if (marked_cells(idx) != MarkedCells::FINISHED)
					{
						marked_cells(idx) = MarkedCells::FINISHED;
						temp_phi(idx) = udf;
					}
					else if (temp_phi(idx) > udf)
						temp_phi(idx) = udf;
				}
			}

			Real val;
			if (marked_cells(idx) == MarkedCells::FINISHED)
			{
				val = temp_phi(idx);

				// Add neighbours to the list
				for (unsigned dir = 0; dir < 4; ++dir)
				{
					Vec2i cidx = Vec2i(idx) + cell_to_cell[dir];

					unsigned axis = dir / 2;
					if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;
				}
			}
			else // Overwrite cell data to background distance if not at a zero crossing
				val = m_narrow_band;

			temp_phi(idx) = (m_phi(idx) < 0.) ? -val : val;
		}
	});

	std::swap(m_phi, temp_phi);

	fast_iterative(marked_cells);
}

void LevelSet2D::fast_iterative(UniformGrid<MarkedCells> &marked_cells)
{
	assert(marked_cells.size() == size());

	//
	// Before starting the iterations, we want to construct the active list of voxels
	// to reinitialize.
	//

	std::vector<Vec2ui> active_list;


	unsigned voxels = size()[0] * size()[1];

	tbb::enumerable_thread_specific<std::vector<Vec2ui>> parallel_active_list;

	tbb::parallel_for(tbb::blocked_range<unsigned>(0, voxels), [&](const tbb::blocked_range<unsigned> &range)
	{
		std::vector<Vec2ui> &local_active_list = parallel_active_list.local();

		for (unsigned r = range.begin(); r != range.end(); ++r)
		{
			Vec2ui idx = marked_cells.unflatten(r);

			if (marked_cells(idx) == MarkedCells::FINISHED)
			{
				// Add neighbours to the list
				for (unsigned dir = 0; dir < 4; ++dir)
				{
					Vec2i cidx = Vec2i(idx) + cell_to_cell[dir];

					unsigned axis = dir / 2;
					if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;

					Vec2ui ucidx(cidx);
					if (marked_cells(Vec2ui(ucidx)) == MarkedCells::UNVISITED)
					{
						local_active_list.push_back(ucidx);
						marked_cells(ucidx) = MarkedCells::VISITED;
					}
				}
			}
		}
	});

	for (auto i = parallel_active_list.begin(); i != parallel_active_list.end(); ++i)
	{
		active_list.insert(active_list.end(), i->begin(), i->end());
	}

	parallel_active_list.clear();

	tbb::parallel_sort(active_list.begin(), active_list.end(), vec_sort);


	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward.

	auto solveEikonal = [&](const Vec2i& idx) -> Real
	{
		Real max = std::numeric_limits<Real>::max();

		Real U_bx = (idx[0] > 0) ? fabs(m_phi(idx[0] - 1, idx[1])) : max;
		Real U_fx = (idx[0] < size()[0] - 1) ? fabs(m_phi(idx[0] + 1, idx[1])) : max;

		Real U_by = (idx[1] > 0) ? fabs(m_phi(idx[0], idx[1] - 1)) : max;
		Real U_fy = (idx[1] < size()[1] - 1) ? fabs(m_phi(idx[0], idx[1] + 1)) : max;

		Real Ux = std::min(U_bx, U_fx);
		Real Uy = std::min(U_by, U_fy);
		Real U;
		if (fabs(Ux - Uy) >= dx())
			U = std::min(Ux, Uy) + dx();
		else
			// Quadratic equation from the Eikonal
			//U = (Uh + Uv) / 2 + .5 * sqrt(pow(Uh + Uv, 2) - 2 * (Uh*Uh + Uv*Uv - dx()*dx()));
			U = .5 * (Ux + Uy + sqrt(2. * dx() * dx() - pow(Ux - Uy, 2)));
		return U;
	};

	ScalarGrid<Real> temp_phi = m_phi;

	Real tol = dx() * 1E-5;
	bool active_cells = true;

	unsigned active_count = active_list.size();

	unsigned loopcount = 0;
	unsigned maxcount = 5 * m_narrow_band / dx();

	while (active_count > 0 && loopcount < maxcount)
	{
		tbb::parallel_for(tbb::blocked_range<unsigned>(0, active_count), [&](const tbb::blocked_range<unsigned> &range)
		{
			std::vector<Vec2ui> &local_list = parallel_active_list.local();
			
			Vec2ui old_idx(m_phi.size());

			for (unsigned r = range.begin(); r != range.end(); ++r)
			{
				Vec2ui idx = active_list[r];

				if (old_idx != idx)
				{
					assert(marked_cells(idx) == MarkedCells::VISITED);

					Real eikonal_phi = solveEikonal(Vec2i(idx));

					// If we hit the narrow band, we don't need to make any changes
					if (eikonal_phi > m_narrow_band) continue;

					temp_phi(idx) = m_phi(idx) < 0 ? -eikonal_phi : eikonal_phi;

					// Check if new phi is converged
					Real phi_val = m_phi(idx);

					Real cval = fabs(eikonal_phi - fabs(m_phi(idx)));

					// If the cell is converged, load up the neighbours that aren't currently being VISITED
					if (fabs(eikonal_phi - fabs(phi_val)) < tol)
					{
						for (unsigned dir = 0; dir < 4; ++dir)
						{
							Vec2i cidx = Vec2i(idx) + cell_to_cell[dir];

							unsigned axis = dir / 2;

							if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;

							Vec2ui ucidx(cidx);
							if (marked_cells(ucidx) == MarkedCells::UNVISITED)
							{
								Real adjacent_eikonal_phi = solveEikonal(cidx);

								// Check if new phi is less than the current value
								Real pval = fabs(m_phi(ucidx));

								if (adjacent_eikonal_phi < fabs(m_phi(ucidx)) && fabs(adjacent_eikonal_phi - fabs(m_phi(ucidx))) > tol)
								{
									temp_phi(ucidx) = m_phi(ucidx) < 0 ? -adjacent_eikonal_phi : adjacent_eikonal_phi;

									local_list.push_back(ucidx);
								}
							}
						}
					}
					// If the cell hasn't converged, toss it back into the list
					else
						local_list.push_back(idx);
				}

				old_idx = idx;
			}
		});

		// Turn off VISITED labels for current list
		tbb::parallel_for(tbb::blocked_range<unsigned>(0, active_count), [&](const tbb::blocked_range<unsigned> &range)
		{
			for (unsigned r = range.begin(); r != range.end(); ++r)
			{
				Vec2ui idx = active_list[r];
				assert(marked_cells(idx) != MarkedCells::FINISHED);
				marked_cells(idx) = MarkedCells::UNVISITED;
			}
		});

		active_list.clear();

		for (auto i = parallel_active_list.begin(); i != parallel_active_list.end(); ++i)
		{
			active_list.insert(active_list.end(), i->begin(), i->end());
		}

		parallel_active_list.clear();

		tbb::parallel_sort(active_list.begin(), active_list.end(), vec_sort);
		
		active_count = active_list.size();

		// Turn on VISITED labels for new list
		tbb::parallel_for(tbb::blocked_range<unsigned>(0, active_count), [&](const tbb::blocked_range<unsigned> &range)
		{
			for (unsigned r = range.begin(); r != range.end(); ++r)
			{
				Vec2ui idx = active_list[r];
				assert(marked_cells(idx) != MarkedCells::FINISHED);
				marked_cells(idx) = MarkedCells::VISITED;
			}
		}); 
		
		std::swap(temp_phi, m_phi);

		++loopcount;
	}
}

void LevelSet2D::reinit()
{	
	ScalarGrid<Real> temp_phi = m_phi;

	UniformGrid<MarkedCells> marked_cells(size(), MarkedCells::UNVISITED);

	// Find zero crossing
	for_each_voxel_range(Vec2ui(0), m_phi.size(), [&](const Vec2ui& cell)
	{
		for (unsigned dir = 0; dir < 4; ++dir)
		{
			Vec2i cidx = Vec2i(cell) + cell_to_cell[dir];

			unsigned axis = dir / 2;
			if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;
			
			Vec2ui ucdix(cidx);
			if (m_phi(cell) * m_phi(ucdix) <= 0.)
			{
				// Update distance
				Vec2R int_pos = interface_search(idx_to_ws(Vec2R(cell)), 5);
				Real udf = dist(int_pos, idx_to_ws(Vec2R(cell)));

				// If the cell has not be updated yet OR the update is lower than a previous
				// update, assign the SDF to the cell
				if (marked_cells(cell) != MarkedCells::FINISHED || abs(temp_phi(cell)) > udf)
				{
					temp_phi(cell) = (m_phi(cell) < 0.) ? -udf : udf;
					marked_cells(cell) = MarkedCells::FINISHED;
				}
			}

		}
		if (marked_cells(cell) != MarkedCells::FINISHED)
		{
			//Real max = std::numeric_limits<Real>::max();
			temp_phi(cell) = (m_phi(cell) < 0.) ? -m_narrow_band : m_narrow_band;
		}
	});

	m_phi = temp_phi;
	fast_iterative(marked_cells);
}

void LevelSet2D::init(const Mesh2D& init_mesh, bool resize)
{
	if (resize)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		Vec2R bb_min(std::numeric_limits<Real>::max());
		Vec2R bb_max(std::numeric_limits<Real>::lowest());

		for (unsigned v = 0; v < init_mesh.vert_size(); ++v) update_both_minmax(bb_min, bb_max, init_mesh.vert(v).point());

		// Just for nice whole numbers, let's clamp the bounding box to be an integer
		// offset in index space then bring it back to world space
		Real max_nb = 10.;
		max_nb = std::min(m_narrow_band/dx(), max_nb);

		bb_min = (Vec2R(floor(bb_min / dx())) - Vec2R(max_nb)) * dx();
		bb_max = (Vec2R(ceil(bb_max / dx())) + Vec2R(max_nb)) * dx();

		clear();
		Transform xform(dx(), bb_min);
		// Since we know how big the mesh is, we know how big our grid needs to be (wrt to grid spacing)
		m_phi = ScalarGrid<Real>(xform, Vec2ui((bb_max - bb_min) / dx()), m_narrow_band);
	}
	else m_phi.resize(size(), m_narrow_band);

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED. The following pass will do a 
	// BFS to assign the remaining UNVISITED cells with the appropriate distances.
	UniformGrid<MarkedCells> marked_cells(size(), MarkedCells::UNVISITED);
	UniformGrid<int> parity_cells(size(), 0);

	for (auto edge : init_mesh.edges())
	{
		// It's easier to work in our index space and just scale the distance later.
		const Vec2R v0 = ws_to_idx(init_mesh.vert(edge.vert(0)).point());
		const Vec2R v1 = ws_to_idx(init_mesh.vert(edge.vert(1)).point());

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		Vec2R vmin, vmax;
		minmax(vmin, vmax, v0, v1);

		Vec2R bb_cmin = ceil(vmin);
		Vec2R bb_fmin = floor(vmin) - Vec2R(1);
		Vec2R bb_fmax = floor(vmax);

		for (int j = bb_cmin[1]; j <= bb_fmax[1]; ++j)
			for (int i = bb_fmax[0]; i >= bb_fmin[0]; --i)
			{
				Vec2R grid_node(i, j);
				Intersection result = exact_edge_intersect(v0, v1, grid_node);

				if (result == Intersection::NO) continue;
				else if (result == Intersection::YES)
				{
					if (v0[1] < v1[1])
					{
						// Decrement the parity and increment since the grid_node is
						// "left" of the mesh-edge crossing the grid-edge
						parity_cells(i + 1, j) += 1;
					}
					else
					{
						parity_cells(i + 1, j) -= 1;
					}
				}
				// If the grid node is explicitly on the mesh-edge, set distance to zero
				// since it might not be exactly zero due to floating point error above.
				else if (result == Intersection::ON)
				{
					// Technically speaking, the zero isocountour means we're inside
					// the surface. So we should change the parity at the node that
					// is intersected even though it's zero and therefore the sign
					// is meaningless.
					if (v0[1] < v1[1]) // grid_node is to the "left" of the edge
					{
						// Decrement the parity and increment since the grid_node is
						// "left" of the mesh-edge crossing the grid-edge
						parity_cells(i, j) += 1;
					}
					else
					{
						parity_cells(i, j) -= 1;
					}

					marked_cells(i, j) = MarkedCells::FINISHED;
					m_phi(i, j) = 0.;
				}

				break;
			}
	}

	// Now that all the x-axis edge crossings have been found, we can compile the parity changes
	// and label grid nodes that are at the interface
	for (unsigned j = 0; j < size()[1]; ++j)
	{
		int parity = m_inverted ? 1 : 0;

		// We loop x-major because that's how we've set up our edge intersection.
		// This is definitely cache incoherent based on the UniformGrid structure..
		// It's probably not a deal breaker in 2-D but it could suck in 3-D
		for (unsigned i = 0; i < size()[0]; ++i) // TODO: double check that I've resized right
		{
			// Update parity before changing sign since the parity values above used the convention
			// of putting the change on the "far" node (i.e. after the mesh-grid intersection).

			Vec2ui cell(i, j);
			parity += parity_cells(cell);
			parity_cells(cell) = parity;
			if (parity > 0) m_phi(cell) = -m_phi(cell);
		}
	}

	// With the parity assigned, loop over the grid once more and label nodes that have a sign change
	// with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
	for (unsigned i = 1; i < (size()[0] - 1); ++i)
		for (unsigned j = 1; j < (size()[1] - 1); ++j)
		{
			const Vec2ui cell(i, j);
			bool change = false;
			int parity = parity_cells(cell);
			bool l_inside = (parity > 0);

			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2i adjacent_cell = Vec2i(cell) + cell_to_cell[dir];

				bool n_inside = (parity_cells(Vec2ui(adjacent_cell)) > 0);

				// Flag to indicate that this cell needs an exact distance based on the mesh
				if (l_inside != n_inside) marked_cells(cell) = MarkedCells::FINISHED;
			}
		}

	// Loop over all the edges in the mesh. Level set grid cells labelled as VISITED will be
	// updated with the distance to the surface if it happens to be shorter than the current
	// distance to the surface.
	for (auto edge : init_mesh.edges())
	{
		// Using the vertices of the edge, we can update distance values for cells
		// within the bounding box of the mesh. It's easier to work in our index space
		// and just scale the distance later.
		const Vec2R v0 = ws_to_idx(init_mesh.vert(edge.vert(0)).point());
		const Vec2R v1 = ws_to_idx(init_mesh.vert(edge.vert(1)).point());

		// Build bounding box

		Vec2R bbmin = floor(min_union(v0, v1)) - Vec2R(2);
		bbmin = max_union(bbmin, Vec2R(0));
		Vec2R bbmax = ceil(max_union(v0, v1)) + Vec2R(2);
		Vec2R top(size()[0] - 1, size()[1] - 1);
		bbmax = min_union(bbmax, top);

		// Update distances to the mesh at grid cells within the bounding box
		assert(bbmin[0] >= 0 && bbmin[1] >= 0 && bbmax[0] < size()[0] && bbmax[1] < size()[1]);
		for (unsigned i = bbmin[0]; i <= bbmax[0]; ++i)
			for (unsigned j = bbmin[1]; j <= bbmax[1]; ++j)
			{
				Vec2ui cell(i, j);
				if (marked_cells(cell) != MarkedCells::UNVISITED)
				{
					Vec2R p(cell);
					Vec2R va = p - v0;
					Vec2R vb = v1 - v0;

					Real c = (dot(va, vb) / dot(vb, vb)); // Find vector coefficient from v0 to v1
					c = ((c < 0.) ? 0. : ((c > 1.) ? 1. : c)); // Truncate scale to be within v0-v1 segment

					Real dist = mag(va - c*vb) * dx();

					// If the parity says the node is inside, set it to be negative
					if (fabs(m_phi(cell)) > dist)
					{
						m_phi(cell) = (parity_cells(cell) > 0) ? -dist : dist;
					}
				}
			}
	}

	fast_marching(marked_cells);
}

void LevelSet2D::fast_marching(UniformGrid<MarkedCells>& marked_cells)
{
	assert(marked_cells.size() == size());

	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward
	auto solveEikonal = [&](const Vec2i& idx) -> Real
	{
		Real max = std::numeric_limits<Real>::max();
		Real U_bx = (idx[0] > 0) ? fabs(m_phi(idx[0] - 1, idx[1])) : max;
		Real U_fx = (idx[0] < size()[0] - 1) ? fabs(m_phi(idx[0] + 1, idx[1])) : max;

		Real U_by = (idx[1] > 0) ? fabs(m_phi(idx[0], idx[1] - 1)) : max;
		Real U_fy = (idx[1] < size()[1] - 1) ? fabs(m_phi(idx[0], idx[1] + 1)) : max;

		Real Uh = min(U_bx, U_fx);
		Real Uv = min(U_by, U_fy);
		Real U;
		if (fabs(Uh - Uv) >= dx())
			U = min(Uh, Uv) + dx();
		else
			// Quadratic equation from the Eikonal
			U = (Uh + Uv) / 2 + .5 * sqrt(pow(Uh + Uv, 2) - 2 * (Uh*Uh + Uv*Uv - dx()*dx()));

		return U;
	};

	// Load up the BFS queue with the unvisited cells next to the finished ones
	using Node = std::pair<Vec2ui, Real>;
	auto cmp = [](const Node& a, const Node& b) -> bool { return fabs(a.second) > fabs(b.second); };
	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> marker_q(cmp);

	for_each_voxel_range(Vec2ui(0), size(), [&](const Vec2ui& cell)
	{
		if (marked_cells(cell) == MarkedCells::FINISHED)
		{
			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2i cidx = Vec2i(cell) + cell_to_cell[dir];
				
				//Boundary check
				unsigned axis = dir / 2;
				if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;

				Vec2ui ucidx(cidx);
				//store non-marked cells in the queue
				if (marked_cells(ucidx) == MarkedCells::UNVISITED)
				{
					Real dist = solveEikonal(cidx);
					m_phi(ucidx) = (m_phi(ucidx) <= 0.) ? -dist : dist;

					assert(dist >= 0);
					Node node(ucidx, dist);

					marker_q.push(node);
					marked_cells(ucidx) = MarkedCells::VISITED;
				}
			}
		}
	});

	while (!marker_q.empty())
	{
		Node curr_node = marker_q.top();
		Vec2ui idx = curr_node.first;
		marker_q.pop();

		// Since you can't just update parts of the priority queue,
		// it's possible that a cell has been solidified at a smaller distance
		// and an older insert if floating around.
		if (marked_cells(idx) == MarkedCells::FINISHED)
		{
			// Make sure that the distance assigned to the cell is smaller than
			// what is floating around
			assert(fabs(m_phi(idx)) <= fabs(curr_node.second));
			continue;
		}
		assert(marked_cells(idx) == MarkedCells::VISITED);

		if (fabs(m_phi(idx)) < m_narrow_band)
		{
			// Debug check that there is indeed a FINISHED cell next to it
			bool marked = false;

			// Loop over the neighbouring cells and load the unvisited cells
			// and update the visited cells
			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2i cidx = Vec2i(idx) + cell_to_cell[dir];

				//Boundary check
				unsigned axis = dir / 2;

				if (cidx[axis] < 0 || cidx[axis] >= size()[axis]) continue;

				Vec2ui ucidx(cidx);

				if (marked_cells(ucidx) == MarkedCells::FINISHED)
					marked = true;
				else // If visited, then we'll update it
				{
					Real dist = solveEikonal(cidx);
					assert(dist >= 0);
					if (dist > m_narrow_band) dist = m_narrow_band;

					// If the computed distance is greater than the existing distance, we can skip it
					if (marked_cells(ucidx) == MarkedCells::VISITED && dist > fabs(m_phi(ucidx)))
						continue;

					m_phi(ucidx) = (m_phi(ucidx) < 0.) ? -dist : dist;

					Node node(ucidx, dist);

					marker_q.push(node);
					marked_cells(ucidx) = MarkedCells::VISITED;
				}
			}

			//Check that a marked cell was indeed visited
			assert(marked);
		}
		// Clamp to narrow band
		else
		{
			m_phi(idx) = (m_phi(idx) < 0.) ? -m_narrow_band : m_narrow_band;
		}

		// Solidify cell now that we've handled all it's neighbours
		marked_cells(idx) = MarkedCells::FINISHED;
	}
}

//
// Marching cubes stencil
//

static const Vec2ui edge_node_map[4] =
{ Vec2ui(0,1), Vec2ui(1,2), Vec2ui(2,3), Vec2ui(3,0) };
//
static const Vec2ui node_in_cell[4] =
{ Vec2ui(0,0), Vec2ui(1,0), Vec2ui(1,1), Vec2ui(0,1) };


// Extract a mesh representation of the interface. Useful for rendering but not much
// else since there will be duplicate vertices per grid edge in the current implementation.

void LevelSet2D::extract_mesh(Mesh2D& surf) const
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;
	
	verts.reserve(surf.vert_size());
	edges.reserve(surf.edge_size());

	surf.clear();
	
	// Run marching squares loop
	for_each_voxel_range(Vec2ui(0), size() - Vec2ui(1), [&](const Vec2ui& cell)
	{
		unsigned mc_idx = 0;

		for (unsigned dir = 0; dir < 4; ++dir)
		{
			Vec2ui node = cell + node_in_cell[dir];
			if (m_phi(node) <= 0.)
			{
				mc_idx += (1 << dir);
			}
		}

		// Connect edges using the marching squares template
		for (unsigned eidx = 0; eidx < 4 && mc_template[mc_idx][eidx] >= 0; eidx += 2)
		{
			// Find first vertex
			unsigned edge = mc_template[mc_idx][eidx];
			Vec2ui node0 = cell + node_in_cell[edge_node_map[edge][0]];
			Vec2ui node1 = cell + node_in_cell[edge_node_map[edge][1]];

			Vec2R v0 = interp_interface(node0, node1);

			// Find second vertex
			edge = mc_template[mc_idx][eidx + 1];
			node0 = cell + node_in_cell[edge_node_map[edge][0]];
			node1 = cell + node_in_cell[edge_node_map[edge][1]];

			Vec2R v1 = interp_interface(node0, node1);

			// Store vertices
			Vec2R v0_ws = idx_to_ws(v0);
			Vec2R v1_ws = idx_to_ws(v1);

			verts.push_back(v0_ws);
			verts.push_back(v1_ws);

			edges.push_back(Vec2ui(verts.size() - 2, verts.size() - 1));
		}
	});

	surf = Mesh2D(edges, verts);
}

// Extract a mesh representation of the interface using dual contouring
void LevelSet2D::extract_dc_mesh(Mesh2D& surf) const
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;

	verts.reserve(surf.vert_size());
	edges.reserve(surf.edge_size());

	surf.clear();

	// Create grid to store index to dual contouring point. Note that phi is
	// center sampled so the DC grid must be node sampled and one cell shorter
	// in each dimension
	UniformGrid<unsigned> dc_points(size() - Vec2ui(1), -1);

	// Run dual contouring loop
	for_each_voxel_range(Vec2ui(0), dc_points.size(), [&](const Vec2ui& cell)
	{
		std::vector<Vec2R> qef_points;
		std::vector<Vec2R> qef_norms;

		// Find zero crossings
		for (unsigned eidx = 0; eidx < 4; ++eidx)
		{
			Vec2ui bidx = cell + node_in_cell[edge_node_map[eidx][0]];
			Vec2ui fidx = cell + node_in_cell[edge_node_map[eidx][1]];

			// Look for zero crossings
			if ((m_phi(bidx) <= 0.) && (m_phi(fidx) > 0.) ||
				(m_phi(bidx) > 0.) && (m_phi(fidx) <= 0.))
			{
				// Find interface point
				Vec2R v0 = interp_interface(bidx, fidx);
				qef_points.push_back(v0);

				// Find associated surface normal
				Vec2R norm = normal(idx_to_ws(v0));
				qef_norms.push_back(norm);
			}
		}

		if (qef_points.size() > 0)
		{
			Eigen::MatrixXd A(qef_points.size(), 2);
			Eigen::VectorXd b(qef_points.size());
			Eigen::VectorXd xcom = Eigen::VectorXd::Zero(2);

			assert(qef_points.size() > 1);
			for (unsigned p = 0; p < qef_points.size(); ++p)
			{
				A(p, 0) = qef_norms[p][0]; A(p, 1) = qef_norms[p][1];
				b(p) = dot(qef_norms[p], qef_points[p]);
				xcom[0] += qef_points[p][0]; xcom[1] += qef_points[p][1];
			}

			xcom /= Real(qef_points.size());

			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			svd.setThreshold(1E-2);

			Eigen::VectorXd np = xcom + svd.solve(b - A * xcom);

			Vec2R cm(xcom[0], xcom[1]);

			Vec2R bbmin = floor(cm);
			Vec2R bbmax = ceil(cm);

			if (np[0] < bbmin[0] || np[1] < bbmin[1] ||
				np[0] > bbmax[0] || np[1] > bbmax[1]) np = xcom;

			verts.push_back(idx_to_ws(Vec2R(np[0], np[1])));
			dc_points(cell) = verts.size() - 1;
		}
	});
	
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		Vec2ui start(0); start[axis] = 1;
		Vec2ui end(size() - Vec2ui(1));

		for_each_voxel_range(start, end, [&](const Vec2ui& cell)
		{
			Vec2ui bidx = cell;
			Vec2ui fidx = cell; fidx[(axis + 1) % 2] += 1;

			// Look for zero crossings
			if ((m_phi(bidx) <= 0.) && (m_phi(fidx) > 0.) ||
				(m_phi(bidx) > 0.) && (m_phi(fidx) <= 0.))
			{
				// Confirm that both adjacent cells have indexed vertices
				Vec2i bcidx = Vec2i(cell) + face_to_cell[axis][0];
				Vec2i fcidx = Vec2i(cell) + face_to_cell[axis][1];

				assert(dc_points(Vec2ui(bcidx)) >= 0 && dc_points(Vec2ui(fcidx)) >= 0);

				if (m_phi(bidx) <= 0.) edges.push_back(Vec2ui(dc_points(Vec2ui(bcidx)), dc_points(Vec2ui(fcidx))));
				else edges.push_back(Vec2ui(dc_points(Vec2ui(fcidx)), dc_points(Vec2ui(bcidx))));
			}
		});
	}

	surf = Mesh2D(edges, verts);
}

Vec2R LevelSet2D::interp_interface(const Vec2ui& i0, const Vec2ui& i1) const
{
	assert(m_phi(i0[0], i0[1]) * m_phi(i1[0], i1[1]) <= 0.0);
	//Find weight to zero isosurface
	Real s = m_phi(i0[0], i0[1]) / (m_phi(i0[0], i0[1]) - m_phi(i1[0], i1[1]));

	if (s < 0.0) s = 0.0;
	else if (s > 1.0) s = 1.0;

	Vec2R dx = Vec2R(i1) - Vec2R(i0);
	return Vec2R(i0[0], i0[1]) + s*dx;
}

void LevelSet2D::surface_union(const LevelSet2D& input_volume)
{
	for_each_voxel_range(Vec2ui(0), size(), [&](const Vec2ui& cell)
	{
		m_phi(cell) = std::min(m_phi(cell), input_volume.interp(idx_to_ws(Vec2R(cell))));
	});
}

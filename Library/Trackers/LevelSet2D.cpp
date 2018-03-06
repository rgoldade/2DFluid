#include <queue>

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
	Transform xform(m_phi.dx(), m_phi.offset() + Vec2R(0.5)*m_phi.dx());
	ScalarGrid<int> temp_grid(xform, m_phi.size());
	temp_grid.draw_grid(renderer);
}

void LevelSet2D::draw_supersampled_values(Renderer& renderer, Real radius, size_t samples, size_t size) const
{
	m_phi.draw_supersampled_values(renderer, radius, samples, size);
}
void LevelSet2D::draw_normals(Renderer& renderer, const Vec3f& colour, Real length) const
{
	m_phi.draw_sample_gradients(renderer, colour, length);
}

void LevelSet2D::draw_surface(Renderer& renderer, const Vec3f& colour)
{
	if (!m_mesh_set) extract_mesh(m_surface);
	m_mesh_set = true;
	m_surface.draw_mesh(renderer, colour);
}

void LevelSet2D::draw_dc_surface(Renderer& renderer, const Vec3f& colour)
{
	extract_dc_mesh(m_surface);
	m_surface.draw_mesh(renderer, colour, true, true, Vec3f(1.,0,0));
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.

Vec2R LevelSet2D::interface_search(Vec2R wpos, size_t iter_limit) const
{
	Real dist = m_phi.interp(wpos);
	size_t count = 0;
	Real epsilon = 1E-2 * dx();

	if (fabs(dist) < m_nb)
	{
		while (fabs(dist) > epsilon && count < iter_limit)
		{
			// Need a const, thread safe normal call (i.e. doesn't build the gradient field even if dirty)
			wpos -= dist * normal_const(wpos);
			dist = m_phi.interp(wpos);
			++count;
		}
	}

	return wpos;
}

Vec2R LevelSet2D::interface_search_idx(const Vec2R& pos, size_t iter_limit) const
{
	Vec2R fpos = ws_to_idx(pos);
	fpos = interface_search(fpos);
	return ws_to_idx(fpos);
}

void LevelSet2D::reinitFIM(bool force_rebuild)
{
	if (!m_dirty_surface && !force_rebuild) return;

	UniformGrid<marked> marked_cells(size(), UNVISITED);
	
	// Find the zero crossings, update their distances and flag as source cells
	ScalarGrid<Real> temp_phi = m_phi;

	build_gradient();

	size_t grain_size = 500;
	size_t voxels = size()[0] * size()[1];

	tbb::enumerable_thread_specific<std::vector<Vec2st>> parallel_active_list;
	tbb::enumerable_thread_specific<std::vector<Vec2st>> parallel_source_list;

	tbb::parallel_for(tbb::blocked_range<size_t>(0, voxels, grain_size), [&](const tbb::blocked_range<size_t> &range)
	{
		std::vector<Vec2st> &local_list = parallel_source_list.local();
		std::vector<Vec2st> &active_local_list = parallel_active_list.local();
		for (size_t r = range.begin(); r != range.end(); ++r)
		{
			Vec2st idx = marked_cells.unstride(r);

			// Check for a zero crossing
			for (size_t dir = 0; dir < 4; ++dir)
			{
				Vec2st cidx;

				cidx[0] = idx[0] + cell_offset[dir][0];
				cidx[1] = idx[1] + cell_offset[dir][1];

				if (cidx[0] < 0 || cidx[1] < 0 || cidx[0] >= size()[0] || cidx[1] >= size()[1]) continue;

				if (m_phi(idx[0], idx[1]) * m_phi(cidx[0], cidx[1]) <= 0.)
				{
					// Update distance
					Vec2R ipos(idx);
					Vec2R int_pos = interface_search(idx_to_ws(ipos), 5);
					Real udf = dist(int_pos, idx_to_ws(ipos));

					// If the cell has not be updated yet OR the update is lower than a previous
					// update, assign the SDF to the cell
					if (marked_cells(idx[0], idx[1]) != FINISHED)
					{
						marked_cells(idx[0], idx[1]) = FINISHED;
						temp_phi(idx[0], idx[1]) = udf;
						local_list.push_back(idx);
					}
					else if (temp_phi(idx[0], idx[1]) > udf)
						temp_phi(idx[0], idx[1]) = udf;
				}
			}
			Real val;
			if (marked_cells(idx[0], idx[1]) == FINISHED)
			{
				val = temp_phi(idx[0], idx[1]);

				// Add neighbours to the list
				for (size_t dir = 0; dir < 4; ++dir)
				{
					Vec2st cidx;

					cidx[0] = idx[0] + cell_offset[dir][0];
					cidx[1] = idx[1] + cell_offset[dir][1];

					if (cidx[0] < 0 || cidx[1] < 0 || cidx[0] >= size()[0] || cidx[1] >= size()[1]) continue;

					active_local_list.push_back(cidx);
				}
				

			}
			else // Overwrite cell data to background distance if not at a zero crossing
				val = m_nb;

			temp_phi(idx[0], idx[1]) = (m_phi(idx[0], idx[1]) < 0.) ? -val : val;
		}
	});

	//
	// Cull finished cells from active list.
	//

	// Combine parallel lists
	std::vector<Vec2st> active_list;

	for (auto i = parallel_active_list.begin(); i != parallel_active_list.end(); ++i)
	{
		active_list.insert(active_list.end(), i->begin(), i->end());
	}

	parallel_active_list.clear();

	tbb::parallel_sort(active_list.begin(), active_list.end(),
		[](const Vec2st &a, const Vec2st &b) { if (a[0] < b[0]) return true; else if (a[0] == b[0] && a[1] < b[1]) return true; return false; });

	size_t active_count = active_list.size();
	tbb::parallel_for(tbb::blocked_range<size_t>(0, active_count, grain_size), [&](const tbb::blocked_range<size_t> &range)
	{
		std::vector<Vec2st> &local_list = parallel_active_list.local();

		Vec2st oidx(m_phi.size()[0], m_phi.size()[1]);
		for (size_t r = range.begin(); r != range.end(); ++r)
		{
			Vec2st nidx = active_list[r];

			if (nidx != oidx)
			{
				if (marked_cells(nidx[0], nidx[1]) != FINISHED)
				{
					marked_cells(nidx[0], nidx[1]) = VISITED;
					local_list.push_back(nidx);
				}	
			}

			oidx = nidx;
		}
	});

	//
	// Combine list to initialize FIM
	//

	active_list.clear();

	for (auto i = parallel_active_list.begin(); i != parallel_active_list.end(); ++i)
	{
		active_list.insert(active_list.end(), i->begin(), i->end());
	}

	parallel_active_list.clear();

	tbb::parallel_sort(active_list.begin(), active_list.end(),
		[](const Vec2st &a, const Vec2st &b) { if (a[0] < b[0]) return true; else if (a[0] == b[0] && a[1] < b[1]) return true; return false; });

	m_phi = temp_phi;


	// Purely for rendering and debugging. Remove after method is verified.
	std::vector<Vec2st> source_list;

	for (auto i = parallel_source_list.begin(); i != parallel_source_list.end(); ++i)
	{
		source_list.insert(source_list.end(), i->begin(), i->end());
	}

	fast_iterative(marked_cells, active_list);

	m_dirty_surface = false;
	m_gradient_set = false;
	m_curvature_set = false;
	m_mesh_set = false;
}

void LevelSet2D::fast_iterative(UniformGrid<marked> &marked_cells, std::vector<Vec2st> &active_list)
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

		Real Ux = min(U_bx, U_fx);
		Real Uy = min(U_by, U_fy);
		Real U;
		if (fabs(Ux - Uy) >= dx())
			U = min(Ux, Uy) + dx();
		else
			// Quadratic equation from the Eikonal
			//U = (Uh + Uv) / 2 + .5 * sqrt(pow(Uh + Uv, 2) - 2 * (Uh*Uh + Uv*Uv - dx()*dx()));
			U = .5 * (Ux + Uy + sqrt(2. * dx() * dx() - pow(Ux - Uy, 2)));
		return U;
	};

	ScalarGrid<Real> temp_phi = m_phi;

	Real tol = dx() * 1E-5;
	bool active_cells = true;
	size_t grain_size = 5;

	size_t active_count = active_list.size();

	int loopcount = 0;
	int maxcount = 5 * m_nb / dx();
	while (active_count > 0 && loopcount < maxcount)
	{	
		tbb::enumerable_thread_specific<std::vector<Vec2st>> parallel_active_list;

		tbb::parallel_for(tbb::blocked_range<size_t>(0, active_count, grain_size), [&](const tbb::blocked_range<size_t> &range)
		{
			std::vector<Vec2st> &local_list = parallel_active_list.local();
			Vec2st oidx(m_phi.size()[0], m_phi.size()[1]);

			for (size_t r = range.begin(); r != range.end(); ++r)
			{
				Vec2st idx = active_list[r];

				if (oidx != idx)
				{
					assert(marked_cells(idx[0], idx[1]) == VISITED);

					Real eikphi = solveEikonal(Vec2i(idx));

					// If we hit the narrow band, we don't need to make any changes
					if (eikphi > m_nb) continue;

					temp_phi(idx[0], idx[1]) = m_phi(idx[0], idx[1]) < 0 ? -eikphi : eikphi;

					// Check if new phi is converged
					Real pval = m_phi(idx[0], idx[1]);

					Real cval = fabs(eikphi - fabs(m_phi(idx[0], idx[1])));

					// If the cell is converged, load up the neighbours that aren't currently being VISITED
					if (fabs(eikphi - fabs(m_phi(idx[0], idx[1]))) < tol)
					{
						for (size_t dir = 0; dir < 4; ++dir)
						{
							Vec2st cidx;

							cidx[0] = idx[0] + cell_offset[dir][0];
							cidx[1] = idx[1] + cell_offset[dir][1];

							if (cidx[0] < 0 || cidx[1] < 0 || cidx[0] >= size()[0] || cidx[1] >= size()[1]) continue;

							if (marked_cells(cidx[0], cidx[1]) == UNVISITED)
							{
								Real eikphi = solveEikonal(Vec2i(cidx));

								// Check if new phi is less than the current value
								Real pval = fabs(m_phi(cidx[0], cidx[1]));
								if (eikphi < fabs(m_phi(cidx[0], cidx[1])) && fabs(eikphi - fabs(m_phi(cidx[0], cidx[1]))) > tol)
								{
									temp_phi(cidx[0], cidx[1]) = m_phi(cidx[0], cidx[1]) < 0 ? -eikphi : eikphi;

									local_list.push_back(cidx);
								}
							}
						}
					}
					// If the cell hasn't converged, toss it back into the list
					else
						local_list.push_back(idx);
				}

				oidx = idx;
			}

		});

		// Turn off VISITED labels for current list
		tbb::parallel_for(tbb::blocked_range<size_t>(0, active_count, grain_size), [&](const tbb::blocked_range<size_t> &range)
		{
			for (size_t r = range.begin(); r != range.end(); ++r)
			{
				Vec2st idx = active_list[r];
				assert(marked_cells(idx[0], idx[1]) != FINISHED);
				marked_cells(idx[0], idx[1]) = UNVISITED;
			}
		});

		active_list.clear();

		for (auto i = parallel_active_list.begin(); i != parallel_active_list.end(); ++i)
		{
			active_list.insert(active_list.end(), i->begin(), i->end());
		}

		parallel_active_list.clear();

		tbb::parallel_sort(active_list.begin(), active_list.end(), [](const Vec2st &a, const Vec2st &b)
		{ 
			if (a[0] < b[0]) return true;
			else if (a[0] == b[0] && a[1] < b[1]) return true;
			return false;
		});

		
		active_count = active_list.size();

		// Turn on VISITED labels for new list
		tbb::parallel_for(tbb::blocked_range<size_t>(0, active_count, grain_size), [&](const tbb::blocked_range<size_t> &range)
		{
			for (size_t r = range.begin(); r != range.end(); ++r)
			{
				Vec2st idx = active_list[r];
				assert(marked_cells(idx[0], idx[1]) != FINISHED);
				marked_cells(idx[0], idx[1]) = VISITED;
			}
		}); 
		
		std::swap(temp_phi, m_phi);

		++loopcount;
	}
}

void LevelSet2D::reinit(size_t iters)
{	
	if (!m_dirty_surface) return;

	for (size_t i = 0; i < iters; ++i)
	{
		// We want a new gradient to run the interface search with
		build_gradient();

		ScalarGrid<Real> temp_phi = m_phi;

		UniformGrid<marked> marked_cells(size(), UNVISITED);

		// Find zero crossing
		for (size_t x = 0; x < size()[0]; ++x)
			for (size_t y = 0; y < size()[1]; ++y)
			{
				//if (fabs(m_phi(x, y)) >= dx() * 2.) continue;

				for (size_t dir = 0; dir < 4; ++dir)
				{
					int cx = x + cell_offset[dir][0];
					int cy = y + cell_offset[dir][1];

					if (cx < 0 || cy < 0 || cx >= size()[0] || cy >= size()[1]) continue;

					if (m_phi(x, y) * m_phi(cx, cy) <= 0.)
					{
						// Update distance
						Vec2R int_pos = interface_search(idx_to_ws(Vec2R(x, y)), 5);
						Real udf = dist(int_pos, idx_to_ws(Vec2R(x, y)));

						// If the cell has not be updated yet OR the update is lower than a previous
						// update, assign the SDF to the cell
						if (marked_cells(x, y) != FINISHED || abs(temp_phi(x, y)) > udf)
						{
							temp_phi(x, y) = (m_phi(x, y) < 0.) ? -udf : udf;
							marked_cells(x, y) = FINISHED;
						}
					}
				}
				if (marked_cells(x, y) != FINISHED)
				{
					//Real max = std::numeric_limits<Real>::max();
					temp_phi(x, y) = (m_phi(x, y) < 0.) ? -m_nb : m_nb;
				}
			}

		m_phi = temp_phi;
		fast_marching(marked_cells);
	}

	m_dirty_surface = false;
	m_gradient_set = false;
	m_curvature_set = false;
	m_mesh_set = false;
}

void LevelSet2D::init(const Mesh2D& init_mesh, bool resize)
{
	if (resize)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		Vec2R bb_min(std::numeric_limits<Real>::max());
		Vec2R bb_max(std::numeric_limits<Real>::lowest());

		for (size_t v = 0; v < init_mesh.vert_size(); ++v) update_both_minmax(init_mesh.vert(v).point(), bb_min, bb_max);

		// Just for nice whole numbers, let's clamp the bounding box to be an integer
		// offset in index space then bring it back to world space
		Real max_nb = 10.;
		max_nb = std::min(m_nb/dx(), max_nb);

		bb_min = (Vec2R(floor(bb_min / dx())) - Vec2R(max_nb)) * dx();
		bb_max = (Vec2R(ceil(bb_max / dx())) + Vec2R(max_nb)) * dx();

		clear();
		Transform xform(dx(), bb_min);
		// Since we know how big the mesh is, we know how big our grid needs to be (wrt to grid spacing)
		m_phi = ScalarGrid<Real>(xform, Vec2st((bb_max - bb_min) / dx()), m_nb);
	}
	else m_phi.resize(size(), m_nb);

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED. The following pass will do a 
	// BFS to assign the remaining UNVISITED cells with the appropriate distances.
	UniformGrid<marked> marked_cells(size(), UNVISITED);
	UniformGrid<int> parity_cells(size(), 0);

	for (auto e : init_mesh.edges())
	{
		// It's easier to work in our index space and just scale the distance later.
		Vec2R v0 = ws_to_idx(init_mesh.vert(e.vert(0)).point());
		Vec2R v1 = ws_to_idx(init_mesh.vert(e.vert(1)).point());

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		Vec2R vmin, vmax;
		minmax(v0, v1, vmin, vmax);

		Vec2i bb_cmin = ceil(vmin);
		Vec2i bb_fmin = floor(vmin) - Vec2i(1);
		Vec2i bb_fmax = floor(vmax);

		for (int j = bb_cmin[1]; j <= bb_fmax[1]; ++j)
			for (int i = bb_fmax[0]; i >= bb_fmin[0]; --i)
			{
				Vec2R grid_node(i, j);
				INTERSECTION result = exact_edge_intersect(v0, v1, grid_node);

				if (result == NO) continue;
				else if (result == YES)
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
				else if (result == ON)
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

					marked_cells(i, j) = FINISHED;
					m_phi(i, j) = 0.;
				}

				break;
			}
	}

	// Now that all the x-axis edge crossings have been found, we can compile the parity changes
	// and label grid nodes that are at the interface
	for (size_t y = 0; y < size()[1]; ++y)
	{
		int parity = m_inverted ? 1 : 0;

		// We loop x-major because that's how we've set up our edge intersection.
		// This is definitely cache incoherent based on the UniformGrid structure..
		// It's probably not a deal breaker in 2-D but it could suck in 3-D
		for (size_t x = 0; x < size()[0]; ++x) // TODO: double check that i've resized right
		{
			// Update parity before changing sign since the parity values above used the convention
			// of putting the change on the "far" node (i.e. after the mesh-grid intersection).
			parity += parity_cells(x, y);
			parity_cells(x, y) = parity;
			if (parity > 0) m_phi(x, y) = -m_phi(x, y);
		}
	}

	// With the parity assigned, loop over the grid once more and label nodes that have a sign change
	// with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
	for (size_t x = 1; x < (size()[0] - 1); ++x)
		for (size_t y = 1; y < (size()[1] - 1); ++y)
		{
			bool change = false;
			int parity = parity_cells(x, y);
			bool l_inside = (parity > 0);

			for (size_t c = 0; c < 4; ++c)
			{
				Vec2i neighbour(x + cell_offset[c][0], y + cell_offset[c][1]);
				bool n_inside = (parity_cells(neighbour[0], neighbour[1]) > 0);

				// Flag to indicate that this cell needs an exact distance based on the mesh
				if (l_inside != n_inside) marked_cells(x, y) = FINISHED;
			}
		}

	// Loop over all the edges in the mesh. Level set grid cells labelled as VISITED will be
	// updated with the distance to the surface if it happens to be shorter than the current
	// distance to the surface.
	for (auto e : init_mesh.edges())
	{
		// Using the vertices of the edge, we can update distance values for cells
		// within the bounding box of the mesh. It's easier to work in our index space
		// and just scale the distance later.
		Vec2R v0 = ws_to_idx(init_mesh.vert(e.vert(0)).point());
		Vec2R v1 = ws_to_idx(init_mesh.vert(e.vert(1)).point());

		// Build bounding box

		Vec2i bbmin = floor(min_union(v0, v1)) - Vec2i(2);
		bbmin = max_union(bbmin, Vec2i(0));
		Vec2i bbmax = ceil(max_union(v0, v1)) + Vec2i(2);
		Vec2i top(size()[0] - 1, size()[1] - 1);
		bbmax = min_union(bbmax, top);

		// Update distances to the mesh at grid cells within the bounding box
		assert(bbmin[0] >= 0 && bbmin[1] >= 0 && bbmax[0] < size()[0] && bbmax[1] < size()[1]);
		for (int x = bbmin[0]; x <= bbmax[0]; ++x)
			for (int y = bbmin[1]; y <= bbmax[1]; ++y)
			{
				if (marked_cells(x, y) != UNVISITED)
				{
					Vec2R p(x, y);
					Vec2R va = p - v0;
					Vec2R vb = v1 - v0;

					Real c = (dot(va, vb) / dot(vb, vb)); // Find vector coefficient from v0 to v1
					c = ((c < 0.) ? 0. : ((c > 1.) ? 1. : c)); // Truncate scale to be within v0-v1 segment

					Real dist = mag(va - c*vb) * dx();

					// If the parity says the node is inside, set it to be negative
					if (fabs(m_phi(x, y)) > dist)
					{
						m_phi(x, y) = (parity_cells(x, y) > 0) ? -dist : dist;
					}
				}
			}
	}

	fast_marching(marked_cells);
	
	m_dirty_surface = false;
	m_mesh_set = false;
	m_gradient_set = false;
	m_curvature_set = false;
}

void LevelSet2D::fast_marching(UniformGrid<marked>& marked_cells)
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
	typedef std::pair<Vec2i, Real> Node;
	auto cmp = [](const Node& a, const Node& b) -> bool { return fabs(a.second) > fabs(b.second); };
	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> marker_q(cmp);

	for (size_t x = 0; x < size()[0]; ++x)
		for (size_t y = 0; y < size()[1]; ++y)
		{
			if (marked_cells(x, y) == FINISHED)
			{
				for (size_t c = 0; c < 4; ++c)
				{
					Vec2i cidx = Vec2i(x, y) + cell_offset[c];
					//Boundary check
					if (cidx[0] < 0 || cidx[1] < 0 ||
						cidx[0] >= size()[0] || cidx[1] >= size()[1]) continue;

					//store non-marked cells in the queue
					if (marked_cells(cidx[0], cidx[1]) == UNVISITED)
					{
						Real dist = solveEikonal(cidx);
						m_phi(cidx[0], cidx[1]) = (m_phi(cidx[0], cidx[1]) <= 0.) ? -dist : dist;

						assert(dist >= 0);
						Node node(cidx, dist);

						marker_q.push(node);
						marked_cells(cidx[0], cidx[1]) = VISITED;
					}
				}
			}
		}

	while (!marker_q.empty())
	{
		Node curr_node = marker_q.top();
		Vec2i idx = curr_node.first;
		marker_q.pop();

		// Since you can't just update parts of the priority queue,
		// it's possible that a cell has been solidified at a smaller distance
		// and an older insert if floating around.
		if (marked_cells(idx[0], idx[1]) == FINISHED)
		{
			// Make sure that the distance assigned to the cell is smaller than
			// what is floating around
			assert(fabs(m_phi(idx[0], idx[1])) <= fabs(curr_node.second));
			continue;
		}
		assert(marked_cells(idx[0], idx[1]) == VISITED);

		if (fabs(m_phi(idx[0], idx[1])) < m_nb)
		{
			// Debug check that there is indeed a FINISHED cell next to it
			bool marked = false;

			// Loop over the neighbouring cells and load the unvisited cells
			// and update the visited cells
			for (size_t c = 0; c < 4; ++c)
			{
				Vec2i cidx = Vec2i(idx[0], idx[1]) + cell_offset[c];

				int i = cidx[0]; int j = cidx[1];

				//Boundary check
				if (i < 0 || j < 0 || i >= size()[0] || j >= size()[1]) continue;

				if (marked_cells(i, j) == FINISHED)
					marked = true;
				else // If visited, then we'll update it
				{
					Real dist = solveEikonal(cidx);
					assert(dist >= 0);
					if (dist > m_nb) dist = m_nb;

					// If the computed distance is greater than the existing distance, we can skip it
					if (marked_cells(i, j) == VISITED && dist > fabs(m_phi(cidx[0], cidx[1])))
						continue;

					m_phi(cidx[0], cidx[1]) = (m_phi(cidx[0], cidx[1]) < 0.) ? -dist : dist;

					Node node(cidx, dist);

					marker_q.push(node);
					marked_cells(cidx[0], cidx[1]) = VISITED;
				}
			}

			//Check that a marked cell was indeed visited
			assert(marked);
		}
		// Clamp to narrow band
		else
		{
			m_phi(idx[0], idx[1]) = (m_phi(idx[0], idx[1]) < 0.) ? -m_nb : m_nb;
		}

		// Solidify cell now that we've handled all it's neighbours
		marked_cells(idx[0], idx[1]) = FINISHED;
	}

	m_dirty_surface = false;
	m_gradient_set = false;
	m_curvature_set = false;
}

//
// Marching cubes stencil
//

static const Vec2st edge_node_map[4] =
{ Vec2st(0,1), Vec2st(1,2), Vec2st(2,3), Vec2st(3,0) };
//
static const Vec2st node_in_cell[4] =
{ Vec2st(0,0), Vec2st(1,0), Vec2st(1,1), Vec2st(0,1) };


// Extract a mesh representation of the interface. Useful for rendering but not much
// else since there will be duplicate vertices per grid edge in the current implementation.

void LevelSet2D::extract_mesh(Mesh2D& surf) const
{
	std::vector<Vec2d> verts;
	std::vector<Vec2i> edges;
	
	verts.reserve(surf.vert_size());
	edges.reserve(surf.edge_size());

	surf.clear();
	
	// Run marching squares loop
	for (size_t x = 0; x < size()[0] - 1; ++x)
		for (size_t y = 0; y < size()[1] - 1; ++y)
		{
			Vec2st idx = Vec2st(x, y);
			size_t mc_idx = 0;

			for (int i = 0; i < 4; ++i)
			{
				Vec2st coord = Vec2st(x, y) + node_in_cell[i];
				if (m_phi(coord[0], coord[1]) <= 0.)
				{
					mc_idx += (1 << i);
				}
			}

			// Connect edges using the marching squares template
			for (size_t e = 0; e < 4 && mc_template[mc_idx][e] >= 0; e += 2)
			{
				// Find first vertex
				size_t edge = mc_template[mc_idx][e];
				Vec2st n0_idx = idx + node_in_cell[edge_node_map[edge][0]];
				Vec2st n1_idx = idx + node_in_cell[edge_node_map[edge][1]];

				Vec2d v0 = interp_interface(n0_idx, n1_idx);

				// Find second vertex
				edge = mc_template[mc_idx][e + 1];
				n0_idx = idx + node_in_cell[edge_node_map[edge][0]];
				n1_idx = idx + node_in_cell[edge_node_map[edge][1]];

				Vec2d v1 = interp_interface(n0_idx, n1_idx);

				// Store vertices
				Vec2d v0_ws = idx_to_ws(v0);
				Vec2d v1_ws = idx_to_ws(v1);

				verts.push_back(v0_ws);
				verts.push_back(v1_ws);
				 
				edges.push_back(Vec2i(verts.size() - 2, verts.size() - 1));
			}
		}

	surf = Mesh2D(edges, verts);

}

// Extract a mesh representation of the interface using dual contouring

void LevelSet2D::extract_dc_mesh(Mesh2D& surf) // Not const because of building normals
{
	std::vector<Vec2R> verts;
	std::vector<Vec2i> edges;

	verts.reserve(surf.vert_size());
	edges.reserve(surf.edge_size());

	surf.clear();

	// Create grid to store index to dual contouring point. Note that phi is
	// center sampled so the DC grid must be node sampled and cell shorter
	// in each dimension
	UniformGrid<int> dc_points(size() - Vec2st(1), -1);

	// Run dual contouring loop
	for (size_t x = 0; x < dc_points.size()[0]; ++x)
		for (size_t y = 0; y < dc_points.size()[1]; ++y)
		{
			std::vector<Vec2R> qef_points;
			std::vector<Vec2R> qef_norms;

			Vec2st idx(x, y);
			// Find zero crossings
			for (int e = 0; e < 4; ++e)
			{
				Vec2st bidx = idx + node_in_cell[edge_node_map[e][0]];
				Vec2st fidx = idx + node_in_cell[edge_node_map[e][1]];

				// Look for zero crossings
				if ((m_phi(bidx[0], bidx[1]) <= 0.) && (m_phi(fidx[0], fidx[1]) > 0.) ||
					(m_phi(bidx[0], bidx[1]) > 0.) && (m_phi(fidx[0], fidx[1]) <= 0.))
				{
					// Find interface point
					Vec2R v0 = interp_interface(bidx, fidx);
					qef_points.push_back(v0);

					// Find associated surface normal
					Vec2R norm = normal(idx_to_ws(v0));
					qef_norms.push_back(norm);
				}
			}

			if(qef_points.size() > 0)
			{

				Eigen::MatrixXd A(qef_points.size(), 2);
				Eigen::VectorXd b(qef_points.size());
				Eigen::VectorXd xcom = Eigen::VectorXd::Zero(2);
				
				assert(qef_points.size() > 1);
				for (size_t p = 0; p < qef_points.size(); ++p)
				{
					A(p, 0) = qef_norms[p][0]; A(p, 1) = qef_norms[p][1];
					b(p) = dot(qef_norms[p], qef_points[p]);
					xcom[0] += qef_points[p][0]; xcom[1] += qef_points[p][1];	
				}

				xcom /= (Real)(qef_points.size());

				Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
				svd.setThreshold(1E-2);

				Eigen::VectorXd np = xcom + svd.solve(b - A * xcom);
				
				Vec2R cm(xcom[0], xcom[1]);

				Vec2i bbmin = floor(cm);
				Vec2i bbmax = ceil(cm);


				if (np[0] < bbmin[0] || np[1] < bbmin[1] ||
					np[0] > bbmax[0] || np[1] > bbmax[1]) np = xcom;
				/*
				if (np[0] < bbmin[0])
				{
					Real alpha = (bbmin[0] - np[0]) / (xcom[0] - np[0]);
					np += alpha * (xcom - np);
				}
				if (np[1] < bbmin[1])
				{
					Real alpha = (bbmin[1] - np[1]) / (xcom[1] - np[1]);
					np += alpha * (xcom - np);
				}
				if (np[0] > bbmax[0])
				{
					Real alpha = (bbmax[0] - np[0]) / (xcom[0] - np[0]);
					np += alpha * (xcom - np);
				}
				if (np[1] > bbmax[1])
				{
					Real alpha = (bbmax[1] - np[1]) / (xcom[1] - np[1]);
					np += alpha * (xcom - np);
				}
				*/

				verts.push_back(idx_to_ws(Vec2R(np[0], np[1])));
				dc_points(x, y) = verts.size() - 1;
			}
		}

	// Connected DC points across faces
	for (size_t x = 1; x < size()[0] - 1; ++x)
		for (size_t y = 0; y < size()[1] - 1; ++y)
		{
			Vec2i idx(x, y);
			Vec2i bidx = idx;
			Vec2i fidx = idx + Vec2i(0,1);

			// Look for zero crossings
			if ((m_phi(bidx[0], bidx[1]) <= 0.) && (m_phi(fidx[0], fidx[1]) > 0.) ||
				(m_phi(bidx[0], bidx[1]) > 0.) && (m_phi(fidx[0], fidx[1]) <= 0.))
			{
				// Confirm that both adjacent cells have indexed vertices
				Vec2i bcidx = idx + face_to_cell[0][0];
				Vec2i fcidx = idx + face_to_cell[0][1];

				assert(dc_points(bcidx[0], bcidx[1]) >= 0 && dc_points(fcidx[0], fcidx[1]) >= 0);

				if (m_phi(bidx[0], bidx[1]) <= 0.) edges.push_back(Vec2i(dc_points(bcidx[0], bcidx[1]), dc_points(fcidx[0], fcidx[1])));
				else edges.push_back(Vec2i(dc_points(fcidx[0], fcidx[1]), dc_points(bcidx[0], bcidx[1])));
			}
		}

	for (size_t x = 0; x < size()[0] - 1; ++x)
		for (size_t y = 1; y < size()[1] - 1; ++y)
		{
			Vec2i idx(x, y);
			Vec2i bidx = idx;
			Vec2i fidx = idx + Vec2i(1, 0);

			// Look for zero crossings
			if ((m_phi(bidx[0], bidx[1]) <= 0.) && (m_phi(fidx[0], fidx[1]) > 0.) ||
				(m_phi(bidx[0], bidx[1]) > 0.) && (m_phi(fidx[0], fidx[1]) <= 0.))
			{
				// Confirm that both adjacent cells have indexed vertices
				Vec2i bcidx = idx + face_to_cell[1][0];
				Vec2i fcidx = idx + face_to_cell[1][1];

				assert(dc_points(bcidx[0], bcidx[1]) >= 0 && dc_points(fcidx[0], fcidx[1]) >= 0);

				if (m_phi(bidx[0], bidx[1]) <= 0.) edges.push_back(Vec2i(dc_points(fcidx[0], fcidx[1]), dc_points(bcidx[0], bcidx[1])));
				else edges.push_back(Vec2i(dc_points(bcidx[0], bcidx[1]), dc_points(fcidx[0], fcidx[1])));
			}
		}

	surf = Mesh2D(edges, verts);
}


Vec2R LevelSet2D::interp_interface(const Vec2st& i0, const Vec2st& i1) const
{
	assert(m_phi(i0[0], i0[1]) * m_phi(i1[0], i1[1]) <= 0.0);
	//Find weight to zero isosurface
	double s = m_phi(i0[0], i0[1]) / (m_phi(i0[0], i0[1]) - m_phi(i1[0], i1[1]));

	if (s < 0.0) s = 0.0;
	else if (s > 1.0) s = 1.0;

	Vec2d dx = Vec2d(i1) - Vec2d(i0);
	return Vec2d(i0[0], i0[1]) + s*dx;
}

void LevelSet2D::gradient_field(VectorGrid<Real>& grad)
{
	grad = VectorGrid<Real>(xform(), size(), 0, VectorGridSettings::STAGGERED);
	
	Real invdx = 1. / dx();
	for (int x = 1; x < grad.size(0)[0] - 1; ++x)
		for (int y = 0; y < grad.size(0)[1]; ++y)
		{
			Vec2i cb = Vec2i(x - 1, y);
			Vec2i cf = Vec2i(x, y);

			if (fabs(m_phi(cb[0], cb[1])) >= m_nb || fabs(m_phi(cf[0], cf[1])) >= m_nb) continue;

			grad(x, y, 0) = (m_phi(cf[0], cf[1]) - m_phi(cb[0], cb[1])) * invdx;
		}

	for (int x = 0; x < grad.size(1)[0]; ++x)
		for (int y = 1; y < grad.size(1)[1] - 1; ++y)
		{
			Vec2i cb = Vec2i(x, y - 1);
			Vec2i cf = Vec2i(x, y);
			
			if (fabs(m_phi(cb[0], cb[1])) >= m_nb || fabs(m_phi(cf[0], cf[1])) >= m_nb) continue;

			grad(x, y, 1) = (m_phi(cf[0], cf[1]) - m_phi(cb[0], cb[1])) * invdx;
		}
}

void LevelSet2D::build_gradient()
{
	if (m_gradient_set) return;

	gradient_field(m_gradient);
	m_gradient_set = true;
}

bool LevelSet2D::curvature_field(ScalarGrid<Real>& curv)
{
	if (!m_gradient_set) return false;
	ScalarGrid<Real> secondorder(xform(), size(), 0, ScalarSampleType::NODE);

	Real invdx = 1. / dx();
	for (int x = 1; x < secondorder.size()[0] - 1; ++x)
		for (int y = 1; y < secondorder.size()[1] - 1; ++y)
		{
			secondorder(x, y) = (m_gradient(x, y, 1) - m_gradient(x - 1, y, 1)) * invdx;
		}

	// With the second order system built, we can move forward and compute the full curvature system
	curv = ScalarGrid<Real>(xform(), size(), 0);
	for(int x = 1; x < curv.size()[0] - 1; ++x)
		for (int y = 1; y < curv.size()[1] - 1; ++y)
		{
			if (fabs(m_phi(x, y)) >= m_nb) continue;

			Real phi_xx = (m_phi(x - 1, y) - 2 * m_phi(x, y) + m_phi(x + 1, y)) * invdx * invdx;
			Real phi_yy = (m_phi(x, y - 1) - 2 * m_phi(x, y) + m_phi(x, y + 1)) * invdx * invdx;

			Vec2R wpos = idx_to_ws(Vec2R(x, y));
			Real phi_xy = secondorder.interp(wpos);

			Real phi_x = m_gradient.interp(wpos, 0);
			Real phi_y = m_gradient.interp(wpos, 1);

			Real phi_x2 = phi_x * phi_x;
			Real phi_y2 = phi_y * phi_y;

			if (phi_x == 0. && phi_y == 0.) curv(x, y) = 0.;
			else
			{
				Real local_curv = phi_x2 * phi_yy - 2. * phi_x * phi_y * phi_xy + phi_y2 * phi_xx / sqrt(pow(phi_x2 + phi_y2, 3.));
				curv(x, y) = clamp(local_curv, -invdx, invdx);
			}
		}

	return true;
}

void LevelSet2D::build_curvature()
{
	if (m_curvature_set) return;

	if (!m_gradient_set) build_gradient();

	m_curvature_set = curvature_field(m_curvature);
	assert(m_curvature_set);
}

void LevelSet2D::surface_union(const LevelSet2D& input_volume)
{
	for (size_t x = 0; x < size()[0]; ++x)
		for (size_t y = 0; y < size()[1]; ++y)
		{
			m_phi(x, y) = std::min(m_phi(x, y), input_volume.interp(idx_to_ws(Vec2R(x, y))));
		}

	reinit();
}
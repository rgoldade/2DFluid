#include <vector>
#include <array>
#include <queue>

#include "PressureProjection.h"
#include "Solver.h"

static int UNSOLVED = -1;
static int SOLVED = 0;

void PressureProjection::draw_pressure(Renderer& renderer) const
{
	m_pressure.draw_supersampled_values(renderer, .25, 3, 2);
}

void PressureProjection::draw_divergence(Renderer& renderer) const
{
	ScalarGrid<Real> div(m_surface.xform(), m_surface.size(), 0);

	for (int x = 0; x < div.size()[0]; ++x)
		for (int y = 0; y < div.size()[1]; ++y)
		{
			if (m_surface(x, y) < 0)
			{
				for (int dir = 0; dir < 4; ++dir)
				{
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];
					double sign = (dir % 2 == 0) ? -1 : 1;
					div(x, y) = sign * m_vel(fx, fy, dir / 2) / m_surface.dx();
				}
			}
			else div(x, y) = 0;
		}
	Real max = div.maxval();
	Real min = div.minval();
	div.draw_supersampled_values(renderer, .25, 3, 2);
}

// Store the i, j coordinates and staggered face axis
typedef Vec3i BubbleFaceIndex;
// Store just the i, j coordinates
typedef Vec2i BubbleCellIndex;

// Perform a BFS through air pockets and build a list of air/liquid faces for each bubble.
// These faces will be uses to form the fluxes of the air pressure in the poisson solve

void build_bubble_components(const LevelSet2D& surface,
								const VectorGrid<Real>& fluid_weights,
								std::vector<std::vector<BubbleFaceIndex>>& bubble_surface_list,
								std::vector<std::vector<BubbleCellIndex>>& bubble_collision_list,
								Renderer& renderer, bool apply_collisions = false)
{
	UniformGrid<marked> cell_labels(surface.size(), UNVISITED);
	VectorGrid<marked> face_labels(surface.xform(), surface.size(), UNVISITED, VectorGridSettings::STAGGERED);

	for (int x = 0; x < surface.size()[0]; ++x)
		for (int y = 0; y < surface.size()[1]; ++y)
		{
			// Find a cell that it outside of the liquid and not yet processed by the BFS
			if (surface(x, y) >= 0 && cell_labels(x, y) == UNVISITED)
			{
				// Make sure that one of the cell faces contains a non-zero fluid weight,
				// suggesting that one of the faces is not completely inside a collision volume.
				// This might be totally redundant given the collision check
				bool nonzero = false;
				for (int dir = 0; dir < 4; ++dir)
				{
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];
					if (fluid_weights(fx, fy, dir / 2) > 0.)
						nonzero = true;
				}
				if (!nonzero) continue;

				std::queue<BubbleCellIndex> air_cell_queue;
				air_cell_queue.push(BubbleCellIndex{ x, y });
				cell_labels(x, y) = VISITED;

				std::vector<BubbleFaceIndex> current_bubble_surface;
				std::vector<BubbleCellIndex> current_bubble_collision;

				while (air_cell_queue.size() > 0)
				{
					BubbleCellIndex current_cell = air_cell_queue.front();
					air_cell_queue.pop();

					assert(cell_labels(current_cell[0], current_cell[1]) == VISITED);

					for (int dir = 0; dir < 4; ++dir)
					{
						int fx = current_cell[0] + cell_to_face[dir][0];
						int fy = current_cell[1] + cell_to_face[dir][1];
						
						int axis = dir / 2;
						// If the face has yet to be processed, let's check if the oppose side is in
						// the fluid surface. Since this current cell should not have been put in the 
						// queue if it was inside the surface, we must be at a bubble boundary.
						if (face_labels(fx, fy, axis) != FINISHED)
						{
							assert(face_labels(fx, fy, axis) != VISITED);

							int cx = current_cell[0] + cell_offset[dir][0];
							int cy = current_cell[1] + cell_offset[dir][1];

							if (cx < 0 || cy < 0 || cx >= surface.size()[0]
								|| cy >= surface.size()[1]) continue;

							if (surface(cx, cy) < 0. && fluid_weights(fx, fy, axis) > 0.)
								current_bubble_surface.push_back(BubbleFaceIndex(fx, fy, axis));

							face_labels(fx, fy, axis) = FINISHED;
						}
					}

					if (apply_collisions)
					{
						bool include_cell = false;
						for (int dir = 0; dir < 4; ++dir)
						{
							int fx = current_cell[0] + cell_to_face[dir][0];
							int fy = current_cell[1] + cell_to_face[dir][1];

							int axis = dir / 2;

							if (fluid_weights(fx, fy, axis) < 1.) include_cell = true;

						}
						if (include_cell)
							current_bubble_collision.push_back(BubbleCellIndex(current_cell[0], current_cell[1]));
					}

					// After processing the faces of the cell, we need to add neighbouring cells into the queue
					// and flood outwards
					for (int dir = 0; dir < 4; ++dir)
					{
						int cx = current_cell[0] + cell_offset[dir][0];
						int cy = current_cell[1] + cell_offset[dir][1];

						if (cx < 0 || cy < 0 || cx >= surface.size()[0]
							|| cy >= surface.size()[1]) continue;

						if (cell_labels(cx, cy) == UNVISITED)
						{
							int fx = current_cell[0] + cell_to_face[dir][0];
							int fy = current_cell[1] + cell_to_face[dir][1];

							if (surface(cx, cy) >= 0. && fluid_weights(fx, fy, dir / 2) > 0.)
							{
								air_cell_queue.push(BubbleCellIndex(cx, cy));
								cell_labels(cx, cy) = VISITED;
							}
						}
					}

					cell_labels(current_cell[0], current_cell[1]) = FINISHED;
				}

				if (current_bubble_surface.size() > 0)
				{
					bubble_surface_list.push_back(current_bubble_surface);
				//	std::cout << "Current bubble size: " << current_bubble_surface.size() << std::endl;
				//	std::cout << "Bubble list size: " << bubble_surface_list.size() << std::endl;

					bubble_collision_list.push_back(current_bubble_collision);
				//	std::cout << "Current collision bubble size: " << current_bubble_collision.size() << std::endl;
				//	std::cout << "Collision bubble list size: " << bubble_collision_list.size() << std::endl;
				}
			}
		}

		//Vec3f colours[] = { Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1),
		//						Vec3f(1,1,0), Vec3f(0,1,1), Vec3f(1,0,1), };

		//// Temporary render of the bubble surface list
		//for (size_t b = 0; b < bubble_surface_list.size(); ++b)
		//{
		//	std::vector<Vec2R> start_points;
		//	std::vector<Vec2R> end_points;

		//	std::vector<BubbleFaceIndex> test_vector = bubble_surface_list[b];
		//	
		//	if (test_vector.size() > 1)
		//	{
		//		for (size_t bl = 0; bl < test_vector.size(); ++bl)
		//		{

		//			BubbleFaceIndex cell = test_vector[bl];
		//			Vec2R pos = fluid_weights.idx_to_ws(Vec2R(cell[0], cell[1]), cell[2]);

		//			if (cell[2] == 0)
		//			{
		//				start_points.push_back(Vec2R(pos[0], pos[1] - fluid_weights.dx() * .5));
		//				end_points.push_back(Vec2R(pos[0], pos[1] + fluid_weights.dx() * .5));
		//			}
		//			else
		//			{
		//				start_points.push_back(Vec2R(pos[0] - fluid_weights.dx() * .5, pos[1]));
		//				end_points.push_back(Vec2R(pos[0] + fluid_weights.dx() * .5, pos[1]));
		//			}
		//		}
		//	}
		//	renderer.add_lines(start_points, end_points, colours[b % 6]);
		//}

		//// Temporary render of the bubble collision list
		//for (size_t b = 0; b < bubble_collision_list.size(); ++b)
		//{
		//	std::vector<Vec2R> start_points;
		//	std::vector<Vec2R> end_points;

		//	std::vector<BubbleCellIndex> test_vector = bubble_collision_list[b];

		//	if (test_vector.size() > 1)
		//	{
		//		for (size_t bl = 0; bl < test_vector.size(); ++bl)
		//		{

		//			BubbleCellIndex cell = test_vector[bl];
		//			Vec2R wpos = surface.idx_to_ws(Vec2R(cell[0], cell[1]));
	

		//			start_points.push_back(Vec2R(wpos[0] - surface.dx() * .5, wpos[1] - surface.dx() * .5));
		//			end_points.push_back(Vec2R(wpos[0] - surface.dx() * .5, wpos[1] + surface.dx() * .5));
		//			
		//			start_points.push_back(Vec2R(wpos[0] - surface.dx() * .5, wpos[1] - surface.dx() * .5));
		//			end_points.push_back(Vec2R(wpos[0] + surface.dx() * .5, wpos[1] - surface.dx() * .5));

		//			start_points.push_back(Vec2R(wpos[0] + surface.dx() * .5, wpos[1] - surface.dx() * .5));
		//			end_points.push_back(Vec2R(wpos[0] + surface.dx() * .5, wpos[1] + surface.dx() * .5));

		//			start_points.push_back(Vec2R(wpos[0] - surface.dx() * .5, wpos[1] + surface.dx() * .5));
		//			end_points.push_back(Vec2R(wpos[0] + surface.dx() * .5, wpos[1] + surface.dx() * .5));
		//		}
		//	}
		//	renderer.add_lines(start_points, end_points, colours[b % 6]);
		//}
}

void PressureProjection::project(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights, Renderer& renderer)
{
	assert(liquid_weights.size(0) == fluid_weights.size(0) &&
			liquid_weights.size(1) == fluid_weights.size(1) &&
			liquid_weights.size(0) == m_vel.size(0) &&
			liquid_weights.size(1) == m_vel.size(1));

	UniformGrid<int> solvable_cells(m_surface.size(), UNSOLVED);

	// This loop is dumb in serial but it's here as a placeholder for parallel later
	for (int x = 0; x < solvable_cells.size()[0]; ++x)
		for (int y = 0; y < solvable_cells.size()[1]; ++y)
		{
			if (m_surface(x, y) < 0.)
			{
				for (int dir = 0; dir < 4; ++dir)
				{
					// A solvable pressure sample point should be inside in the surface
					// and with at least one non-zero cut-cell length (i.e. not extrapolated
					// too far into the collision volume)
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];

					int axis = dir / 2;
					if (fluid_weights(fx, fy, axis) > 0)
						solvable_cells(x, y) = SOLVED;
				}
			}
		}

	int solvecount = 0;
	for (size_t x = 0; x < solvable_cells.size()[0]; ++x)
		for (size_t y = 0; y < solvable_cells.size()[1]; ++y)
		{
			if (solvable_cells(x, y) >= 0)
				solvable_cells(x, y) = solvecount++;
		}

	// Build a list of air bubbles by storing a list of air/liquid faces for each bubble
	std::vector<std::vector<BubbleFaceIndex>> bubble_surface_list;
	std::vector<std::vector<BubbleCellIndex>> bubble_collision_list;

	int bubblecount = 0;
	int bubblenonzeros = 0;
	if (m_enforce_bubbles)
	{
		build_bubble_components(m_surface, fluid_weights, bubble_surface_list, bubble_collision_list, renderer, m_colvel_set);

		for (auto& b : bubble_surface_list) bubblenonzeros += b.size();

		bubblecount = bubble_surface_list.size() - 1;
	}

	Solver solver(solvecount + bubblecount, solvecount * 7 + bubblenonzeros);

	// Build linear system
	double dx = m_surface.dx();
	for (int x = 0; x < solvable_cells.size()[0]; ++x)
		for (int y = 0; y < solvable_cells.size()[1]; ++y)
		{
			int idx = solvable_cells(x, y);
			if (idx >= 0)
			{
				// Build RHS divergence
				for (int dir = 0; dir < 4; ++dir)
				{
					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];
					double sign = (dir % 2 == 0) ? 1 : -1;
					double div = sign * m_vel(fx, fy, dir / 2) * fluid_weights(fx, fy, dir / 2) / dx;
					
					solver.add_rhs(idx, div);
				}

				// Build collision vel divergence
				if (m_colvel_set)
				{
					for (int dir = 0; dir < 4; ++dir)
					{
						int fx = x + cell_to_face[dir][0];
						int fy = y + cell_to_face[dir][1];
						double sign = (dir % 2 == 0) ? 1. : -1.;
						double div = sign * m_collision_vel(fx, fy, dir / 2) * (1.0 - fluid_weights(fx, fy, dir / 2)) / dx;

						solver.add_rhs(idx, div);
					}
				}

				if(m_divergence)
					solver.add_rhs(idx, (*m_divergence)(x, y));

				// Build row
				double middle_coeff = 0;
				for (int dir = 0; dir < 4; ++dir)
				{
					int cx = x + cell_offset[dir][0];
					int cy = y + cell_offset[dir][1];

					// Bounds check. If out-of-bounds, treat like a grid-aligned collision.
					if (cx < 0 || cy < 0 || cx >= m_surface.size()[0] || cy >= m_surface.size()[1]) continue;

					int fx = x + cell_to_face[dir][0];
					int fy = y + cell_to_face[dir][1];

					double coeff = fluid_weights(fx, fy, dir / 2 ) * m_dt / dx / dx;
					
					// If neighbouring cell is solvable, it should have an entry in the system
					int cidx = solvable_cells(cx, cy);
					if (cidx >= 0)
					{
						solver.add_element(idx, cidx, -coeff);
						middle_coeff += coeff;
					}
					else if (fluid_weights(fx, fy, dir / 2) > 0.) // Preventing any complications deep inside collision
					{
						double theta = liquid_weights(fx, fy, dir / 2);
						if (theta < 0.01) theta = 0.01;
						middle_coeff += coeff / theta;

						// Add surface pressure to RHS
						if (m_sp_set)
						{
							Real spress = lerp(m_surface_pressure(cx, cy), m_surface_pressure(x, y), theta);
							solver.add_rhs(idx, spress * m_sp_scale * coeff / theta);
						}

					}					
				}

				if (middle_coeff > 0.)	solver.add_element(idx, idx, middle_coeff);
				else solver.add_element(idx, idx, 1.);
			}
		}

	// Sort bubble list so the largest bubble is last. We only need to enfore n-1 bubbles
	// since the final bubble is enforced implicitly by the divergence free constraint
	// in the liquid component

	// Add bubbles to the linear system
	if (m_enforce_bubbles)
	{
		std::vector<std::pair<int, int>> bubble_order(bubble_surface_list.size());
		for (size_t b = 0; b < bubble_order.size(); ++b)
			bubble_order[b] = std::pair<int, int>(b, bubble_surface_list[b].size());

		std::sort(bubble_order.begin(), bubble_order.end(),
			[](const std::pair<int, int> &v0, const std::pair<int, int> &v1) -> bool
		{ return v0.second < v1.second; });

		for (size_t b = 0; b < bubble_order.size() - 1; ++b)
		{
			size_t bubble = bubble_order[b].first;

			double middle_coeff = 0.;
			double bubble_divergence = 0;
			int bidx = solvecount + b;

			for (size_t bf = 0; bf < bubble_surface_list[bubble].size(); ++bf)
			{
				BubbleFaceIndex bubbleface = bubble_surface_list[bubble][bf];
				int fx = bubbleface[0];
				int fy = bubbleface[1];
				int axis = bubbleface[2];

				double theta = liquid_weights(fx, fy, axis);
				if (theta < 0.01) theta = 0.01;
				double coeff = fluid_weights(fx, fy, axis) * m_dt / dx / dx / theta;

				int cidx;
				double sign;
				// Check which cell is inside the surface
				if (m_surface(fx + face_to_cell[axis][0][0], fy + face_to_cell[axis][0][1]) < 0.)
				{
					assert(m_surface(fx + face_to_cell[axis][1][0], fy + face_to_cell[axis][1][1]) >= 0);
					cidx = solvable_cells(fx + face_to_cell[axis][0][0], fy + face_to_cell[axis][0][1]);

					// Set solveable face to use the bubble index. This allows us to lookup the right pressure sample
					// value for updating the velocity
					solvable_cells(fx + face_to_cell[axis][1][0], fy + face_to_cell[axis][1][1]) = bidx;

					sign = 1;
				}
				else if (m_surface(fx + face_to_cell[axis][1][0], fy + face_to_cell[axis][1][1]) < 0)
				{
					cidx = solvable_cells(fx + face_to_cell[axis][1][0], fy + face_to_cell[axis][1][1]);

					// Set solveable face to use the bubble index. This allows up to lookup the right pressure sample
					// value for updating the velocity
					solvable_cells(fx + face_to_cell[axis][0][0], fy + face_to_cell[axis][0][1]) = bidx;

					sign = -1;
				}
				else assert(false);

				// Update both sides of the symmetric matrix
				solver.add_element(bidx, cidx, -coeff);
				solver.add_element(cidx, bidx, -coeff);

				middle_coeff += coeff;

				// Add velocity terms to the divergence calculator
				double div = sign * m_vel(fx, fy, axis) * fluid_weights(fx, fy, axis) / dx;
				solver.add_rhs(bidx, div);
			}

			solver.add_element(bidx, bidx, middle_coeff);

			// If there are moving solids in the scene, we might need to enforce their movement onto the bubbles
			if (m_colvel_set)
			{
				Real div = 0;
				for (size_t bf = 0; bf < bubble_collision_list[bubble].size(); ++bf)
				{
					BubbleCellIndex bubblecell = bubble_collision_list[bubble][bf];
					int x = bubblecell[0];
					int y = bubblecell[1];

					for (int dir = 0; dir < 4; ++dir)
					{
						int fx = x + cell_to_face[dir][0];
						int fy = y + cell_to_face[dir][1];
						double sign = (dir % 2 == 0) ? 1. : -1.;
						div += sign * m_collision_vel(fx, fy, dir / 2) * (1.0 - fluid_weights(fx, fy, dir / 2)) / dx;
					}
				}
				solver.add_rhs(bidx, div);
			}

		}
	}



	solver.solve();

	// Load solution into pressure grid
	for (int x = 0; x < solvable_cells.size()[0]; ++x)
		for (int y = 0; y < solvable_cells.size()[1]; ++y)
		{
			int idx = solvable_cells(x, y);
			if (idx >= 0)
			{
				double p = solver.sol(idx);
				m_pressure(x, y) = p;
			}
		}
}

void PressureProjection::apply_solution(VectorGrid<Real>& vel, const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights)
{
	assert(vel.size(0) == m_vel.size(0) &&
			vel.size(1) == m_vel.size(1) &&
			liquid_weights.size(0) == m_vel.size(0) &&
			liquid_weights.size(1) == m_vel.size(1) &&
			fluid_weights.size(0) == m_vel.size(0) &&
			fluid_weights.size(1) == m_vel.size(1));
			
	for (int y = 0; y < vel.size(0)[1]; ++y)
	{
		vel(0, y, 0) = 0.;
		vel(vel.size(0)[0] - 1, y, 0) = 0.;
	}

	for (int x = 1; x < vel.size(0)[0] - 1; ++x)
		for (int y = 0; y < vel.size(0)[1]; ++y)
		{
			Vec2i cb = Vec2i(x - 1, y);
			Vec2i cf = Vec2i(x, y);

			if (fluid_weights(x, y, 0) > 0.)
			{
				Real theta = liquid_weights(x, y, 0);
				if (theta > 0.)
				{
					if (theta < 0.01) theta = 0.01;

					Real pb = 0;
					if (m_surface(cb[0], cb[1]) < 0.)
						pb = m_pressure(cb[0], cb[1]);
					else
					{
						if (m_enforce_bubbles)
						{
							assert(m_surface(cf[0], cf[1]) < 0.);
							pb = m_pressure(cb[0], cb[1]);
						}
						if (m_sp_set)
							pb += lerp(m_surface_pressure(cb[0], cb[1]), m_surface_pressure(cf[0], cf[1]), theta) * m_sp_scale;
					}

					Real pf = 0;
					if (m_surface(cf[0], cf[1]) < 0.)
						pf = m_pressure(cf[0], cf[1]);
					else
					{
						if (m_enforce_bubbles)
						{
							assert(m_surface(cb[0], cb[1]) < 0.);
							pf = m_pressure(cf[0], cf[1]);
						}
						if (m_sp_set)
							pf += lerp(m_surface_pressure(cf[0], cf[1]), m_surface_pressure(cb[0], cb[1]), theta) * m_sp_scale;
					}
					
					vel(x, y, 0) = m_vel(x, y, 0) - m_dt * (pf - pb) / m_surface.dx() / theta;
					m_valid(x, y, 0) = 1.;
				}
				else vel(x, y, 0) = 0.;
			}
			else vel(x, y, 0) = 0.;
		}

	for (int x = 0; x < vel.size(1)[0]; ++x)
	{
		vel(x, 0, 1) = 0.;
		vel(x, vel.size(1)[1] - 1, 1) = 0.;
	}

	for (int x = 0; x < vel.size(1)[0]; ++x)
		for (int y = 1; y < vel.size(1)[1] - 1; ++y)
		{
			Vec2i cb = Vec2i(x, y - 1);
			Vec2i cf = Vec2i(x, y);

			if (fluid_weights(x, y, 1) > 0.)
			{
				Real theta = liquid_weights(x, y, 1);
				if (theta > 0.)
				{
					if (theta < 0.01) theta = 0.01;

					Real pb = 0;
					if (m_surface(cb[0], cb[1]) < 0.)
						pb = m_pressure(cb[0], cb[1]);
					else
					{
						if (m_enforce_bubbles)
						{
							assert(m_surface(cf[0], cf[1]) < 0.);
							pb = m_pressure(cb[0], cb[1]);
						}
						if (m_sp_set)
							pb += lerp(m_surface_pressure(cb[0], cb[1]), m_surface_pressure(cf[0], cf[1]), theta) * m_sp_scale;
					}

					Real pf = 0;
					if (m_surface(cf[0], cf[1]) < 0.)
						pf = m_pressure(cf[0], cf[1]);
					else
					{
						if (m_enforce_bubbles)
						{
							assert(m_surface(cb[0], cb[1]) < 0.);
							pf = m_pressure(cf[0], cf[1]);
						}
						if (m_sp_set)
							pf += lerp(m_surface_pressure(cf[0], cf[1]), m_surface_pressure(cb[0], cb[1]), theta) * m_sp_scale;
					}

					vel(x, y, 1) = m_vel(x, y, 1) - m_dt * (pf - pb) / m_surface.dx() / theta;
					m_valid(x, y, 1) = 1.;
				}
				else vel(x, y, 1) = 0.;
			}
			else vel(x, y, 1) = 0.;
		}
}


void PressureProjection::apply_valid(VectorGrid<Real> &valid)
{
	valid = m_valid;
}



size_t ni;
size_t nj;
int u_index(int i, int j) {
	return i + j*(ni + 1);
}
int v_index(int i, int j) {
	return i + j*ni + nj*(ni + 1);
}
int c_index(int i, int j) {
	return i + j*ni + ni*(nj + 1) + (ni + 1)*nj;
}
int b_index(int bubble_no) {
	return bubble_no + ni*nj + ni*(nj + 1) + (ni + 1)*nj;
}

void PressureProjection::batty_pressure_solve(const VectorGrid<Real>& liquid_weights, const VectorGrid<Real>& fluid_weights, Renderer& renderer, VectorGrid<Real>& valid, VectorGrid<Real>& vel)
{
	const LevelSet2D& liquid_phi = m_surface;
	const ScalarGrid<Real>& u_weights = fluid_weights.grid(0);
	const ScalarGrid<Real>& v_weights = fluid_weights.grid(1);

	// Find bubbles
	UniformGrid<int> cell_visited(m_surface.size(), 0);
	std::vector<std::vector<Vec2ui>> bubbles;
	
	//find all the individual bubbles
	for (int start_i = 0; start_i < cell_visited.size()[0]; ++start_i)
	{
		for (int start_j = 0; start_j < cell_visited.size()[1]; ++start_j)
		{
			if (cell_visited(start_i, start_j) == 0 && liquid_phi(start_i, start_j) >= 0 && (u_weights(start_i, start_j)  > 0 ||
				u_weights(start_i + 1, start_j)  > 0 ||
				v_weights(start_i, start_j) > 0 ||
				v_weights(start_i, start_j + 1) > 0))
			{
				std::cout << "Starting bubble search\n";
				std::vector<Vec2ui> cur_bubble;
				std::queue<Vec2ui> cells_to_visit;
				Vec2ui start_cell(start_i, start_j);
				cells_to_visit.push(start_cell);
				while (cells_to_visit.size() > 0)
				{
					Vec2ui cur_cell = cells_to_visit.front();
					cells_to_visit.pop();

					if (cell_visited(cur_cell[0], cur_cell[1]) == 1)
						continue;
					else
						cell_visited(cur_cell[0], cur_cell[1]) = 1;

					//add this cell to the active bubble
					cur_bubble.push_back(cur_cell);

					int i = cur_cell[0];
					int j = cur_cell[1];
					//now add the four neighbour cells
					if (liquid_phi(i + 1, j) >= 0 && u_weights(i + 1, j) > 0) {
						cells_to_visit.push(Vec2ui(i + 1, j));
					}
					if (liquid_phi(i - 1, j) >= 0 && u_weights(i, j) > 0) {
						cells_to_visit.push(Vec2ui(i - 1, j));
					}
					if (liquid_phi(i, j + 1) >= 0 && v_weights(i, j + 1) > 0) {
						cells_to_visit.push(Vec2ui(i, j + 1));
					}
					if (liquid_phi(i, j - 1) >= 0 && v_weights(i, j) > 0) {
						cells_to_visit.push(Vec2ui(i, j - 1));
					}
				}
				
				bubbles.push_back(cur_bubble);
			}
		}
	}
	std::cout << "Bubble count: " << bubbles.size() << std::endl;

	std::cout << "Set up matrix\n";
	ni = m_surface.size()[0];
	nj = m_surface.size()[1];

	const ScalarGrid<Real>& u = m_vel.grid(0);
	const ScalarGrid<Real>& v = m_vel.grid(1);

	int system_size = ni*nj + ni*(nj + 1) + (ni + 1)*nj + (int)bubbles.size() - 1;
	Eigen::SparseMatrix<double> A_mat(system_size, system_size);
	Eigen::VectorXd x_vec(system_size), b_vec(system_size);

	std::vector<Eigen::Triplet<double>> triplets;

	Real dx = m_surface.dx();

	auto is_active_face = [&](bool u_face, int i, int j) -> bool
	{
		return  u_face &&  i != 0 && i != ni && j != -1 && j != nj &&
				(liquid_phi(i - 1, j) <= 0 || liquid_phi(i, j) <= 0) && (u_weights(i, j) > 0) ||
				!u_face &&  i != -1 && i != ni && j != 0 && j != nj &&
				(liquid_phi(i, j - 1) <= 0 || liquid_phi(i, j) <= 0) && (v_weights(i, j) > 0);
	};

	std::cout << "U-part\n";
	//u-updates
	for (int j = 0; j < nj; ++j)
		for (int i = 0; i < ni + 1; ++i)
		{
			int index = u_index(i, j);
			if (is_active_face(true, i, j))
			{
				double gf_weight = liquid_weights(i, j, 0);// (fraction_inside(liquid_phi(i - 1, j), liquid_phi(i, j));
				if (gf_weight < 0.01) gf_weight = 0.01;
				triplets.push_back(Eigen::Triplet<double>(index, index, u_weights(i, j)*gf_weight));
				b_vec[index] = u_weights(i, j)*gf_weight * u(i, j);

				if (liquid_phi(i - 1, j) < 0)
					triplets.push_back(Eigen::Triplet<double>(index, c_index(i - 1, j), -u_weights(i, j) / dx));
				else if(m_sp_set)
					b_vec[index] += lerp(m_surface_pressure(i- 1, j), m_surface_pressure(i, j), gf_weight) * m_sp_scale;
					
				if (liquid_phi(i, j) < 0)
					triplets.push_back(Eigen::Triplet<double>(index, c_index(i, j), +u_weights(i, j) / dx));
				else if (m_sp_set)
					b_vec[index] -= lerp(m_surface_pressure(i - 1, j), m_surface_pressure(i, j), gf_weight) * m_sp_scale;
			}
			else {
				triplets.push_back(Eigen::Triplet<double>(index, index, 1.0));
				b_vec[index] = 0;
			}
	}

	std::cout << "V-part\n";
	//v-updates
	for (int j = 0; j < nj + 1; ++j)
		for (int i = 0; i < ni; ++i)
		{
			int index = v_index(i, j);
			if (is_active_face(false, i, j)) {
				double gf_weight = liquid_weights(i, j, 1);
				if (gf_weight < 0.01) gf_weight = 0.01;
				triplets.push_back(Eigen::Triplet<double>(index, index, v_weights(i, j)*gf_weight));
				b_vec[index] = v_weights(i, j)*gf_weight*v(i, j);

				if (liquid_phi(i, j - 1) < 0)
					triplets.push_back(Eigen::Triplet<double>(index, c_index(i, j - 1), -v_weights(i, j) / dx));
				else if (m_sp_set)
					b_vec[index] += lerp(m_surface_pressure(i, j - 1), m_surface_pressure(i, j), gf_weight) * m_sp_scale;

				if (liquid_phi(i, j) < 0)
					triplets.push_back(Eigen::Triplet<double>(index, c_index(i, j), +v_weights(i, j) / dx));
				else if (m_sp_set)
					b_vec[index] -= lerp(m_surface_pressure(i, j - 1), m_surface_pressure(i, j), gf_weight) * m_sp_scale;
		}
		else {
			triplets.push_back(Eigen::Triplet<double>(index, index, 1.0));
			b_vec[index] = 0;
		}
	}

	std::cout << "Divergence\n";

	//divergence-free bits
	for (int j = 0; j < nj; ++j) {
		for (int i = 0; i < ni; ++i) {

			int index = c_index(i, j);
			float centre_phi = liquid_phi(i, j);

			if (centre_phi < 0 && (is_active_face(true, i, j) || is_active_face(true, i + 1, j) || is_active_face(false, i, j) || is_active_face(false, i, j + 1))) {
				//left
				if (is_active_face(true, i, j)) {
					float term = u_weights(i, j) / (dx);
					int left_index = u_index(i, j);
					triplets.push_back(Eigen::Triplet<double>(index, left_index, term));

					//symmetrize
					//triplets.push_back(Eigen::Triplet<double>(left_index, index, term));
				}
				if (is_active_face(true, i + 1, j)) {
					float term = u_weights(i + 1, j) / (dx);
					int right_index = u_index(i + 1, j);
					triplets.push_back(Eigen::Triplet<double>(index, right_index, -term));

					//symmetrize
					//triplets.push_back(Eigen::Triplet<double>(right_index, index, -term));
				}
				if (is_active_face(false, i, j)) {
					float term = v_weights(i, j) / (dx);
					int bottom_index = v_index(i, j);
					triplets.push_back(Eigen::Triplet<double>(index, bottom_index, term));
					//symmetrize
					//triplets.push_back(Eigen::Triplet<double>(bottom_index, index, term));
				}

				if (is_active_face(false, i, j + 1)) {
					float term = v_weights(i, j + 1) / (dx);
					int top_index = v_index(i, j + 1);
					triplets.push_back(Eigen::Triplet<double>(index, top_index, -term));

					//symmetrize
					//triplets.push_back(Eigen::Triplet<double>(top_index, index, -term));
				}
				b_vec[index] = 0;
			}
			else {
				triplets.push_back(Eigen::Triplet<double>(index, index, 1));
				b_vec(index) = 0;
			}

		}
	}


	bool do_bubbles = true;

	std::cout << "Bubbles\n";

	//bubble constraints

	std::sort(bubbles.begin(), bubbles.end(),
		[](const std::vector<Vec2ui> &v0, const std::vector<Vec2ui> &v1) -> bool
	{ return v0.size() < v1.size(); });

	std::vector<std::vector<Vec3ui>> temp_bubbles;
	for (int b = 0; b < bubbles.size() - 1; ++b) {
		//one constraint for each of the bubbles except one.
		int index = b_index(b);
		if (do_bubbles) {

			std::vector<Vec3ui> temp_bface;
			for (int c = 0; c < bubbles[b].size(); ++c)
			{
				Vec2ui cell = bubbles[b][c];

				int i = cell[0];
				int j = cell[1];
				if (u_weights(i, j) > 0 && liquid_phi(i - 1, j) < 0) {

					temp_bface.push_back(Vec3ui(i, j, 0));

					int face_index = u_index(i, j);
					double term = -u_weights(i, j) / dx;
					triplets.push_back(Eigen::Triplet<double>(index, face_index, term));
					triplets.push_back(Eigen::Triplet<double>(face_index, index, term));
				}
				if (u_weights(i + 1, j) > 0 && liquid_phi(i + 1, j) < 0) {
					temp_bface.push_back(Vec3ui(i+1, j, 0));
					int face_index = u_index(i + 1, j);
					double term = u_weights(i + 1, j) / dx;
					triplets.push_back(Eigen::Triplet<double>(index, face_index, term));
					triplets.push_back(Eigen::Triplet<double>(face_index, index, term));
				}
				if (v_weights(i, j) > 0 && liquid_phi(i, j - 1) < 0) {
					temp_bface.push_back(Vec3ui(i, j, 1));
					int face_index = v_index(i, j);
					double term = -v_weights(i, j) / dx;
					triplets.push_back(Eigen::Triplet<double>(index, face_index, term));
					triplets.push_back(Eigen::Triplet<double>(face_index, index, term));
				}
				if (v_weights(i, j + 1) > 0 && liquid_phi(i, j + 1) < 0) {
					temp_bface.push_back(Vec3ui(i, j + 1, 1));
					int face_index = v_index(i, j + 1);
					double term = v_weights(i, j + 1) / dx;
					triplets.push_back(Eigen::Triplet<double>(index, face_index, term));
					triplets.push_back(Eigen::Triplet<double>(face_index, index, term));
				}

			}

			temp_bubbles.push_back(temp_bface);
		}
		else {
			triplets.push_back(Eigen::Triplet<double>(index, index, 1));
		}
		b_vec(index) = 0;

	}


	bool SPD_form = true; //SPD form seems to be slower with this solver (due to dense blocks?)

	std::cout << "Build from triplets\n";
	//solve system
	A_mat.setFromTriplets(triplets.begin(), triplets.end());

	//check symmetry
	//check nullspaces
	//check number of surfaces

	//Apply Schur complement to get smaller system to solve.
	if (SPD_form) {
		int faceCount = ni*(nj + 1) + (ni + 1)*nj;
		int pressureCount = ni*nj + bubbles.size() - 1;
		Eigen::SparseMatrix<double> I = A_mat.block(0, 0, faceCount, faceCount);
		Eigen::SparseMatrix<double> I_inv = I;
		for (int k = 0; k < I_inv.outerSize(); ++k)
			for (Eigen::SparseMatrix<double>::InnerIterator it(I_inv, k); it; ++it)
			{
				it.value();
				it.row();   // row index
				it.col();   // col index (here it is equal to k)
				it.index(); // inner index, here it is equal to it.row()
				if (it.value() > 0.001) {
					I_inv.coeffRef(it.row(), it.col()) = 1.0 / it.value();
				}
			}

		Eigen::SparseMatrix<double> D = A_mat.block(faceCount, 0, pressureCount, faceCount);
		Eigen::SparseMatrix<double> G = A_mat.block(0, faceCount, faceCount, pressureCount);
		Eigen::SparseMatrix<double> BR = A_mat.block(faceCount, faceCount, pressureCount, pressureCount);
		Eigen::SparseMatrix<double> Poisson = BR - D * I_inv * G;
		std::cout << "FaceCount: " << faceCount << std::endl;
		std::cout << "pressureCount : " << pressureCount << std::endl;

		std::cout << "D Size: " << D.rows() << " " << D.cols() << std::endl;
		std::cout << "P Size: " << Poisson.rows() << " " << Poisson.cols() << std::endl;
		//Eigen::saveMarket(Poisson, "matrix2.mat");

		Eigen::VectorXd c_vec = b_vec.block(0, 0, faceCount, 1);
		Eigen::VectorXd d_vec = b_vec.block(faceCount, 0, pressureCount, 1);
		Eigen::VectorXd rhs2 = d_vec - D * I_inv * c_vec;
		std::cout << "RHS length: " << rhs2.size() << std::endl;

		//now given the pressure, compute the velocities
		Eigen::SparseLU<Eigen::SparseMatrix<double> > solver2;
		solver2.compute(Poisson);

		if (solver2.info() != Eigen::Success) {
			std::cout << "Small sys fail\n";
		}
		Eigen::VectorXd soln_P = solver2.solve(rhs2);

		if (solver2.info() != Eigen::Success) {
			std::cout << "Small sys fail\n";
		}
		Eigen::VectorXd soln_U = I_inv*(c_vec - G*soln_P);

		//store back results
		for (int j = 0; j < nj; ++j)
			for (int i = 0; i < ni + 1; ++i)
			{
				int index = u_index(i, j);
				if (is_active_face(true, i, j)) {
					vel(i, j, 0) = (float)soln_U(index);
					valid(i, j, 0) = 1;
				}
				else
					valid(i, j, 0) = 0;
		}
		for (int j = 0; j < nj + 1; ++j)
			for (int i = 0; i < ni; ++i)
			{
				int index = v_index(i, j);
				if (is_active_face(false, i, j)) {
					vel(i, j, 1) = (float)soln_U(index);
					valid(i, j, 1) = 1;
				}
				else
					valid(i, j, 1) = 0;
		}

	}
}
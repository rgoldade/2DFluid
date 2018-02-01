#include <limits>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "Core.h"
#include "UniformGrid.h"

#include "MeshRebuilder.h"

void MeshRebuilder::draw_intersections(Renderer &renderer)
{
	std::vector<Vec2R> vert_points;

	for (auto v : m_intersection_list) vert_points.push_back(idx_to_ws(v));

	renderer.add_points(vert_points, Vec3f(0,0,1), 2);
}

static const Vec2i cube_node_offset[4] =
				{ Vec2i(-1, -1), Vec2i(-1, 1), Vec2i(1, 1), Vec2i(1, -1) };

static const Vec2i cube_edge_offset[4] =
				{ Vec2i(0,-1), Vec2i(1,0), Vec2i(0,1), Vec2i(-1,0) };

static const Vec2i edge_cube_offset[2][2] =
				{ { Vec2i(0, -1), Vec2i(0, 1) },
				{ Vec2i(-1, 0), Vec2i(1, 0) } };

static const Vec2i edge_node_offset[2][2] =
				{ { Vec2i(-1, 0), Vec2i(1, 0) },
				{ Vec2i(0, -1), Vec2i(0, 1) } };

static const Vec2i node_edge_offset[2][2] =
				{ { Vec2i(-1, 0), Vec2i(1, 0) },
				{ Vec2i(0, -1), Vec2i(0, 1) } };

static Vec2R line_intersect_point(const Vec2R& a, const Vec2R& b, const Vec2R& c, const Vec2R& d)
{
	Real x1 = a[0];
	Real x2 = b[0];
	Real x3 = c[0];
	Real x4 = d[0];

	Real y1 = a[1];
	Real y2 = b[1];
	Real y3 = c[1];
	Real y4 = d[1];

	// Compute determinant -- skipping determinant check
	// assuming a predicate has been verified already
	Real det = (y2 - y1) * (x3 - x4) - (y4 - y3) * (x1 - x2);

	Real rhs_0 = x1 * y2 - x2 * y1;
	Real rhs_1 = x3 * y4 - x4 * y3;

	Vec2R point;

	point[0] = ((x3 - x4) * rhs_0 - (x1 - x2) * rhs_1) / det;
	point[1] = ((y3 - y4) * rhs_0 + (y2 - y1) * rhs_1) / det;

	return point;
}

void MeshRebuilder::rebuild(Mesh2D &mesh, bool dualcontouring)
{
	// Debug list reset
	m_intersection_list.clear();

	// Scale the mesh to the index space of the backgroud
	// grid during the rebuild phase. The final mesh state
	// will be scaled back to world space.
	mesh.scale(1 / m_dx);

	std::vector<Vec2R> vert_norms;
	
	if (dualcontouring)
	{
		// Build vertex normals to 
		vert_norms.resize(mesh.vert_size(), Vec2R(0));

		for (size_t e = 0; e < mesh.edge_size(); ++e)
		{
			vert_norms[mesh.edge(e).vert(0)] += mesh.unnormal(e);
			vert_norms[mesh.edge(e).vert(1)] += mesh.unnormal(e);
		}

		for (auto &vn : vert_norms) vn = normalized(vn);
	}

	// Build AABB for mesh to initialize storage grids
	Real max = std::numeric_limits<Real>::max();
	Real min = std::numeric_limits<Real>::lowest();
	
	Vec2R bb_min(max);
	Vec2R bb_max(min);

	for (auto e : mesh.edges())
	{
		Vec2R v0 = mesh.vert(e.vert(0)).point();
		Vec2R v1 = mesh.vert(e.vert(1)).point();

		update_both_minmax(v0, bb_min, bb_max);
		update_both_minmax(v1, bb_min, bb_max);
	}

	bb_min -= Vec2R(1.);

	// Build grids to store intersections, normals
	// and direction partiy (for multiple intersections)
	// and multiple intersection counts 

	// It's easiest to represent the pieces of the grid with
	// direct integers (the node at the origin is (0,0) and
	// it's edge to the right is (1,0))

	// TODO: maybe go back to offsets?
	Vec2i full_bb_min = 2 * floor(bb_min);
	Vec2i full_bb_max = 2 * ceil(bb_max);

	// Edge intersections
	m_intersections.reset(mesh.edge_size());

	// Intersection normals
	m_normals.reset(mesh.edge_size());

	// Directional parity count
	m_directions.reset(mesh.edge_size());

	m_count.reset(mesh.edge_size());

	// Inside/outside node 
	m_nodes.reset(2 * mesh.edge_size());

	for (size_t ex = 0; ex < mesh.edge_size(); ++ex)// (auto e : mesh.edges())
	{
		Edge2D e = mesh.edge(ex);
		Vec2R v0 = mesh.vert(e.vert(0)).point();
		Vec2R v1 = mesh.vert(e.vert(1)).point();

		// Degenerate check
		if (v0 == v1) continue;

		Vec2R vmin, vmax;
		minmax(v0, v1, vmin, vmax);

		Vec2i bb_cmin = ceil(vmin);
		Vec2i bb_fmin = floor(vmin) - Vec2i(1);
		Vec2i bb_fmax = floor(vmax);

		for (int j = bb_cmin[1]; j <= bb_fmax[1]; ++j)
			for (int i = bb_fmax[0]; i >= bb_fmin[0]; --i)
			{
				Vec2R grid_node(i, j);

				Vec2R ip;
				Vec2i idx;

				// Check if mesh edge crosses grid edge between grid_node and its
				// paired vertex
				INTERSECTION result = exact_edge_intersect(v0, v1, grid_node, XAXIS);

				if (result == NO) continue; //If there is no intersection, skip to next iteration

				// The grid edge either intersects the mesh or the grid node is on the mesh.
				// We adjust the index accordingly to point to the grid-edge or the grid-node
				// if it's an intersection or ON (respectively)
				else if (result == YES)
				{
					ip = line_intersect_point(v0, v1, grid_node, grid_node + Vec2R(1, 0));
					
					// Clamp ip to grid edge line segment
					ip[1] = grid_node[1];
					if (ip[0] < grid_node[0]) ip[0] = grid_node[0];
					if (ip[0] > grid_node[0] + 1.) ip[0] = grid_node[0] + 1.;

					idx = Vec2i(i * 2 + 1, j * 2);

					// Insert a dummy hold for grid nodes to be updated later
					m_nodes.insert(idx - Vec2i(1, 0), 0);
					m_nodes.insert(idx + Vec2i(1, 0), 0);

					// Debug list
					m_intersection_list.push_back(ip);
				}
				else if (result == ON)
				{
					// Because the grid node falls on the mesh-edge, the intersection
					// point is exactly the grid edge
					ip = grid_node;
					idx = Vec2i(i * 2, j * 2);

					m_nodes.insert(idx, 0);
					m_nodes.insert(idx - Vec2i(2, 0), 0);
					m_nodes.insert(idx + Vec2i(2, 0), 0);
					m_nodes.insert(idx - Vec2i(0, 2), 0);
					m_nodes.insert(idx + Vec2i(0, 2), 0);

					//TODO: stand in flag to say there is a surface node here.. remove one day
					// TODO: this might actually not even be needed tbh
					if (!m_directions.find(idx)) m_directions.insert(idx, 0);

					// Debug list
					m_intersection_list.push_back(ip);
				}

				// If we're re-meshing with dual contouring, we want to store the normal of the
				// mesh at the intersection point.
				Vec2R bary_norm;
				if (dualcontouring)
				{
					// Compute barycenter weights at the intersection point
					//Vec2R b0 = ip - v0, b1 = v1 - v0;
					//Real s = mag(b0) / mag(b1);
					//s = (s > 1.) ? 1 : s;
					//bary_norm = (1. - s) * vert_norms[e.vert(0)] + s * vert_norms[e.vert(1)];
					//normalize(bary_norm);

					bary_norm = mesh.normal(ex);
				}

				// Store intersection information -- agnostic to node or edge
				// Check for duplicate insertion
				if (m_count.find(idx))
				{
					Real count = m_count.get(idx);

					ip = (m_intersections.get(idx) * count + ip) / (count + 1.);
					m_intersections.replace(idx, ip);

					// TODO: replace edge normal with vertex normal if "on"

					// Averaging normals is probably not a good idea..
					// TODO: come up with something better.. two opposing normals could give a zero vector...
					// It might be better just to chuck out setting normals anyway
					if (dualcontouring)
					{
						Vec2R new_norm = (m_normals.get(idx) * count + bary_norm) / (++count);
						normalize(new_norm);
						m_normals.replace(idx, new_norm);
					}

					m_count.replace(idx, count);
				}

				else
				{
					m_intersections.insert(idx, ip);
					if (dualcontouring) m_normals.insert(idx, bary_norm);
					m_count.insert(idx, 1.);
				}

				//Exit loop to prevent false positives
				break;
			}

		for (int i = bb_cmin[0]; i <= bb_fmax[0]; ++i)
			for (int j = bb_fmax[1]; j >= bb_fmin[1]; --j)
			{
				Vec2R grid_node(i, j);

				Vec2R ip;
				Vec2i idx;

				// Check if mesh edge crosses grid edge between grid_node and its
				// paired vertex
				INTERSECTION result = exact_edge_intersect(v0, v1, grid_node, YAXIS);

				if (result == NO) continue;

				if (result == YES)
				{
					//Get intersection point
					ip = line_intersect_point(v0, v1, grid_node, grid_node + Vec2R(0, 1));

					// Clamp ip to grid edge line segment
					ip[0] = grid_node[0];
					if (ip[1] < grid_node[1]) ip[1] = grid_node[1];
					if (ip[1] > grid_node[1] + 1.) ip[1] = grid_node[1] + 1.;


					//Edge position key for storage
					idx = Vec2i(i * 2, j * 2 + 1);

					m_nodes.insert(idx - Vec2i(0, 1), 0);
					m_nodes.insert(idx + Vec2i(0, 1), 0);

					// Debug list
					m_intersection_list.push_back(ip);
				}
				if (result == ON)
				{
					ip = grid_node;
					idx = Vec2i(i * 2, j * 2);

					m_nodes.insert(idx, false);
					m_nodes.insert(idx - Vec2i(2, 0), 0);
					m_nodes.insert(idx + Vec2i(2, 0), 0);
					m_nodes.insert(idx - Vec2i(0, 2), 0);
					m_nodes.insert(idx + Vec2i(0, 2), 0);

					// Debug list
					m_intersection_list.push_back(ip);
				}

				// If we're re-meshing with dual contouring, we want to store the normal of the
				// mesh at the intersection point.
				Vec2R bary_norm;
				if (dualcontouring)
				{
					// Compute barycenter weights at the intersection point
					//Vec2R b0 = ip - v0, b1 = v1 - v0;
					//Real s = mag(b0) / mag(b1);
					//s = (s > 1.) ? 1 : s;
					//bary_norm = (1. - s) * vert_norms[e.vert(0)] + s * vert_norms[e.vert(1)];
					//normalize(bary_norm);

					bary_norm = mesh.normal(ex);
				}

				if (m_count.find(idx))
				{
					Real count = m_count.get(idx);

					ip = (m_intersections.get(idx) * count + ip) / (count + 1.);
					m_intersections.replace(idx, ip);

					if (dualcontouring)
					{
						Vec2R new_norm = (m_normals.get(idx) * count + bary_norm) / (++count);
						normalize(new_norm);
						m_normals.replace(idx, new_norm);
					}

					m_count.replace(idx, count);
				}
				else
				{
					m_intersections.insert(idx, ip);
					if (dualcontouring) m_normals.insert(idx, bary_norm);
					m_count.insert(idx, 1.);
				}

				// Update direction parity
				int direction;

				// Check if it already exists (//TODO: this could probably be put into the above check right?)
				if (!m_directions.find(idx)) direction = 0;
				else direction = m_directions.get(idx);


				if (v0[0] > v1[0]) // Entering into mesh edge is pointed upwards
					direction += 1;
				else
				{
					assert(v0[0] < v1[0]);
					direction -= 1;
				}

				m_directions.replace(idx, direction);

				//Exit loop to prevent false positives
				break;
			}
	}

	// TODO: write a bounding box iteration loop to be replaced in the future with a proper 
	// list iteration
	std::vector<Vec2i> magic_edges;
	for (int i = full_bb_min[0]; i <= full_bb_max[0]; i += 2)
	{
		int parity = 0;

		for (int j = full_bb_min[1]; j <= full_bb_max[1]; j += 2)
		{
			Vec2i idx(i, j);

			//Check if a node has been assigned an intersection
			//TODO: this is rare.. might be a costly look up
			if (m_directions.find(idx))
			{
				int parchange = m_directions.get(idx);
				if (parchange < 0) parity += parchange;

				assert(m_nodes.find(idx));
				m_nodes.replace(idx, (parity > 0));

				if (parchange > 0) parity += parchange;

				//Add to "magic edges" list
				magic_edges.push_back(idx + node_edge_offset[0][0]);
				magic_edges.push_back(idx + node_edge_offset[0][1]);
				magic_edges.push_back(idx + node_edge_offset[1][0]);
				magic_edges.push_back(idx + node_edge_offset[1][1]);
			}
			else if (m_nodes.find(idx))
			{
				m_nodes.replace(idx, (parity > 0));
			}

			// Check for edge intersections on the upcoming edge
			Vec2i eidx = idx + Vec2i(0, 1);
			if (m_directions.find(eidx)) parity += m_directions.get(eidx);
		}
	}

	// Create edge intersections associated with node intersections based on node parity
	// Since the intersection happened at the node, the intersection should be placed there.
	// Creating the intersection will move the averaged intersection position but this is not unlike
	// a level set when trying to find the zero crossing.

	if (magic_edges.size() > 0)
	{
		auto vecSort = [](const Vec2i& a, const Vec2i& b) -> bool
		{
			if (a[0] < b[0]) return true;
			else if (a[0] == b[0] && a[1] < b[1]) return true;
			else return false;
		};

		std::sort(magic_edges.begin(), magic_edges.end(), vecSort);
		Vec2i eidx = magic_edges[0] - Vec2i(1, 1); // offset backwards to pass duplicate check

		for (const auto& e : magic_edges)
		{
			if (eidx != e)
			{
				eidx = e;
				Vec2i offset;
				if (eidx[0] % 2 == 0) offset = Vec2i(0, 1);
				else offset = Vec2i(1, 0);

				// "at" is a slower but uses safety checks.. this shouldn't fail
				bool bnode = m_nodes.at(eidx - offset);
				bool fnode = m_nodes.at(eidx + offset);

				// Verify that there is a parity change in the end
				if (bnode != fnode)
				{
					Vec2R ip, norm;
					if (!bnode)
					{
						//Verify that the back node was "on"
						if (m_directions.find(eidx - offset))
						{
							ip = m_intersections.get(eidx - offset);
							norm = m_normals.get(eidx - offset);
						}
						else if (m_directions.find(eidx + offset))
						{
							ip = m_intersections.get(eidx + offset);
							norm = m_normals.get(eidx + offset);
						}
						else assert(false);
					}
					else// if (!fnode)
					{
						// Verify that the front node was "on"
						if (m_directions.find(eidx + offset))
						{
							ip = m_intersections.get(eidx + offset);
							norm = m_normals.get(eidx + offset);
						}
						// If not (which could happen), check back no was "on"
						else if (m_directions.find(eidx - offset))
						{
							ip = m_intersections.get(eidx - offset);
							norm = m_normals.get(eidx - offset);
						}
						else assert(false);
					}

					if (m_count.find(eidx))
					{
						Real count = m_count.get(eidx);

						ip = (m_intersections.get(eidx) * count + ip) / (count + 1.);
						m_intersections.replace(eidx, ip);

						norm = (m_normals.get(eidx) * count + norm) / (++count);
						m_normals.replace(eidx, norm);

						m_count.replace(eidx, count);
					}
					else
					{
						m_intersections.replace(eidx, ip);
						m_normals.replace(eidx, norm);
						m_count.replace(eidx, 1.);
					}
				}
			}
		}
	}

	// Build table of approved grid intersections
	HashGrid2D<Vec2R> new_intersections;
	new_intersections.reset(m_intersections.size());

	for (int i = full_bb_min[0] + 1; i <= full_bb_max[0]; i += 2)
		for (int j = full_bb_min[1]; j <= full_bb_max[1]; j += 2)
		{
			if (m_intersections.find(Vec2i(i, j)))
			{
				Vec2i idx(i, j);
				//Only keep edge intersections with inside/outside changes	
				if (m_nodes.get(idx + edge_node_offset[0][0]) != m_nodes.get(idx + edge_node_offset[0][1]))
				{
					Vec2R vert = m_intersections.get(idx);
					new_intersections.insert(idx, vert);
				}
			}
		}

	// Instead of deleting points, this just creates a copy of new points.. not sure
	// if this is really the best move in the long run
	for (int i = full_bb_min[0]; i <= full_bb_max[0]; i += 2)
		for (int j = full_bb_min[1] + 1; j <= full_bb_max[1]; j += 2)
		{
			if (m_intersections.find(Vec2i(i, j)))
			{
				Vec2i idx(i, j);
				//Only keep edge intersections with inside/outside changes				
				if (m_nodes.get(idx + edge_node_offset[1][0]) != m_nodes.get(idx + edge_node_offset[1][1]))
				{
					Vec2R vert = m_intersections.get(idx);
					new_intersections.insert(idx, vert);
				}
			}
		}

	// Time to clear the original mesh and build a new one
	mesh.clear();
	if (dualcontouring)
	{
		std::vector<Vec2R> vert_list;
		HashGrid2D<size_t> mesh_verts(new_intersections.size());

		for (int i = full_bb_min[0] + 1; i <= full_bb_max[0]; i += 2)
			for (int j = full_bb_min[1] + 1; j <= full_bb_max[1]; j += 2)
			{
				Vec2i idx(i, j);

				if (new_intersections.find(idx + cube_edge_offset[0]) ||
					new_intersections.find(idx + cube_edge_offset[1]) ||
					new_intersections.find(idx + cube_edge_offset[2]) ||
					new_intersections.find(idx + cube_edge_offset[3]))
				{
					size_t planecount = 0;
					for (size_t e = 0; e < 4; ++e)
					{
						Vec2i eidx = idx + cube_edge_offset[e];
						if (new_intersections.find(eidx))
						{
							++planecount;
						}
					}

					Eigen::MatrixXd A(planecount, 2);
					Eigen::VectorXd b(planecount);
					Eigen::VectorXd xcom = Eigen::VectorXd::Zero(2);

					Real count = 0;

					for (size_t e = 0; e < 4; ++e)
					{
						Vec2i eidx = idx + cube_edge_offset[e];
						if (new_intersections.find(eidx))
						{
							Vec2i n0;
							Vec2i n1;
							if (eidx[0] % 2 != 0)
							{
								n0 = (eidx + edge_node_offset[0][0]);
								n1 = (eidx + edge_node_offset[0][1]);
							}
							else
							{
								n0 = (eidx + edge_node_offset[1][0]);
								n1 = (eidx + edge_node_offset[1][1]);
							}
							assert(m_nodes.at(n0) != m_nodes.at(n1));

							Vec2R n = m_normals.at(eidx);
							Vec2R p = new_intersections.at(eidx);

							A(count, 0) = n[0]; A(count, 1) = n[1];
							b(count) = dot(n, p);

							xcom[0] += p[0]; xcom[1] += p[1];
							++count;
						}
					}

					assert(count > 1.);

					xcom /= count;

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV); //TODO: double check the appropriate flags, etc
					svd.setThreshold(1E-2);

					Eigen::VectorXd np = xcom + svd.solve(b - A * xcom);

					Vec2R cm(xcom[0], xcom[1]);

					Vec2i bbmin = floor(cm);
					Vec2i bbmax = ceil(cm);
					
					if (np[0] < bbmin[0] || np[0] > bbmax[0]
						|| np[1] < bbmin[1] || np[1] > bbmax[1])
							np = xcom;

					/*Vec2i bbmin = floor(cm) - Vec2i(1., 1.);
					Vec2i bbmax = ceil(cm) + Vec2i(1., 1.);

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
					}*/

					// Store index of dc point at the cell for look up during the edge building phase
					// This idx refers to cell, not node 
					size_t vidx = vert_list.size();

					vert_list.push_back(Vec2R(np[0], np[1]));
					mesh_verts.insert(idx, vidx);
				}
			}

		std::vector<Vec2i> edge_list;
		for (int i = full_bb_min[0] + 1; i <= full_bb_max[0]; i += 2)
			for (int j = full_bb_min[1]; j <= full_bb_max[1]; j += 2)
			{
				Vec2i eidx(i, j);
				if (new_intersections.find(eidx))
				{
					assert(mesh_verts.find(eidx + edge_cube_offset[0][0]));
					size_t v0 = mesh_verts.get(eidx + edge_cube_offset[0][0]);

					assert(mesh_verts.find(eidx + edge_cube_offset[0][1]));
					size_t v2 = mesh_verts.get(eidx + edge_cube_offset[0][1]);

					if (m_nodes.at(eidx + edge_node_offset[0][0]))
					{
						edge_list.push_back(Vec2i(v2, v0));
					}
					else
					{
						edge_list.push_back(Vec2i(v0, v2));
					}
				}
			}

		for (int i = full_bb_min[0]; i <= full_bb_max[0]; i += 2)
			for (int j = full_bb_min[1] + 1; j <= full_bb_max[1]; j += 2)
			{
				Vec2i eidx(i, j);
				if (new_intersections.find(eidx))
				{
					assert(mesh_verts.find(eidx + edge_cube_offset[1][0]));
					size_t v0 = mesh_verts.get(eidx + edge_cube_offset[1][0]);

					assert(mesh_verts.find(eidx + edge_cube_offset[1][1]));
					size_t v2 = mesh_verts.get(eidx + edge_cube_offset[1][1]);

					if (m_nodes.at(eidx + edge_node_offset[1][1]))
					{
						edge_list.push_back(Vec2i(v2, v0));
					}
					else
					{
						edge_list.push_back(Vec2i(v0, v2));
					}
				}
			}

		mesh.initialize(edge_list, vert_list);

		//std::cout << "Edge collapse step" << std::endl;

		//// Edge collapse
		//for (auto e : mesh.edges())
		//{
		//	Vec2R v0 = mesh.vert(e.vert(0)).point();
		//	Vec2R v1 = mesh.vert(e.vert(1)).point();

		//	if (dist(v0, v1) < 0.1)
		//	{
		//		std::cout << "Collapsing edge. Vertex 1: " << v0 << ". Vertex 2: " << v1 << std::endl;
		//		v0 = (v0 + v1) * .5;
		//		v1 = v0;

		//		std::cout << "Collapsed edge. Vertex 1: " << v0 << ". Vertex 2: " << v1 << std::endl;

		//		mesh.set_vert(e.vert(0), v0);
		//		mesh.set_vert(e.vert(1), v1);
		//	}
		//}
	}
	// Rebuild with marching squares
	else
	{
		std::vector<Vec2R> vert_list;
		HashGrid2D<size_t> mesh_verts(new_intersections.size());
		// Insert vertices into mesh and store index for edge building
		for (int i = full_bb_min[0] + 1; i <= full_bb_max[0]; i += 2)
			for (int j = full_bb_min[1]; j <= full_bb_max[1]; j += 2)
			{
				Vec2i eidx(i, j);
				if (new_intersections.find(eidx))
				{
					Vec2R vert = new_intersections.at(eidx);
					size_t vidx = vert_list.size();
					vert_list.push_back(vert);

					mesh_verts.insert(eidx, vidx);
				}
			}

		for (int i = full_bb_min[0]; i <= full_bb_max[0]; i += 2)
			for (int j = full_bb_min[1] + 1; j <= full_bb_max[1]; j += 2)
			{
				Vec2i eidx(i, j);
				if (new_intersections.find(eidx))
				{
					Vec2R vert = new_intersections.at(eidx);
					size_t vidx = vert_list.size();
					vert_list.push_back(vert);

					mesh_verts.insert(eidx, vidx);
				}
			}

		std::vector<Vec2i> edge_list;
		//Build edges using marching cubes stencil -- having an "active grid" would work
		for (int i = full_bb_min[0] + 1; i <= full_bb_max[0] - 1; i += 2)
			for (int j = full_bb_min[1] + 1; j <= full_bb_max[1] - 1; j += 2)
			{
				// Check for edge intersections -- assuming if there's one, there should be two
				// The assert checks *should* fail if there aren't two
				Vec2i idx(i, j);
				if (new_intersections.find(idx + cube_edge_offset[0]) ||
					new_intersections.find(idx + cube_edge_offset[1]) ||
					new_intersections.find(idx + cube_edge_offset[2]) ||
					new_intersections.find(idx + cube_edge_offset[3]))
				{
					size_t key = 0;

					Vec2i dl_idx = idx + cube_node_offset[0];
					Vec2i ul_idx = idx + cube_node_offset[1];
					Vec2i ur_idx = idx + cube_node_offset[2];
					Vec2i dr_idx = idx + cube_node_offset[3];

					if (m_nodes.find(dl_idx))
					{
						if (m_nodes.at(dl_idx)) key += 1;
					}
					else
					{
						// If the previous node does no exist, these two nodes must be the same
						// parity or there is an unreported intersection.
						assert(m_nodes.at(ul_idx) == m_nodes.at(dr_idx));

						// If the neighbouring node is inside, this node must also be inside otherwise
						// there is a problem in the code
						if (m_nodes.at(ul_idx)) 		key += 1;
					}

					if (m_nodes.find(dr_idx))
					{
						if (m_nodes.at(dr_idx))		key += 2;
					}
					else
					{
						assert(m_nodes.at(dl_idx) == m_nodes.at(ur_idx));
						if (m_nodes.at(ur_idx))		key += 2;
					}

					if (m_nodes.find(ur_idx))
					{
						if (m_nodes.at(ur_idx))		key += 4;
					}
					else
					{
						assert(m_nodes.at(ul_idx) == m_nodes.at(dr_idx));
						if (m_nodes.at(ul_idx))	key += 4;
					}

					if (m_nodes.find(ul_idx))
					{
						if (m_nodes.at(ul_idx))	key += 8;
					}
					else
					{
						assert(m_nodes.at(dl_idx) == m_nodes.at(ur_idx));
						if (m_nodes.at(dl_idx))	key += 8;
					}

					for (size_t e = 0; e < 4 && mc_template[key][e] >= 0; e += 2)
					{
						size_t edge = mc_template[key][e];
						assert(mesh_verts.find(idx + cube_edge_offset[edge]));
						size_t v0 = mesh_verts.get(idx + cube_edge_offset[edge]);

						edge = mc_template[key][e + 1];
						assert(mesh_verts.find(idx + cube_edge_offset[edge]));
						size_t v1 = mesh_verts.get(idx + cube_edge_offset[edge]);

						edge_list.push_back(Vec2i(v0, v1));

					}
				}
			}
		mesh.initialize(edge_list, vert_list);
	}

	// Back to world space
	mesh.scale(m_dx);
}

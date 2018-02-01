#include "Mesh2D.h"

void Mesh2D::draw_mesh(Renderer& renderer,
	Vec3f edge_colour,
	bool render_edge_normals,
	bool render_verts,
	Vec3f vert_colour)
{
	std::vector<Vec2R> start_points;
	std::vector<Vec2R> end_points;

	for (auto e : m_edges)
	{
		Vec2R sp = m_verts[e.vert(0)].point();
		start_points.push_back(sp);

		Vec2R ep = m_verts[e.vert(1)].point();
		end_points.push_back(ep);
	}

	renderer.add_lines(start_points, end_points, edge_colour);
	
	if (render_edge_normals)
	{
		std::vector<Vec2R> start_norms;
		std::vector<Vec2R> end_norms;

		// Scale by average edge length
		Real avg_length = 0.;
		Real inv_edges = 1.0 / (Real)(m_edges.size());
		for (auto e : m_edges)
			avg_length += mag(m_verts[e.vert(0)].point() - m_verts[e.vert(1)].point()) * inv_edges;
	
		for (size_t e = 0; e < m_edges.size(); ++e)
		{
			Vec2R p = (m_verts[m_edges[e].vert(0)].point() + m_verts[m_edges[e].vert(1)].point()) * .5;
			Vec2R norm = normal(e);
			
			start_norms.push_back(p);
			end_norms.push_back(p + norm * avg_length);
		}

		renderer.add_lines(start_norms, end_norms, Vec3f(0.));
	}

	if (render_verts)
	{
		std::vector<Vec2R> vert_points;

		for (auto v : m_verts) vert_points.push_back(v.point());

		renderer.add_points(vert_points, vert_colour, 2);
	}
}

bool Mesh2D::unit_test() const
{
	bool passed = true;

	// Verify vertex has two or more adjacent edges. Meaning no dangling edge.
	for (size_t v = 0; v < m_verts.size(); ++v)
	{
		if (m_verts[v].valence() < 2)
		{
			std::cout << "Unit test failed in valence check. Vertex: " << v << ". Valence: " << m_verts[v].valence() << std::endl;
			passed = false;
		}
	}

	// Verify edge has two adjacent vertices. Meaning no dangling edge.
	for (size_t e = 0; e < m_edges.size(); ++e)
	{
		if (m_edges[e].vert(0) < 0 || m_edges[e].vert(1) < 0)
		{
			std::cout << "Unit test failed in edge's vertex count test. Edge: " << e << std::endl;
			passed = false;
		}
	}

	// Verify vertex's adjacent edge reciprocates
	for (size_t v = 0; v < m_verts.size(); ++v)
	{
		for (size_t vi = 0; vi < m_verts[v].valence(); ++vi)
		{
			int e = m_verts[v].edge(vi);
			if (!m_edges[e].find_vert(v))
			{
				std::cout << "Unit test failed in adjacent edge test. Vertex: " << v << ". Edge: " << e << std::endl;
				passed = false;
			}
		}
	}

	// Verify edge's adjacent vertex reciprocates
	for (size_t e = 0; e < m_edges.size(); ++e)
	{
		size_t v0 = m_edges[e].vert(0);
		if (!m_verts[v0].find_edge(e))
		{
			passed = false;
			std::cout << "Unit test failed in adjacent vertex test. Vertex: " << v0 << ". Edge: " << e << std::endl;
		}

		size_t v1 = m_edges[e].vert(1);
		if (!m_verts[v1].find_edge(e))
		{
			std::cout << "Unit test failed in adjacent vertex test. Vertex: " << v1 << ". Edge: " << e << std::endl;
			passed = false;
		}
	}

	// TODO: write a test that verifies winding order

	return passed;
}
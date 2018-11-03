#pragma once

#include "Common.h"

#include "VectorGrid.h"
#include "Integrator.h"

#include "Renderer.h"

///////////////////////////////////
//
// Mesh2d.h/cpp
// Ryan Goldade 2016
//
// 2-d mesh container with edge and
// vertex accessors.
//
////////////////////////////////////

// Vertex class stores it's point, normal and adjacent edges
// as well as accessors and modifiers

class Vertex2D
{
public:
	Vertex2D() : m_point(Vec2R(0))
	{}

	Vertex2D(const Vec2R& point) : m_point(point)
	{}

	const Vec2R& point() const
	{
		return m_point;
	}

	void set_point(const Vec2R& point)
	{
		m_point = point;
	}

	void operator=(const Vec2R& point)
	{
		m_point = point;
	}

	// Get edge stored at the eidx position in the m_edges list
	unsigned edge(unsigned eidx) const
	{
		assert(eidx < m_edges.size());
		return m_edges[eidx];
	}

	void add_edge(unsigned eidx)
	{
		m_edges.push_back(eidx);
	}

	// Search through the edge list for a matching edge
	// index to old_eidx. If found, replace that edge
	// index with new_eidx.
	bool replace_edge(unsigned old_eidx, unsigned new_eidx)
	{
		auto result = std::find(m_edges.begin(), m_edges.end(), old_eidx);

		if (result == m_edges.end()) return false;
		else
		{
			*result = new_eidx;
		}
		return true;
	}

	// Search through the edge list and return true if 
	// there is an edge index that matches eidx
	bool find_edge(unsigned eidx) const
	{
		auto result = std::find(m_edges.begin(), m_edges.end(), eidx);
		return result != m_edges.end();
	}

	unsigned valence() const
	{
		return m_edges.size();
	}

	template<typename T>
	void operator*=(const T& s)
	{
		m_point *= s;
	}

	template<typename T>
	void operator+=(const T& s)
	{
		m_point += s;
	}

private:

	Vec2R m_point;
	std::vector<unsigned> m_edges;
};

class Edge2D
{
public:
	// Negative magic numbers mean unassigned. There should never be a negative
	// vertex index or a negative material.
	//Edge2D() : m_verts(Vec2i(-1))
	//{}

	Edge2D(const Vec2ui& verts) : m_verts(verts)
	{}

	Vec2ui verts() const
	{
		return m_verts;
	}

	unsigned vert(unsigned vidx) const
	{
		assert(vidx < 2);
		return m_verts[vidx];
	}

	// Given a vertex index, find the opposite vertex index
	// on the edge. An assert fail will be thrown if there
	// is no match as a debug warning.
	unsigned adjacent_vert(unsigned vidx) const
	{
		if (m_verts[0] == vidx)
			return m_verts[1];
		else if (m_verts[1] == vidx)
			return m_verts[0];
		
		assert(false);
		return 0;
	}

	void replace_vert(unsigned old_vidx, unsigned new_vidx)
	{
		if (m_verts[0] == old_vidx)
			m_verts[0] = new_vidx;
		else if (m_verts[1] == old_vidx)
			m_verts[1] = new_vidx;
		else assert(false);
	}

	bool find_vert(unsigned vidx) const
	{
		if (m_verts[0] == vidx || m_verts[1] == vidx) return true;
		return false;
	}

	void reverse()
	{
		std::swap(m_verts[0], m_verts[1]);
	}

private:
	// Each edge can be viewed as having a "tail" and a "head"
	// vertex. The orientation is used in reference to "left" and
	// "right" turns when talking about the materials on the edge.
	Vec2ui m_verts;
};

class Mesh2D
{
public:
	// Vanilla constructor leaves initialization up to the caller
	Mesh2D()
	{}

	// Initialize mesh container with edges and the associated vertices.
	// Input mesh should be water-tight with no dangling edges
	// (i.e. no vertex has a valence less than 2).
	Mesh2D(const std::vector<Vec2ui>& edges, const std::vector<Vec2R>& verts)
	{
		m_edges.reserve(edges.size());
		for (const auto& edge : edges) m_edges.push_back(Edge2D(edge));

		m_verts.reserve(verts.size());
		for (const auto& vert : verts) m_verts.push_back(Vertex2D(vert));

		// Update vertices to store adjacent edges in their edge lists
		for (unsigned e = 0; e < m_edges.size(); ++e)
		{
			unsigned v0 = m_edges[e].vert(0);
			m_verts[v0].add_edge(e);

			unsigned v1 = m_edges[e].vert(1);
			m_verts[v1].add_edge(e);
		}
	}

	void reinitialize(const std::vector<Vec2ui>& edges, const std::vector<Vec2R>& verts)
	{
		m_edges.clear();
		m_edges.reserve(edges.size());
		for (const auto& edge : edges) m_edges.push_back(Edge2D(edge));

		m_verts.clear();
		m_verts.reserve(verts.size());
		for (const auto& vert : verts) m_verts.push_back(Vertex2D(vert));

		// Update vertices to store adjacent edges in their edge lists
		for (unsigned e = 0; e < m_edges.size(); ++e)
		{
			unsigned v0 = m_edges[e].vert(0);
			m_verts[v0].add_edge(e);

			unsigned v1 = m_edges[e].vert(1);
			m_verts[v1].add_edge(e);
		}
	}
	
	// Add more mesh pieces to an already existing mesh (although the existing mesh could
	// empty). The incoming mesh edges point to vertices (and vice versa) from 0 to ne-1 locally. They need
	// to be offset by the edge/vertex size in the existing mesh.
	void insert_mesh(const Mesh2D& mesh)
	{
		unsigned edge_count = m_edges.size();
		unsigned vert_count = m_verts.size();

		m_verts.insert(m_verts.end(), mesh.m_verts.begin(), mesh.m_verts.end());
		m_edges.insert(m_edges.end(), mesh.m_edges.begin(), mesh.m_edges.end());

		// Update vertices to new edges
		for (unsigned v = vert_count; v < m_verts.size(); ++v)
		{
			for (unsigned e = 0; e < m_verts[v].valence(); ++e)
			{
				unsigned ei = m_verts[v].edge(e);
				m_verts[v].replace_edge(ei, ei + edge_count);
			}
		}
		for (unsigned e = edge_count; e < m_edges.size(); ++e)
		{
			unsigned v0 = m_edges[e].vert(0);
			m_edges[e].replace_vert(v0, v0 + vert_count);

			unsigned v1 = m_edges[e].vert(1);
			m_edges[e].replace_vert(v1, v1 + vert_count);
		}
	}

	const std::vector<Edge2D>& edges() const
	{
		return m_edges;
	}

	const Edge2D& edge(unsigned idx) const
	{
		return m_edges[idx];
	}

	const std::vector<Vertex2D>& verts() const
	{
		return m_verts;
	}

	Vertex2D vert(size_t idx) const
	{
		return m_verts[idx];
	}

	void set_vert(unsigned idx, const Vec2R& vert)
	{
		m_verts[idx].set_point(vert);
	}

	void clear()
	{
		m_verts.clear();
		m_edges.clear();
	}

	unsigned edge_size() const
	{
		return m_edges.size();
	}

	unsigned vert_size() const
	{
		return m_verts.size();
	}

	Vec2R unnormal(unsigned e) const
	{
		Edge2D edge = m_edges[e];
		Vec2R tan = m_verts[edge.vert(1)].point() - m_verts[edge.vert(0)].point();
		if (tan == Vec2R(0.))
			return Vec2R(0.); //Return nothing if degenerate edge

		return Vec2R(-tan[1], tan[0]);
	}

	Vec2R normal(unsigned e) const
	{
		return normalized(unnormal(e));
	}

	//Reverse winding order
	void reverse()
	{
		for (auto& e : m_edges) e.reverse();
	}

	void scale(Real s)
	{
		for (auto& v : m_verts) v *= s;
	}

	void translate(const Vec2R& t)
	{
		for (auto& v : m_verts) v += t;
	}

	// Test for degenerate edge (i.e. an edge with zero length)
	bool is_edge_degenerate(unsigned eidx) const
	{
		const auto& edge = m_edges[eidx];
		return m_verts[edge.vert(0)].point() == m_verts[edge.vert(1)].point();
	}

	bool unit_test() const;

	void draw_mesh(Renderer& renderer,
		Vec3f edge_colour = Vec3f(0),
		bool render_edge_normals = false,
		bool render_verts = false,
		Vec3f vert_colour = Vec3f(0));

	template<typename VelocityField>
	void advect(Real dt, const VelocityField& vel, const IntegrationOrder order);

private:

		std::vector<Edge2D> m_edges;
		std::vector<Vertex2D> m_verts;
};

template<typename VelocityField>
void Mesh2D::advect(Real dt, const VelocityField& vel, const IntegrationOrder order)
{
	for (auto& v : m_verts)
	{
		v.set_point(Integrator(dt, v.point(), vel, order));
	}
}
#pragma once

#include<memory>

#include "Core.h"
#include "Vec.h"

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

	inline Vec2R point() const
	{
		return m_point;
	}

	inline void set_point(const Vec2R& p)
	{
		m_point = p;
	}

	inline void operator=(const Vec2R& p)
	{
		m_point = p;
	}

	// Get edge stored at the eidx position in the m_edges list
	inline int edge(int eidx) const
	{
		assert(eidx < m_edges.size());
		return m_edges[eidx];
	}

	inline void add_edge(int eidx)
	{
		m_edges.push_back(eidx);
	}

	// Search through the edge list for a matching edge
	// index to old_eidx. If found, replace that edge
	// index with new_eidx.
	inline void replace_edge(int old_eidx, int new_eidx)
	{
		int fidx = m_edges.size();
		for (int e = 0; e < m_edges.size(); ++e)
		{
			if (m_edges[e] == old_eidx)
			{
				fidx = e;
				continue;
			}
		}

		assert(fidx != m_edges.size());

		m_edges[fidx] = new_eidx;
	}

	// Search through the edge list and return true if 
	// there is an edge index that matches eidx
	inline bool find_edge(int eidx) const
	{
		for (int e = 0; e < m_edges.size(); ++e)
			if (m_edges[e] == eidx) return true;

		return false;
	}

	inline size_t valence() const
	{
		return m_edges.size();
	}

	template<typename T>
	inline void operator*=(const T& s)
	{
		m_point *= s;
	}

	template<typename T>
	inline void operator+=(const T& s)
	{
		m_point += s;
	}

private:

	Vec2R m_point;
	std::vector<int> m_edges;
};

class Edge2D
{
public:
	// Negative magic numbers mean unassigned. There should never be a negative
	// vertex index or a negative material.
	Edge2D() : m_verts(Vec2i(-1))
	{}

	Edge2D(const Vec2i& verts) : m_verts(verts)
	{}

	Edge2D(const Vec2i& verts, const Vec2i& mat) : m_verts(verts)
	{}

	inline Vec2i verts() const
	{
		return m_verts;
	}

	inline int vert(int vidx) const
	{
		assert(vidx < 2);
		return m_verts[vidx];
	}

	// Given a vertex index, find the opposite vertex index
	// on the edge. An assert fail will be thrown if there
	// is no match as a debug warning.
	inline int adjacent_vert(int vidx) const
	{
		if (m_verts[0] == vidx)
			return m_verts[1];
		else if (m_verts[1] == vidx)
			return m_verts[0];
		else assert(false);
	}

	inline void replace_vert(int old_vidx, int new_vidx)
	{
		if (m_verts[0] == old_vidx)
			m_verts[0] = new_vidx;
		else if (m_verts[1] == old_vidx)
			m_verts[1] = new_vidx;
		else assert(false);
	}

	inline bool find_vert(int vidx) const
	{
		if (m_verts[0] == vidx || m_verts[1] == vidx) return true;
		return false;
	}

	inline void reverse()
	{
		std::swap(m_verts[0], m_verts[1]);
	}

private:
	// Each edge can be viewed as having a "tail" and a "head"
	// vertex. The orientation is used in reference to "left" and
	// "right" turns when talking about the materials on the edge.
	Vec2i m_verts;
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
	Mesh2D(const std::vector<Vec2i>& edges, const std::vector<Vec2R>& verts)
	{
		m_edges.reserve(edges.size());
		for (auto e : edges) m_edges.push_back(Edge2D(e));

		m_verts.reserve(verts.size());
		for (auto v : verts) m_verts.push_back(Vertex2D(v));

		// Update vertices to store adjacent edges in their edge lists
		for (size_t e = 0; e < m_edges.size(); ++e)
		{
			int v0 = m_edges[e].vert(0);
			assert(v0 >= 0);
			m_verts[v0].add_edge(e);

			int v1 = m_edges[e].vert(1);
			assert(v1 >= 0);
			m_verts[v1].add_edge(e);
		}
	}

	void initialize(const std::vector<Vec2i>& edges, const std::vector<Vec2R>& verts)
	{
		m_edges.clear();
		m_edges.reserve(edges.size());
		for (auto e : edges) m_edges.push_back(Edge2D(e));

		m_verts.clear();
		m_verts.reserve(verts.size());
		for (auto v : verts) m_verts.push_back(Vertex2D(v));

		// Update vertices to store adjacent edges in their edge lists
		for (size_t e = 0; e < m_edges.size(); ++e)
		{
			int v0 = m_edges[e].vert(0);
			assert(v0 >= 0);
			m_verts[v0].add_edge(e);

			int v1 = m_edges[e].vert(1);
			assert(v1 >= 0);
			m_verts[v1].add_edge(e);
		}
	}


	// Add more mesh pieces to an already existing mesh (although the existing mesh could
	// empty). The incoming mesh edges point to vertices (and vice versa) from 0 to ne-1 locally. They need
	// to be offset by the edge/vertex size in the existing mesh.
	void insert_mesh(const Mesh2D& mesh)
	{
		size_t ne = m_edges.size();
		size_t nv = m_verts.size();

		m_verts.insert(m_verts.end(), mesh.m_verts.begin(), mesh.m_verts.end());
		m_edges.insert(m_edges.end(), mesh.m_edges.begin(), mesh.m_edges.end());

		// Update vertices to new edges
		for (size_t v = nv; v < m_verts.size(); ++v)
		{
			for (size_t e = 0; e < m_verts[v].valence(); ++e)
			{
				size_t ei = m_verts[v].edge(e);
				m_verts[v].replace_edge(ei, ei + ne);
			}
		}
		for (size_t e = ne; e < m_edges.size(); ++e)
		{
			size_t v0 = m_edges[e].vert(0);
			m_edges[e].replace_vert(v0, v0 + nv);

			size_t v1 = m_edges[e].vert(1);
			m_edges[e].replace_vert(v1, v1 + nv);
		}
	}

	inline const std::vector<Edge2D>& edges() const
	{
		return m_edges;
	}

	inline Edge2D edge(size_t idx) const
	{
		return m_edges[idx];
	}

	inline const std::vector<Vertex2D>& verts() const
	{
		return m_verts;
	}

	inline Vertex2D vert(size_t idx) const
	{
		return m_verts[idx];
	}

	inline void set_vert(size_t idx, const Vec2R& vert)
	{
		m_verts[idx].set_point(vert);
	}

	inline void clear()
	{
		m_verts.clear();
		m_edges.clear();
	}

	inline size_t edge_size() const
	{
		return m_edges.size();
	}

	inline size_t vert_size() const
	{
		return m_verts.size();
	}

	inline Vec2R unnormal(size_t e) const
	{
		Edge2D edge = m_edges[e];
		Vec2R tan = m_verts[edge.vert(1)].point() - m_verts[edge.vert(0)].point();
		if (tan == Vec2R(0.))
		{
			std::cout << "Degenerate edge" << std::endl;
			return Vec2R(0.); //Return nothing if degernate edge
		}

		return Vec2R(-tan[1], tan[0]);
	}

	inline Vec2R normal(size_t e) const
	{
		return normalized(unnormal(e));
	}

	//Reverse winding order
	void reverse()
	{
		for (auto& e : m_edges) e.reverse();
	}

	void scale(const Real s)
	{
		for (auto& v : m_verts) v *= s;
	}

	void translate(const Vec2R& t)
	{
		for (auto& v : m_verts) v += t;
	}

	// Test for degenerate edge (i.e. an edge with zero length)
	bool is_edge_degenerate(size_t eidx) const
	{
		auto edge = m_edges[eidx];
		return m_verts[edge.vert(0)].point() == m_verts[edge.vert(1)].point();
	}

	bool unit_test() const;

	void draw_mesh(Renderer& renderer,
		Vec3f edge_colour = Vec3f(0),
		bool render_edge_normals = false,
		bool render_verts = false,
		Vec3f vert_colour = Vec3f(0));

	template<typename VelField, typename Integrator>
	void advect(Real dt, const VelField& vel, const Integrator& f);

private:

		std::vector<Edge2D> m_edges;
		std::vector<Vertex2D> m_verts;
};

template<typename VelField, typename Integrator>
void Mesh2D::advect(Real dt, const VelField& vel, const Integrator& f)
{
	for (auto& v : m_verts)
	{
		v.set_point(f(v.point(), dt, vel));
	}
}

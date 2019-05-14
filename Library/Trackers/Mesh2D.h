#ifndef LIBRARY_MESH2D_H
#define LIBRARY_MESH2D_H

#include "Common.h"
#include "Integrator.h"
#include "Renderer.h"
#include "VectorGrid.h"

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
	Vertex2D() : myPoint(Vec2R(0))
	{}

	Vertex2D(const Vec2R& point) : myPoint(point)
	{}

	const Vec2R& point() const
	{
		return myPoint;
	}

	void setPoint(const Vec2R& point)
	{
		myPoint = point;
	}

	void operator=(const Vec2R& point)
	{
		myPoint = point;
	}

	// Get edge stored at the eidx position in the mEdges list
	unsigned edge(unsigned index) const
	{
		assert(index < myEdges.size());
		return myEdges[index];
	}

	void addEdge(unsigned index)
	{
		myEdges.push_back(index);
	}

	// Search through the edge list for a matching edge
	// index to old_eidx. If found, replace that edge
	// index with new_eidx.
	bool replaceEdge(unsigned oldIndex, unsigned newIndex)
	{
		auto result = std::find(myEdges.begin(), myEdges.end(), oldIndex);

		if (result == myEdges.end()) return false;
		else
			*result = newIndex;

		return true;
	}

	// Search through the edge list and return true if 
	// there is an edge index that matches eidx
	bool findEdge(unsigned index) const
	{
		auto result = std::find(myEdges.begin(), myEdges.end(), index);
		return result != myEdges.end();
	}

	unsigned valence() const
	{
		return myEdges.size();
	}

	template<typename T>
	void operator*=(const T& s)
	{
		myPoint *= s;
	}

	template<typename T>
	void operator+=(const T& s)
	{
		myPoint += s;
	}

private:

	Vec2R myPoint;
	std::vector<unsigned> myEdges;
};

class Edge2D
{
public:
	// Negative magic numbers mean unassigned. There should never be a negative
	// vertex index or a negative material.
	//Edge2D() : mVerts(Vec2i(-1))
	//{}

	Edge2D(const Vec2ui& vertices) : myVerts(vertices)
	{}

	Vec2ui vertices() const
	{
		return myVerts;
	}

	unsigned vertex(unsigned index) const
	{
		assert(index < 2);
		return myVerts[index];
	}

	// Given a vertex index, find the opposite vertex index
	// on the edge. An assert fail will be thrown if there
	// is no match as a debug warning.
	unsigned adjacentVertex(unsigned index) const
	{
		if (myVerts[0] == index)
			return myVerts[1];
		else if (myVerts[1] == index)
			return myVerts[0];
		
		assert(false);
		return 0;
	}

	void replaceVertex(unsigned oldIndex, unsigned newIndex)
	{
		if (myVerts[0] == oldIndex)
			myVerts[0] = newIndex;
		else if (myVerts[1] == oldIndex)
			myVerts[1] = newIndex;
		else assert(false);
	}

	bool findVertex(unsigned index) const
	{
		if (myVerts[0] == index || myVerts[1] == index) return true;
		return false;
	}

	void reverse()
	{
		std::swap(myVerts[0], myVerts[1]);
	}

private:
	// Each edge can be viewed as having a "tail" and a "head"
	// vertex. The orientation is used in reference to "left" and
	// "right" turns when talking about the materials on the edge.
	Vec2ui myVerts;
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
	Mesh2D(const std::vector<Vec2ui>& edges, const std::vector<Vec2R>& vertices)
	{
		myEdges.reserve(edges.size());
		for (const auto& edge : edges) myEdges.push_back(Edge2D(edge));

		myVertices.reserve(vertices.size());
		for (const auto& vert : vertices) myVertices.push_back(Vertex2D(vert));

		// Update vertices to store adjacent edges in their edge lists
		for (unsigned edge = 0; edge < myEdges.size(); ++edge)
		{
			unsigned v0 = myEdges[edge].vertex(0);
			myVertices[v0].addEdge(edge);

			unsigned v1 = myEdges[edge].vertex(1);
			myVertices[v1].addEdge(edge);
		}
	}

	void reinitialize(const std::vector<Vec2ui>& edges, const std::vector<Vec2R>& verts)
	{
		myEdges.clear();
		myEdges.reserve(edges.size());
		for (const auto& edge : edges) myEdges.push_back(Edge2D(edge));

		myVertices.clear();
		myVertices.reserve(verts.size());
		for (const auto& vert : verts) myVertices.push_back(Vertex2D(vert));

		// Update vertices to store adjacent edges in their edge lists
		for (unsigned e = 0; e < myEdges.size(); ++e)
		{
			unsigned v0 = myEdges[e].vertex(0);
			myVertices[v0].addEdge(e);

			unsigned v1 = myEdges[e].vertex(1);
			myVertices[v1].addEdge(e);
		}
	}
	
	// Add more mesh pieces to an already existing mesh (although the existing mesh could
	// empty). The incoming mesh edges point to vertices (and vice versa) from 0 to ne-1 locally. They need
	// to be offset by the edge/vertex size in the existing mesh.
	void insertMesh(const Mesh2D& mesh)
	{
		unsigned edgeCount = myEdges.size();
		unsigned vertexCount = myVertices.size();

		myVertices.insert(myVertices.end(), mesh.myVertices.begin(), mesh.myVertices.end());
		myEdges.insert(myEdges.end(), mesh.myEdges.begin(), mesh.myEdges.end());

		// Update vertices to new edges
		for (unsigned vertexIndex = vertexCount; vertexIndex < myVertices.size(); ++vertexIndex)
		{
			for (unsigned neighbourEdgeIndex = 0; neighbourEdgeIndex < myVertices[vertexIndex].valence(); ++neighbourEdgeIndex)
			{
				unsigned edgeIndex = myVertices[vertexIndex].edge(neighbourEdgeIndex);
				myVertices[vertexIndex].replaceEdge(edgeIndex, edgeIndex + edgeCount);
			}
		}
		for (unsigned edgeIndex = edgeCount; edgeIndex < myEdges.size(); ++edgeIndex)
		{
			unsigned vertexIndex = myEdges[edgeIndex].vertex(0);
			myEdges[edgeIndex].replaceVertex(vertexIndex, vertexIndex + vertexCount);

			vertexIndex = myEdges[edgeIndex].vertex(1);
			myEdges[edgeIndex].replaceVertex(vertexIndex, vertexIndex + vertexCount);
		}
	}

	const std::vector<Edge2D>& edges() const
	{
		return myEdges;
	}

	const Edge2D& edge(unsigned idx) const
	{
		return myEdges[idx];
	}

	const std::vector<Vertex2D>& vertices() const
	{
		return myVertices;
	}

	Vertex2D vertex(unsigned idx) const
	{
		return myVertices[idx];
	}

	void setVertex(unsigned idx, const Vec2R& vert)
	{
		myVertices[idx].setPoint(vert);
	}

	void clear()
	{
		myVertices.clear();
		myEdges.clear();
	}

	unsigned edgeListSize() const
	{
		return myEdges.size();
	}

	unsigned vertexListSize() const
	{
		return myVertices.size();
	}

	Vec2R unnormal(unsigned e) const
	{
		Edge2D edge = myEdges[e];
		Vec2R tangent = myVertices[edge.vertex(1)].point() - myVertices[edge.vertex(0)].point();
		if (tangent == Vec2R(0.))
			return Vec2R(0.); //Return nothing if degenerate edge

		return Vec2R(-tangent[1], tangent[0]);
	}

	Vec2R normal(unsigned e) const
	{
		return normalize(unnormal(e));
	}

	//Reverse winding order
	void reverse()
	{
		for (auto& e : myEdges) e.reverse();
	}

	void scale(Real s)
	{
		for (auto& v : myVertices) v *= s;
	}

	void translate(const Vec2R& t)
	{
		for (auto& v : myVertices) v += t;
	}

	// Test for degenerate edge (i.e. an edge with zero length)
	bool isEdgeDegenerate(unsigned eidx) const
	{
		const auto& edge = myEdges[eidx];
		return myVertices[edge.vertex(0)].point() == myVertices[edge.vertex(1)].point();
	}

	bool unitTest() const;

	void drawMesh(Renderer& renderer,
		Vec3f edgeColour = Vec3f(0),
		Real edgeWidth = 1.,
		bool doRenderEdgeNormals = false,
		bool doRenderVerts = false,
		Vec3f vertColour = Vec3f(0));

	template<typename VelocityField>
	void advect(Real dt, const VelocityField& vel, const IntegrationOrder order);

private:

		std::vector<Edge2D> myEdges;
		std::vector<Vertex2D> myVertices;
};

template<typename VelocityField>
void Mesh2D::advect(Real dt, const VelocityField& vel, const IntegrationOrder order)
{
	for (auto& v : myVertices)
	{
		v.setPoint(Integrator(dt, v.point(), vel, order));
	}
}

#endif
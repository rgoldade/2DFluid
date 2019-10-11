#ifndef LIBRARY_EDGE_MESH_H
#define LIBRARY_EDGE_MESH_H

#include "Common.h"
#include "Integrator.h"
#include "Renderer.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// EdgeMesh.h/cpp
// Ryan Goldade 2016
//
// 2-d mesh container with edge and
// vertex accessors.
//
////////////////////////////////////

// Vertex class stores it's point, normal and adjacent edges
// as well as accessors and modifiers

class Vertex
{
public:
	Vertex() : myPoint(Vec2R(0))
	{}

	Vertex(const Vec2R& point) : myPoint(point)
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

	// Get edge stored at the index position in the edges list
	int edge(int index) const
	{
		return myEdges[index];
	}

	void addEdge(int index)
	{
		assert(index >= 0);
		myEdges.push_back(index);
	}

	// Search through the edge list for a matching edge
	// index to old_eidx. If found, replace that edge
	// index with new_eidx.
	bool replaceEdge(int oldIndex, int newIndex)
	{
		assert(oldIndex >= 0 && newIndex >= 0);
		auto result = std::find(myEdges.begin(), myEdges.end(), oldIndex);

		if (result == myEdges.end())
			return false;
		else
			*result = newIndex;

		return true;
	}

	// Search through the edge list and return true if 
	// there is an edge index that matches eidx
	bool findEdge(int index) const
	{
		auto result = std::find(myEdges.begin(), myEdges.end(), index);
		return result != myEdges.end();
	}

	int valence() const
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
	std::vector<int> myEdges;
};

class Edge
{
public:

	Edge(const Vec2i& vertices) : myVertices(vertices)
	{}

	Vec2i vertices() const
	{
		return myVertices;
	}

	int vertex(int index) const
	{
		return myVertices[index];
	}

	// Given a vertex index, find the opposite vertex index
	// on the edge. An assert fail will be thrown if there
	// is no match as a debug warning.
	int adjacentVertex(int index) const
	{
		if (myVertices[0] == index)
			return myVertices[1];
		else if (myVertices[1] == index)
			return myVertices[0];
		
		assert(false);
		return -1;
	}

	void replaceVertex(int oldIndex, int newIndex)
	{
		if (myVertices[0] == oldIndex)
			myVertices[0] = newIndex;
		else if (myVertices[1] == oldIndex)
			myVertices[1] = newIndex;
		else assert(false);
	}

	bool findVertex(int index) const
	{
		if (myVertices[0] == index || myVertices[1] == index) return true;
		return false;
	}

	void reverse()
	{
		std::swap(myVertices[0], myVertices[1]);
	}

private:
	// Each edge can be viewed as having a "tail" and a "head"
	// vertex. The orientation is used in reference to "left" and
	// "right" turns when talking about the materials on the edge.
	Vec2i myVertices;
};

class EdgeMesh
{
public:
	// Vanilla constructor leaves initialization up to the caller
	EdgeMesh()
	{}

	// Initialize mesh container with edges and the associated vertices.
	// Input mesh should be water-tight with no dangling edges
	// (i.e. no vertex has a valence less than 2).
	EdgeMesh(const std::vector<Vec2i>& edges, const std::vector<Vec2R>& vertices)
	{
		myEdges.reserve(edges.size());
		for (const auto& edge : edges) myEdges.push_back(Edge(edge));

		myVertices.reserve(vertices.size());
		for (const auto& vert : vertices) myVertices.push_back(Vertex(vert));

		// Update vertices to store adjacent edges in their edge lists
		for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
		{
			int vertIndex = myEdges[edgeIndex].vertex(0);
			myVertices[vertIndex].addEdge(edgeIndex);

			vertIndex = myEdges[edgeIndex].vertex(1);
			myVertices[vertIndex].addEdge(edgeIndex);
		}
	}

	void reinitialize(const std::vector<Vec2i>& edges, const std::vector<Vec2R>& vertices);
	
	// Add more mesh pieces to an already existing mesh (although the existing mesh could
	// empty). The incoming mesh edges point to vertices (and vice versa) from 0 to edge count - 1 locally. They need
	// to be offset by the edge/vertex size in the existing mesh.
	void insertMesh(const EdgeMesh& mesh);

	const std::vector<Edge>& edges() const
	{
		return myEdges;
	}

	const Edge& edge(unsigned idx) const
	{
		return myEdges[idx];
	}

	const std::vector<Vertex>& vertices() const
	{
		return myVertices;
	}

	Vertex vertex(int index) const
	{
		return myVertices[index];
	}

	void setVertex(int index, const Vec2R& vertex)
	{
		myVertices[index].setPoint(vertex);
	}

	void clear()
	{
		myVertices.clear();
		myEdges.clear();
	}

	int edgeListSize() const
	{
		return myEdges.size();
	}

	int vertexListSize() const
	{
		return myVertices.size();
	}

	Vec2R scaledNormal(int edgeIndex) const
	{
		Edge edge = myEdges[edgeIndex];
		Vec2R tangent = myVertices[edge.vertex(1)].point() - myVertices[edge.vertex(0)].point();
		if (tangent == Vec2R(0.))
			return Vec2R(0.); //Return nothing if degenerate edge

		return Vec2R(-tangent[1], tangent[0]);
	}

	Vec2R normal(int edgeIndex) const
	{
		return normalize(scaledNormal(edgeIndex));
	}

	//Reverse winding order
	void reverse()
	{
		for (auto& edge : myEdges) edge.reverse();
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
	bool isEdgeDegenerate(int edgeIndex) const
	{
		const auto& edge = myEdges[edgeIndex];
		return myVertices[edge.vertex(0)].point() == myVertices[edge.vertex(1)].point();
	}

	bool unitTestMesh() const;

	void drawMesh(Renderer& renderer,
		Vec3f edgeColour = Vec3f(0),
		Real edgeWidth = 1.,
		bool doRenderEdgeNormals = false,
		bool doRenderVertices = false,
		Vec3f vertexColour = Vec3f(0));

	template<typename VelocityField>
	void advect(Real dt, const VelocityField& velocity, const IntegrationOrder order);

private:

		std::vector<Edge> myEdges;
		std::vector<Vertex> myVertices;
};

template<typename VelocityField>
void EdgeMesh::advect(Real dt, const VelocityField& velocity, const IntegrationOrder order)
{
	for (auto& vertex : myVertices)
	{
		vertex.setPoint(Integrator(dt, vertex.point(), velocity, order));
	}
}

#endif
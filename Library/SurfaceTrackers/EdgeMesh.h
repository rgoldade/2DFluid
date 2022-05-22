#ifndef FLUIDSIM2D_EDGE_MESH_H
#define FLUIDSIM2D_EDGE_MESH_H

#include <algorithm>
#include <vector>

#include "Integrator.h"
#include "Renderer.h"
#include "Utilities.h"

///////////////////////////////////
//
// EdgeMesh.h/cpp
// Ryan Goldade 2016
//
// 2-d mesh container with edge and
// vertex accessors.
//
////////////////////////////////////

namespace FluidSim2D
{

class EdgeMesh
{

public:
	// Vanilla constructor leaves initialization up to the caller
	EdgeMesh()
	{}

	// Initialize mesh container with edges and the associated vertices.
	// Input mesh should be water-tight with no dangling edges
	// (i.e. no vertex has a valence less than 2).
	EdgeMesh(const VecVec2i& edges, const VecVec2d& vertices);

	void reinitialize(const VecVec2i& edges, const VecVec2d& vertices);

	// Add more mesh pieces to an already existing mesh (although the existing mesh could
	// empty). The incoming mesh edges point to vertices (and vice versa) from 0 to edge count - 1 locally. They need
	// to be offset by the edge/vertex size in the existing mesh.
	void insertMesh(const EdgeMesh& mesh);

	const VecVec2i& edges() const;
	const VecVec2d& vertices() const;

	FORCE_INLINE const Vec2i& edge(int index) const
	{
		return myEdges[index];
	}

	FORCE_INLINE const Vec2d& vertex(int index) const
	{
		return myVertices[index];
	}

	FORCE_INLINE void setVertex(int index, const Vec2d& vertex)
	{
		myVertices[index] = vertex;
	}

	void clear();

	FORCE_INLINE int edgeCount() const
	{
		return int(myEdges.size());
	}

	FORCE_INLINE int vertexCount() const
	{
		return int(myVertices.size());
	}

	FORCE_INLINE Vec2d scaledNormal(int edgeIndex) const
	{
		const Vec2i& edge = myEdges[edgeIndex];
		Vec2d tangent = myVertices[edge[1]] - myVertices[edge[0]];

		return Vec2d(-tangent[1], tangent[0]);
	}

	FORCE_INLINE Vec2d normal(int edgeIndex) const
	{
		Vec2d normal = scaledNormal(edgeIndex);
		double norm = normal.norm();

		if (norm > 0)
		{
			return normal / norm;
		}

		return Vec2d::Zero();
	}

	AlignedBox2d boundingBox() const;

	//Reverse winding order
	void reverse();

	void scale(double s);

	void translate(const Vec2d& t);

	// Test for degenerate edge (i.e. an edge with zero length)
	bool isEdgeDegenerate(int edgeIndex) const
	{
		const Vec2i& edge = myEdges[edgeIndex];
		return myVertices[edge[0]] == myVertices[edge[1]];
	}

	void drawMesh(Renderer& renderer,
					Vec3d edgeColour = Vec3d::Zero(),
					double edgeWidth = 1,
					bool doRenderEdgeNormals = false,
					bool doRenderVertices = false,
					Vec3d vertexColour = Vec3d::Zero()) const;

	template<typename VelocityField>
	void advectMesh(double dt, const VelocityField& velocity, const IntegrationOrder order);

	bool unitTestMesh() const;

private:

	void initialize(const VecVec2i& edges, const VecVec2d& vertices);

	VecVec2i myEdges;
	VecVec2d myVertices;
};

template<typename VelocityField>
void EdgeMesh::advectMesh(double dt, const VelocityField& velocity, const IntegrationOrder order)
{
	for (Vec2d& vertex : myVertices)
	{
		vertex = Integrator(dt, vertex, velocity, order);
	}
}

}

#endif
#include "EdgeMesh.h"

#include <iostream>
#include <string>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

namespace FluidSim2D
{

EdgeMesh::EdgeMesh(const VecVec2i& edges, const VecVec2d& vertices)
{
	initialize(edges, vertices);
}

void EdgeMesh::reinitialize(const VecVec2i& edges, const VecVec2d& vertices)
{
	myEdges.clear();
	myVertices.clear();

	initialize(edges, vertices);
}

void EdgeMesh::insertMesh(const EdgeMesh& mesh)
{
	int edgeCount = int(myEdges.size());
	int vertexCount = int(myVertices.size());

	myVertices.insert(myVertices.end(), mesh.myVertices.begin(), mesh.myVertices.end());
	myEdges.insert(myEdges.end(), mesh.myEdges.begin(), mesh.myEdges.end());

	for (int edgeIndex = edgeCount; edgeIndex < int(myEdges.size()); ++edgeIndex)
	{
		for (int localVertexIndex : {0, 1})
		{
			int vertexIndex = myEdges[edgeIndex][localVertexIndex];

			myEdges[edgeIndex][localVertexIndex] = vertexIndex + vertexCount;

			assert(vertexIndex >= 0 && vertexIndex < mesh.vertexCount());
			assert(myEdges[edgeIndex][localVertexIndex] < myVertices.size());
		}
	}
}

const VecVec2i& EdgeMesh::edges() const
{
	return myEdges;
}

const VecVec2d& EdgeMesh::vertices() const
{
	return myVertices;
}

AlignedBox2d EdgeMesh::boundingBox() const
{
	AlignedBox2d bbox;

	for (const Vec2d& vertex : myVertices)
		bbox.extend(vertex);

	return bbox;
}

void EdgeMesh::clear()
{
	myVertices.clear();
	myEdges.clear();
}

void EdgeMesh::reverse()
{
	for (auto& edge : myEdges)
	{
		std::swap(edge[0], edge[1]);
	}
}

void EdgeMesh::scale(double s)
{
	for (auto& vertex : myVertices)
	{
		vertex.array() *= s;
	}
}

void EdgeMesh::translate(const Vec2d& t)
{
	for (auto& vertex : myVertices)
	{
		vertex += t;
	}
}

void EdgeMesh::drawMesh(const std::string& label,
							Vec3d edgeColour,
							double edgeWidth,
							bool doRenderEdgeNormals,
							bool doRenderVertices,
							Vec3d vertexColour) const
{
	auto* cn = polyscope::registerCurveNetwork2D(label + " surface", myVertices, myEdges);
	cn->setColor(glm::vec3((float)edgeColour[0], (float)edgeColour[1], (float)edgeColour[2]));
	cn->setRadius(edgeWidth);

	if (doRenderEdgeNormals)
	{
		VecVec2d normalNodes;
		normalNodes.reserve(myEdges.size() * 2);

		double averageLength = 0.;
		for (const Vec2i& edge : myEdges)
			averageLength += (myVertices[edge[0]] - myVertices[edge[1]]).norm();
		averageLength /= double(myEdges.size());

		for (int edgeIndex = 0; edgeIndex < int(myEdges.size()); ++edgeIndex)
		{
			Vec2d midPoint = .5 * (myVertices[myEdges[edgeIndex][0]] + myVertices[myEdges[edgeIndex][1]]);
			Vec2d endPoint = midPoint + normal(edgeIndex) * averageLength;

			normalNodes.push_back(midPoint);
			normalNodes.push_back(endPoint);
		}

		polyscope::registerCurveNetworkSegments2D(label + " surface normals", normalNodes);
	}

	if (doRenderVertices)
	{
		auto* pc = polyscope::registerPointCloud2D(label + " surface vertices", myVertices);
		pc->setPointColor(glm::vec3((float)vertexColour[0], (float)vertexColour[1], (float)vertexColour[2]));
	}
}

bool EdgeMesh::unitTestMesh() const
{
	std::vector<std::vector<int>> adjacentEdges(myVertices.size());

	for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
	{
		for (int localVertexIndex : {0, 1})
		{
			int vertexIndex = myEdges[edgeIndex][localVertexIndex];
			adjacentEdges[vertexIndex].push_back(edgeIndex);
		}
	}

	// Verify vertex has two or more adjacent edges. Meaning no dangling edge.
	for (int vertexIndex = 0; vertexIndex < myVertices.size(); ++vertexIndex)
	{
		if (adjacentEdges[vertexIndex].size() < 2)
		{
			std::cout << "Unit test failed in valence check. Vertex: " << vertexIndex << ". Valence: " << adjacentEdges[vertexIndex].size() << std::endl;
			return false;
		}
	}

	// Verify edge has two adjacent vertices. Meaning no dangling edge.
	for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
	{
		for (int localVertexIndex : {0, 1})
		{
			if (myEdges[edgeIndex][localVertexIndex] < 0 || myEdges[edgeIndex][localVertexIndex] >= myVertices.size())
			{
				std::cout << "Unit test failed in edge's vertex count test. Edge: " << edgeIndex << std::endl;
				return false;
			}
		}
	}

	// Verify vertex's adjacent edge reciprocates
	for (int vertexIndex = 0; vertexIndex < myVertices.size(); ++vertexIndex)
	{
		for (int edgeIndex : adjacentEdges[vertexIndex])
		{
			if (!(myEdges[edgeIndex][0] == vertexIndex || myEdges[edgeIndex][1] == vertexIndex))
			{
				std::cout << "Unit test failed in adjacent edge test. Vertex: " << vertexIndex << ". Edge: " << edgeIndex << std::endl;
				return false;
			}
		}
	}

	// Verify edge's adjacent vertex reciprocates
	for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
	{
		for (int localVertexIndex : {0, 1})
		{
			int vertexIndex = myEdges[edgeIndex][localVertexIndex];
			if (std::find(adjacentEdges[vertexIndex].begin(), adjacentEdges[vertexIndex].end(), edgeIndex) == adjacentEdges[vertexIndex].end())
			{
				std::cout << "Unit test failed in adjacent vertex test. Vertex: " << vertexIndex << ". Edge: " << edgeIndex << std::endl;
				return false;
			}
		}
	}

	// TODO: write a test that verifies winding order

	return true;
}

void EdgeMesh::initialize(const VecVec2i& edges, const VecVec2d& vertices)
{
	myEdges.insert(myEdges.end(), edges.begin(), edges.end());
	myVertices.insert(myVertices.end(), vertices.begin(), vertices.end());
}


}
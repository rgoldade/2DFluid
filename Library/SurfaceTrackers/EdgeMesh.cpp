#include "EdgeMesh.h"

#include <iostream>

namespace FluidSim2D::SurfaceTrackers
{

void EdgeMesh::initialize(const std::vector<Vec2i>& edges, const std::vector<Vec2f>& vertices)
{
	myEdges.reserve(edges.size());
	for (const auto& edge : edges) myEdges.emplace_back(edge);

	myVertices.reserve(vertices.size());
	for (const auto& vertex : vertices) myVertices.emplace_back(vertex);

	// Update vertices to store adjacent edges in their edge lists
	for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
	{
		int vertIndex = myEdges[edgeIndex].vertex(0);
		myVertices[vertIndex].addEdge(edgeIndex);

		vertIndex = myEdges[edgeIndex].vertex(1);
		myVertices[vertIndex].addEdge(edgeIndex);
	}
}

void EdgeMesh::insertMesh(const EdgeMesh& mesh)
{
	int edgeCount = myEdges.size();
	int vertexCount = myVertices.size();

	myVertices.insert(myVertices.end(), mesh.myVertices.begin(), mesh.myVertices.end());
	myEdges.insert(myEdges.end(), mesh.myEdges.begin(), mesh.myEdges.end());

	// Update vertices to new edges
	tbb::parallel_for(tbb::blocked_range<int>(vertexCount, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
			for (int neighbourEdgeIndex = 0; neighbourEdgeIndex < myVertices[vertexIndex].valence(); ++neighbourEdgeIndex)
			{
				int edgeIndex = myVertices[vertexIndex].edge(neighbourEdgeIndex);
				assert(edgeIndex >= 0 && edgeIndex < mesh.edgeCount());

				myVertices[vertexIndex].replaceEdge(edgeIndex, edgeIndex + edgeCount);
			}
	});

	tbb::parallel_for(tbb::blocked_range<int>(edgeCount, myEdges.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int edgeIndex = range.begin(); edgeIndex != range.end(); ++edgeIndex)
		{
			int vertexIndex = myEdges[edgeIndex].vertex(0);
			assert(vertexIndex >= 0 && vertexIndex < mesh.vertexCount());
			myEdges[edgeIndex].replaceVertex(vertexIndex, vertexIndex + vertexCount);

			vertexIndex = myEdges[edgeIndex].vertex(1);
			assert(vertexIndex >= 0 && vertexIndex < mesh.vertexCount());
			myEdges[edgeIndex].replaceVertex(vertexIndex, vertexIndex + vertexCount);
		}
	});
}

void EdgeMesh::drawMesh(Renderer& renderer,
						Vec3f edgeColour,
						float edgeWidth,
						bool doRenderEdgeNormals,
						bool doRenderVertices,
						Vec3f vertexColour) const
{
	std::vector<Vec2f> startPoints;
	std::vector<Vec2f> endPoints;

	for (const auto& edge : myEdges)
	{
		Vec2f start = myVertices[edge.vertex(0)].point();
		startPoints.push_back(start);

		Vec2f end = myVertices[edge.vertex(1)].point();
		endPoints.push_back(end);
	}

	renderer.addLines(startPoints, endPoints, edgeColour, edgeWidth);

	if (doRenderEdgeNormals)
	{
		std::vector<Vec2f> startNormals;
		std::vector<Vec2f> endNormals;

		// Scale by average edge length
		float averageLength = 0.;
		for (const auto& edge : myEdges)
			averageLength += mag(myVertices[edge.vertex(0)].point() - myVertices[edge.vertex(1)].point());

		averageLength /= float(myEdges.size());

		for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
		{
			Vec2f midPoint = .5 * (myVertices[myEdges[edgeIndex].vertex(0)].point() + myVertices[myEdges[edgeIndex].vertex(1)].point());
			Vec2f edgeNormal = normal(edgeIndex);

			startNormals.push_back(midPoint);
			endNormals.push_back(midPoint + edgeNormal * averageLength);
		}

		renderer.addLines(startNormals, endNormals, Vec3f(0.));
	}

	if (doRenderVertices)
	{
		std::vector<Vec2f> vertexPoints;

		for (const auto& vertex : myVertices) vertexPoints.push_back(vertex.point());

		renderer.addPoints(vertexPoints, vertexColour, 2);
	}
}

bool EdgeMesh::unitTestMesh() const
{
	// Verify vertex has two or more adjacent edges. Meaning no dangling edge.
	for (int vertexIndex = 0; vertexIndex < myVertices.size(); ++vertexIndex)
	{
		if (myVertices[vertexIndex].valence() < 2)
		{
			std::cout << "Unit test failed in valence check. Vertex: " << vertexIndex << ". Valence: " << myVertices[vertexIndex].valence() << std::endl;
			return false;
		}
	}

	// Verify edge has two adjacent vertices. Meaning no dangling edge.
	for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
	{
		if (myEdges[edgeIndex].vertex(0) < 0 || myEdges[edgeIndex].vertex(1) < 0)
		{
			std::cout << "Unit test failed in edge's vertex count test. Edge: " << edgeIndex << std::endl;
			return false;
		}
	}

	// Verify vertex's adjacent edge reciprocates
	for (int vertexIndex = 0; vertexIndex < myVertices.size(); ++vertexIndex)
	{
		for (int adjacentEdge = 0; adjacentEdge < myVertices[vertexIndex].valence(); ++adjacentEdge)
		{
			int edgeIndex = myVertices[vertexIndex].edge(adjacentEdge);
			if (!myEdges[edgeIndex].findVertex(vertexIndex))
			{
				std::cout << "Unit test failed in adjacent edge test. Vertex: " << vertexIndex << ". Edge: " << edgeIndex << std::endl;
				return false;
			}
		}
	}

	// Verify edge's adjacent vertex reciprocates
	for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
	{
		int vertexIndex = myEdges[edgeIndex].vertex(0);
		if (!myVertices[vertexIndex].findEdge(edgeIndex))
		{
			std::cout << "Unit test failed in adjacent vertex test. Vertex: " << vertexIndex << ". Edge: " << edgeIndex << std::endl;
			return false;
		}

		vertexIndex = myEdges[edgeIndex].vertex(1);
		if (!myVertices[vertexIndex].findEdge(edgeIndex))
		{
			std::cout << "Unit test failed in adjacent vertex test. Vertex: " << vertexIndex << ". Edge: " << edgeIndex << std::endl;
			return false;
		}
	}

	// TODO: write a test that verifies winding order

	return true;
}

}
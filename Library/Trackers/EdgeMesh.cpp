#include <iostream>

#include "EdgeMesh.h"

void EdgeMesh::drawMesh(Renderer& renderer,
	Vec3f edgeColour,
	Real edgeWidth,
	bool renderEdgeNormals,
	bool renderVertices,
	Vec3f vertexColour)
{
	std::vector<Vec2R> startPoints;
	std::vector<Vec2R> endPoints;

	for (const auto& edge : myEdges)
	{
		Vec2R start = myVertices[edge.vertex(0)].point();
		startPoints.push_back(start);

		Vec2R end = myVertices[edge.vertex(1)].point();
		endPoints.push_back(end);
	}

	renderer.addLines(startPoints, endPoints, edgeColour, edgeWidth);
	
	if (renderEdgeNormals)
	{
		std::vector<Vec2R> startNormals;
		std::vector<Vec2R> endNormals;

		// Scale by average edge length
		Real averageLength = 0.;
		for (const auto& edge : myEdges)
			averageLength += mag(myVertices[edge.vertex(0)].point() - myVertices[edge.vertex(1)].point());
	
		averageLength /= Real(myEdges.size());

		for (int edgeIndex = 0; edgeIndex < myEdges.size(); ++edgeIndex)
		{
			Vec2R midPoint = .5 * (myVertices[myEdges[edgeIndex].vertex(0)].point() + myVertices[myEdges[edgeIndex].vertex(1)].point());
			Vec2R edgeNormal = normal(edgeIndex);
			
			startNormals.push_back(midPoint);
			endNormals.push_back(midPoint + edgeNormal * averageLength);
		}

		renderer.addLines(startNormals, endNormals, Vec3f(0.));
	}

	if (renderVertices)
	{
		std::vector<Vec2R> vertexPoints;

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
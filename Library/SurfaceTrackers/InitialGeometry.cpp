#include "InitialGeometry.h"

namespace FluidSim2D::SurfaceTrackers
{
// This convention follows from normals are "left" turns
EdgeMesh makeCircleMesh(const Vec2f& center, float radius, float divisions)
{
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	float startAngle = 0;
	float endAngle = 2. * PI;

	float angleResolution = (endAngle - startAngle) / divisions;

	Vec2f startPoint(radius * cos(endAngle) + center[0],
		radius * sin(endAngle) + center[1]);
	verts.push_back(startPoint);

	// Loop in CW order, allowing for the "left" turn to be outside circle
	for (float theta = endAngle; theta > startAngle; theta -= angleResolution)
	{
		Vec2f nextPoint(radius * cos(theta - angleResolution) + center[0],
			radius * sin(theta - angleResolution) + center[1]);

		verts.push_back(nextPoint);

		unsigned vertIndex = verts.size();

		edges.emplace_back(vertIndex - 2, vertIndex - 1);
	}

	//Close mesh
	edges.emplace_back(verts.size() - 1, 0);

	return EdgeMesh(edges, verts);
}

EdgeMesh makeSquareMesh(const Vec2f& center, const Vec2f& scale)
{
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	verts.emplace_back(1.0 * scale[0], -1.0 * scale[1]);
	verts.emplace_back(-1.0 * scale[0], -1.0 * scale[1]);
	verts.emplace_back(-1.0 * scale[0], 1.0 * scale[1]);
	verts.emplace_back(1.0 * scale[0], 1.0 * scale[1]);

	edges.emplace_back(0, 1);
	edges.emplace_back(1, 2);
	edges.emplace_back(2, 3);
	edges.emplace_back(3, 0);

	EdgeMesh mesh = EdgeMesh(edges, verts);

	mesh.translate(center);

	return mesh;
}

EdgeMesh makeDiamondMesh(const Vec2f& center, const Vec2f& scale)
{
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	verts.emplace_back(1.0 * scale[0], 0);
	verts.emplace_back(0, -1.0 * scale[1]);
	verts.emplace_back(-1.0 * scale[0], 0);
	verts.emplace_back(0, 1.0 * scale[1]);

	edges.emplace_back(0, 1);
	edges.emplace_back(1, 2);
	edges.emplace_back(2, 3);
	edges.emplace_back(3, 0);

	EdgeMesh mesh = EdgeMesh(edges, verts);

	mesh.translate(center);

	return mesh;
}

// Initial conditions for testing level set methods.

EdgeMesh makeNotchedDiskMesh()
{
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	//Make circle portion
	float startAngle = -PI / 2. + acos(1. - sqr(2.5) / (2. * sqr(15.)));
	float endAngle = 3. * PI / 2. - acos(1. - sqr(2.5) / (2 * sqr(15.)));

	float angleResolution = (endAngle - startAngle) / 100.;

	Vec2f center(50, 75);
	float radius = 15;


	Vec2f startPoint(radius * cos(startAngle) + center[0],
		radius * sin(startAngle) + center[1]);

	verts.push_back(startPoint);

	for (float theta = startAngle; theta < endAngle; theta += angleResolution)
	{
		Vec2f nextPoint(radius * cos(theta + angleResolution) + center[0],
			radius * sin(theta + angleResolution) + center[1]);

		verts.push_back(nextPoint);

		unsigned vertIndex = verts.size();

		edges.emplace_back(vertIndex - 1, vertIndex - 2);

		startPoint = nextPoint;
	}

	//Make gap
	Vec2f gapPoint = startPoint + Vec2f(0.0, 25.0);

	verts.push_back(gapPoint);
	unsigned vertIndex = verts.size();
	edges.emplace_back(vertIndex - 1, vertIndex - 2);

	Vec2f nextGapPoint = gapPoint + Vec2f(5.0, 0.0);

	verts.push_back(nextGapPoint);
	vertIndex = verts.size();
	edges.emplace_back(vertIndex - 1, vertIndex - 2);

	//Close mesh
	edges.emplace_back(0, vertIndex - 1);

	return EdgeMesh(edges, verts);
}

EdgeMesh makeVortexMesh()
{
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	//Make circle
	float startAngle = 0;
	float endAngle = 2. * PI;

	float angleResolution = (endAngle - startAngle) / 100.0;

	Vec2f center(0.50, 0.75);
	float radius = 0.15;

	Vec2f startPoint(radius * cos(startAngle) + center[0],
		radius * sin(startAngle) + center[1]);

	verts.push_back(startPoint);

	for (float theta = startAngle; theta < endAngle; theta += angleResolution)
	{
		Vec2f nextPoint(radius * cos(theta + angleResolution) + center[0],
			radius * sin(theta + angleResolution) + center[1]);

		verts.push_back(nextPoint);

		unsigned vertIndex = verts.size();

		edges.emplace_back(vertIndex - 1, vertIndex - 2);

		startPoint = nextPoint;
	}

	//Close mesh
	edges.emplace_back(verts.size() - 1, 0);

	return EdgeMesh(edges, verts);
}

}
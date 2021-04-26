#include "InitialGeometry.h"

namespace FluidSim2D
{

// This convention follows from normals are "left" turns
EdgeMesh makeCircleMesh(const Vec2d& center, double radius, double divisions)
{
	VecVec2d verts;
	VecVec2i edges;

	double startAngle = 0;
	double endAngle = 2. * PI;

	double angleResolution = (endAngle - startAngle) / divisions;

	verts.emplace_back(radius * cos(endAngle) + center[0], radius * sin(endAngle) + center[1]);

	// Loop in CW order, allowing for the "left" turn to be outside circle
	for (double theta = endAngle - angleResolution; theta >= startAngle + angleResolution; theta -= angleResolution)
	{
		verts.emplace_back(radius * cos(theta) + center[0], radius * sin(theta) + center[1]);

		int vertIndex = int(verts.size());

		edges.emplace_back(vertIndex - 2, vertIndex - 1);
	}

	//Close mesh
	edges.emplace_back(verts.size() - 1, 0);

	return EdgeMesh(edges, verts);
}

EdgeMesh makeSquareMesh(const Vec2d& center, const Vec2d& scale)
{
	VecVec2d verts;
	VecVec2i edges;

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

EdgeMesh makeDiamondMesh(const Vec2d& center, const Vec2d& scale)
{
	VecVec2d verts;
	VecVec2i edges;

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
	VecVec2d verts;
	VecVec2i edges;

	//Make circle portion
	double startAngle = -PI / 2. + acos(1. - std::pow(2.5, 2) / (2. * std::pow(15., 2)));
	double endAngle = 3. * PI / 2. - acos(1. - std::pow(2.5, 2) / (2 * std::pow(15., 2)));

	double angleResolution = (endAngle - startAngle) / 100.;

	Vec2d center(50, 75);
	double radius = 15;

	Vec2d startPoint(radius * cos(startAngle) + center[0],
		radius * sin(startAngle) + center[1]);

	verts.push_back(startPoint);

	for (double theta = startAngle; theta < endAngle; theta += angleResolution)
	{
		Vec2d nextPoint(radius * cos(theta + angleResolution) + center[0],
			radius * sin(theta + angleResolution) + center[1]);

		verts.push_back(nextPoint);

		int vertIndex = int(verts.size());

		edges.emplace_back(vertIndex - 1, vertIndex - 2);

		startPoint = nextPoint;
	}

	//Make gap
	Vec2d gapPoint = startPoint + Vec2d(0, 25);

	verts.push_back(gapPoint);
	int vertIndex = int(verts.size());
	edges.emplace_back(vertIndex - 1, vertIndex - 2);

	Vec2d nextGapPoint = gapPoint + Vec2d(5, 0);

	verts.push_back(nextGapPoint);
	vertIndex = int(verts.size());
	edges.emplace_back(vertIndex - 1, vertIndex - 2);

	//Close mesh
	edges.emplace_back(0, vertIndex - 1);

	return EdgeMesh(edges, verts);
}

EdgeMesh makeVortexMesh()
{
	VecVec2d verts;
	VecVec2i edges;

	//Make circle
	double startAngle = 0;
	double endAngle = 2. * PI;

	double angleResolution = (endAngle - startAngle) / 100.0;

	Vec2d center(0.50, 0.75);
	double radius = 0.15;

	Vec2d startPoint(radius * cos(startAngle) + center[0],
		radius * sin(startAngle) + center[1]);

	verts.push_back(startPoint);

	for (double theta = startAngle; theta < endAngle; theta += angleResolution)
	{
		Vec2d nextPoint(radius * cos(theta + angleResolution) + center[0],
			radius * sin(theta + angleResolution) + center[1]);

		verts.push_back(nextPoint);

		int vertIndex = int(verts.size());

		edges.emplace_back(vertIndex - 1, vertIndex - 2);

		startPoint = nextPoint;
	}

	//Close mesh
	edges.emplace_back(verts.size() - 1, 0);

	return EdgeMesh(edges, verts);
}

}
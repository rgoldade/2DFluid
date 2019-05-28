#ifndef LIBRARY_INITIALCONDITIONS_H
#define LIBRARY_INITIALCONDITIONS_H

///////////////////////////////////
//
// InitialConditions.h
// Ryan Goldade 2017
//
// List of initial surface configurations
// to speed up scene creation.
//
////////////////////////////////////

#include "Common.h"
#include "EdgeMesh.h"

// This convention follows from normals are "left" turns
EdgeMesh circleMesh(const Vec2R& center = Vec2R(0), Real radius = 1., Real divisions = 10.)
{
	std::vector<Vec2R> verts;
	std::vector<Vec2i> edges;

	Real startAngle = 0;
	Real endAngle = 2. * Util::PI;

	Real angleResolution = (endAngle - startAngle) / divisions;

	Vec2R startPoint(radius * cos(endAngle) + center[0],
		radius * sin(endAngle) + center[1]);
	verts.push_back(startPoint);

	// Loop in CW order, allowing for the "left" turn to be outside circle
	for (Real theta = endAngle; theta > startAngle; theta -= angleResolution)
	{
		Vec2R nextPoint(radius * cos(theta - angleResolution) + center[0],
			radius * sin(theta - angleResolution) + center[1]);

		verts.push_back(nextPoint);

		unsigned vertIndex = verts.size();

		edges.push_back(Vec2i(vertIndex - 2, vertIndex - 1));
	}

	//Close mesh
	edges.push_back(Vec2i(verts.size() - 1, 0));

	return EdgeMesh(edges, verts);
}

EdgeMesh squareMesh(const Vec2R& center = Vec2R(0), const Vec2R& scale = Vec2R(1.))
{
	std::vector<Vec2R> verts;
	std::vector<Vec2i> edges;

	verts.push_back(Vec2R( 1.0 * scale[0], -1.0 * scale[1]));
	verts.push_back(Vec2R(-1.0 * scale[0], -1.0 * scale[1]));
	verts.push_back(Vec2R(-1.0 * scale[0],  1.0 * scale[1]));
	verts.push_back(Vec2R( 1.0 * scale[0],  1.0 * scale[1]));
	
	edges.push_back(Vec2i(0, 1));
	edges.push_back(Vec2i(1, 2));
	edges.push_back(Vec2i(2, 3));
	edges.push_back(Vec2i(3, 0));

	EdgeMesh mesh = EdgeMesh(edges, verts);

	mesh.translate(center);

	return mesh;
}

// Initial conditions for testing level set methods.

EdgeMesh notchedDiskMesh()
{
	std::vector<Vec2R> verts;
	std::vector<Vec2i> edges;

	//Make circle portion
	Real startAngle = -Util::PI / 2. + acos(1. - Util::sqr(2.5) / (2. * Util::sqr(15.)));
	Real endAngle = 3. * Util::PI / 2. - acos(1. - Util::sqr(2.5) / (2 * Util::sqr(15.)));

	Real angleResolution = (endAngle - startAngle) / 100.;

	Vec2R center(50, 75);
	Real radius = 15;


	Vec2R startPoint(radius * cos(startAngle) + center[0],
		radius * sin(startAngle) + center[1]);

	verts.push_back(startPoint);

	for (Real theta = startAngle; theta < endAngle; theta += angleResolution)
	{
		Vec2R nextPoint(radius * cos(theta + angleResolution) + center[0],
			radius * sin(theta + angleResolution) + center[1]);

		verts.push_back(nextPoint);

		unsigned vertIndex = verts.size();

		edges.push_back(Vec2i(vertIndex - 1, vertIndex - 2));

		startPoint = nextPoint;
	}

	//Make gap
	Vec2R gapPoint = startPoint + Vec2R(0.0, 25.0);

	verts.push_back(gapPoint);
	unsigned vertIndex = verts.size();
	edges.push_back(Vec2i(vertIndex - 1, vertIndex - 2));

	Vec2R nextGapPoint = gapPoint + Vec2R(5.0, 0.0);

	verts.push_back(nextGapPoint);
	vertIndex = verts.size();
	edges.push_back(Vec2i(vertIndex - 1, vertIndex - 2));

	//Close mesh
	edges.push_back(Vec2i(0, vertIndex - 1));

	return EdgeMesh(edges, verts);
}

EdgeMesh vortexMesh()
{
	std::vector<Vec2R> verts;
	std::vector<Vec2i> edges;

	//Make circle
	Real startAngle = 0;
	Real endAngle = 2. * Util::PI;

	Real angleResolution = (endAngle - startAngle) / 100.0;

	Vec2R center(0.50, 0.75);
	Real radius = 0.15;

	Vec2R startPoint(radius * cos(startAngle) + center[0],
		radius * sin(startAngle) + center[1]);

	verts.push_back(startPoint);

	for (Real theta = startAngle; theta < endAngle; theta += angleResolution)
	{
		Vec2R nextPoint(radius * cos(theta + angleResolution) + center[0],
			radius * sin(theta + angleResolution) + center[1]);

		verts.push_back(nextPoint);

		unsigned vertIndex = verts.size();

		edges.push_back(Vec2i(vertIndex - 1, vertIndex - 2));

		startPoint = nextPoint;
	}

	//Close mesh
	edges.push_back(Vec2i(verts.size() - 1, 0));

	return EdgeMesh(edges, verts);
}

#endif
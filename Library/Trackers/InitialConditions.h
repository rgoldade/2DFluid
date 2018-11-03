#pragma once

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
#include "Mesh2D.h"

// This convention follows from normals are "left" turns
Mesh2D circle_mesh(const Vec2R& center = Vec2R(0), Real radius = 1., Real divisions = 10.)
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;

	Real start_angle = 0;
	Real end_angle = 2. * M_PI;

	Real arc_res = (end_angle - start_angle) / divisions;

	Vec2R v0(radius * cos(end_angle) + center[0],
		radius * sin(end_angle) + center[1]);
	verts.push_back(v0);

	// Loop in CW order, allowing for the "left" turn to be outside circle
	for (Real theta = end_angle; theta > start_angle; theta -= arc_res)
	{
		Vec2R v1(radius * cos(theta - arc_res) + center[0],
			radius * sin(theta - arc_res) + center[1]);

		verts.push_back(v1);

		unsigned idx = verts.size();

		edges.push_back(Vec2ui(idx - 2, idx - 1));

		v0 = v1;
	}

	//Close mesh
	edges.push_back(Vec2ui(verts.size() - 1, 0));

	Mesh2D mesh = Mesh2D(edges, verts);

	return mesh;
}

Mesh2D square_mesh(const Vec2R& center = Vec2R(0), const Vec2R& scale = Vec2R(1.))
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;

	verts.push_back(Vec2R( 1.0 * scale[0], -1.0 * scale[1]));
	verts.push_back(Vec2R(-1.0 * scale[0], -1.0 * scale[1]));
	verts.push_back(Vec2R(-1.0 * scale[0],  1.0 * scale[1]));
	verts.push_back(Vec2R( 1.0 * scale[0],  1.0 * scale[1]));
	
	edges.push_back(Vec2ui(0, 1));
	edges.push_back(Vec2ui(1, 2));
	edges.push_back(Vec2ui(2, 3));
	edges.push_back(Vec2ui(3, 0));

	Mesh2D mesh = Mesh2D(edges, verts);

	mesh.translate(center);

	return mesh;
}

// Initial conditions for testing level set methods.

Mesh2D notched_disk_mesh()
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;

	//Make circle portion
	Real start_angle = -M_PI / 2.0 + acos(1 - 2.5*2.5 / (2 * 15 * 15));
	Real end_angle = 3.0*M_PI / 2.0 - acos(1 - 2.5*2.5 / (2 * 15 * 15));

	Real arc_res = (end_angle - start_angle) / 100.0;

	Vec2R center(50.0, 75.0);
	Real radius = 15.0;


	Vec2R v0(radius * cos(start_angle) + center[0],
		radius * sin(start_angle) + center[1]);

	verts.push_back(v0);

	for (Real theta = start_angle; theta < end_angle; theta += arc_res)
	{
		Vec2R v1(radius * cos(theta + arc_res) + center[0],
			radius * sin(theta + arc_res) + center[1]);

		verts.push_back(v1);

		unsigned idx = verts.size();

		edges.push_back(Vec2ui(idx - 1, idx - 2));

		v0 = v1;
	}

	//Make gap
	Vec2R v1 = v0 + Vec2R(0.0, 25.0);

	verts.push_back(v1);
	unsigned idx = verts.size();
	edges.push_back(Vec2ui(idx - 1, idx - 2));

	Vec2R v2 = v1 + Vec2R(5.0, 0.0);

	verts.push_back(v2);
	idx = verts.size();
	edges.push_back(Vec2ui(idx - 1, idx - 2));

	//Close mesh
	edges.push_back(Vec2ui(0, idx - 1));

	Mesh2D mesh = Mesh2D(edges, verts);

	return mesh;
}


Mesh2D vortex_mesh()
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;

	//Make circle
	Real start_angle = 0.0;
	Real end_angle = 2.0 * M_PI;

	Real arc_res = (end_angle - start_angle) / 100.0;

	Vec2R center(0.50, 0.75);
	Real radius = 0.15;

	Vec2R v0(radius * cos(start_angle) + center[0],
		radius * sin(start_angle) + center[1]);

	verts.push_back(v0);

	for (Real theta = start_angle; theta < end_angle; theta += arc_res)
	{
		Vec2R v1(radius * cos(theta + arc_res) + center[0],
			radius * sin(theta + arc_res) + center[1]);

		verts.push_back(v1);

		unsigned idx = verts.size();

		edges.push_back(Vec2ui(idx - 1, idx - 2));

		v0 = v1;
	}

	//Close mesh
	edges.push_back(Vec2ui(verts.size() - 1, 0));

	Mesh2D mesh = Mesh2D(edges, verts);

	return mesh;
}

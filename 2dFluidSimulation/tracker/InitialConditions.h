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

#include "core.h"
#include "vec.h"

#include "Mesh2d.h"

// This convention follows from normals are "left" turns
Mesh2D circle_mesh(const Vec2R& center = Vec2R(0), Real radius = 1., Real divisions = 10.)
{
	std::vector<Vec2R> verts;
	std::vector<Vec2i> edges;

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

		int idx = verts.size();

		edges.push_back(Vec2i(idx - 2, idx - 1));

		v0 = v1;
	}

	//Close mesh
	edges.push_back(Vec2i(verts.size() - 1, 0));

	Mesh2D mesh = Mesh2D(edges, verts);

	return mesh;
}

Mesh2D square_mesh(const Vec2R& center = Vec2R(0), const Vec2R& scale = Vec2R(1.))
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


	Mesh2D mesh = Mesh2D(edges, verts);

	mesh.translate(center);

	return mesh;
}
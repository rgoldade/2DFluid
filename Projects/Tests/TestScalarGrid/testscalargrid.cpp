#include "Common.h"

#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "LevelSet2D.h"

#include "Renderer.h"
#include "Transform.h"

Renderer* g_renderer;

bool g_do_scalar_test = false;
bool g_do_vector_test = false;
bool g_do_level_set_test = true;

int main(int argc, char** argv)
{
	g_renderer = new Renderer("Grid Test", Vec2ui(1000), Vec2R(-1), 12, &argc, argv);

	// Test scalar grid
	if (g_do_scalar_test)
	{
		Real dx = 2.5;
		Vec2R topright(20);
		Vec2ui size(topright / dx);
		Transform xform(dx, Vec2R(0));
		ScalarGrid<Real> testgrid(xform, size, ScalarGridSettings::SampleType::NODE, ScalarGridSettings::BorderType::CLAMP);

		for_each_voxel_range(Vec2ui(0), testgrid.size(), [&](const Vec2ui& cell)
		{
			Vec2R world_pos = testgrid.idx_to_ws(Vec2R(cell));
			testgrid(cell) = mag(world_pos);
		});

		testgrid.draw_grid(*g_renderer);
		testgrid.draw_sample_points(*g_renderer);
		testgrid.draw_supersampled_values(*g_renderer, .5, 5, 10);
		//testgrid.draw_sample_gradients(*g_renderer);
	}
	// Test vector grid
	if(g_do_vector_test)
	{
		Real dx = 2.5;
		Vec2R topright(20);
		Vec2ui size(topright / dx);
		Transform xform(dx, Vec2R(0));
		VectorGrid<Real> testvectorgrid(xform, size, VectorGridSettings::SampleType::STAGGERED, ScalarGridSettings::BorderType::CLAMP);

		for (unsigned axis = 0; axis < 2; ++axis)
		{
			for_each_voxel_range(Vec2ui(0), testvectorgrid.size(axis), [&](const Vec2ui& cell)
			{
				Vec2R world_pos = testvectorgrid.idx_to_ws(Vec2R(cell), axis);
				testvectorgrid(cell, axis) = mag(world_pos) + .1 * (Real)(rand() % 10 - 5);
			});
		}

		testvectorgrid.draw_grid(*g_renderer);
		testvectorgrid.draw_sample_points(*g_renderer);
		testvectorgrid.draw_supersampled_values(*g_renderer, .25, 20);
		testvectorgrid.draw_sample_point_vectors(*g_renderer);
		g_renderer->add_point(Vec2R(0, 0), Vec3f(0, 1, 1));
	}
	
	if (g_do_level_set_test)
	{
		Real dx = .25;
		Vec2R topright(10);
		Vec2ui size(topright / dx);
		Transform xform(dx, Vec2R(0));
		LevelSet2D testlevelset(xform, size, 5);

		for_each_voxel_range(Vec2ui(0), testlevelset.size(), [&](const Vec2ui& cell)
		{
			Vec2R wpos = testlevelset.idx_to_ws(Vec2R(cell));
			wpos -= Vec2R(topright/2);
			//TODO: turn this back to operator overload
			testlevelset(cell) = mag(wpos) - 3.;
		});
		
		testlevelset.reinit();
		testlevelset.draw_grid(*g_renderer);
		testlevelset.draw_supersampled_values(*g_renderer, .25, 5, 5);
		testlevelset.draw_normals(*g_renderer);


		testlevelset.draw_surface(*g_renderer, Vec3f(1,0,0));
	}


	g_renderer->run();
}

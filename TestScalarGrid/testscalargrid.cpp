#include <stdlib.h>

#include "core.h"
#include "vec.h"

#include "ScalarGrid.h"
#include "UniformGrid.h"
#include "VectorGrid.h"
#include "LevelSet2d.h"

#include "Renderer.h"
#include "Transform.h"

Renderer* g_renderer;

bool g_do_scalar_test = false;
bool g_do_vector_test = false;
bool g_do_level_set_test = true;
// TODO: test offset... didn't do any offsets....... ugh

int main(int argc, char** argv)
{
	g_renderer = new Renderer("Grid Test", Vec2i(1000), Vec2R(-1), 12, &argc, argv);

	// Test scalar grid
	if (g_do_scalar_test)
	{
		Real dx = 2.5;
		Vec2R topright(20);
		Vec2st size(topright / dx);
		Transform xform(dx, Vec2R(0));
		ScalarGrid<Real> testgrid(xform, size, ScalarGridSettings::NODE, ScalarGridSettings::CLAMP);

		for (size_t i = 0; i < testgrid.size()[0]; ++i)
			for (size_t j = 0; j < testgrid.size()[1]; ++j)
			{
				Vec2R wpos = testgrid.idx_to_ws(Vec2R(i, j));
				testgrid(i, j) = sqrt(wpos[0] * wpos[0] + wpos[1] * wpos[1]);
			}

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
		Vec2st size(topright / dx);
		Transform xform(dx, Vec2R(0));
		VectorGrid<Real> testvectorgrid(xform, size, VectorGridSettings::STAGGERED, ScalarGridSettings::CLAMP);

		for (size_t i = 0; i < testvectorgrid.size(0)[0]; ++i)
			for (size_t j = 0; j < testvectorgrid.size(0)[1]; ++j)
			{
				Vec2R wpos = testvectorgrid.idx_to_ws(Vec2R(i, j), 0);
				testvectorgrid(i, j, 0) = sqrt(wpos[0] * wpos[0] + wpos[1] * wpos[1]) + .1 * (Real) (rand() % 10 - 5);
			}

		for (size_t i = 0; i < testvectorgrid.size(1)[0]; ++i)
			for (size_t j = 0; j < testvectorgrid.size(1)[1]; ++j)
			{
				Vec2R wpos = testvectorgrid.idx_to_ws(Vec2R(i, j), 1);
				testvectorgrid(i, j, 1) = sqrt(wpos[0] * wpos[0] + wpos[1] * wpos[1]) + .1 * (Real)(rand() % 10 - 5);
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
		Vec2st size(topright / dx);
		Transform xform(dx, Vec2R(0));
		LevelSet2D testlevelset(xform, size, 5);

		for (size_t i = 0; i < testlevelset.size()[0]; ++i)
			for (size_t j = 0; j < testlevelset.size()[1]; ++j)
			{
				Vec2R wpos = testlevelset.idx_to_ws(Vec2R(i, j));
				wpos -= Vec2R(topright/2);
				//TODO: turn this back to operator overload
				testlevelset.set_phi(Vec2st(i,j), sqrt(wpos[0] * wpos[0] + wpos[1] * wpos[1]) - 3);
			}

		
		testlevelset.reinit();
		testlevelset.draw_grid(*g_renderer);
		testlevelset.draw_supersampled_values(*g_renderer, .25, 5, 5);
		testlevelset.draw_normals(*g_renderer);


		testlevelset.draw_surface(*g_renderer, Vec3f(1,0,0));
	}


	g_renderer->run();
}

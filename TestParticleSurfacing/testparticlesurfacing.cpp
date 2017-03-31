#include <memory>

#include "core.h"
#include "vec.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "MarkerParticles.h"
#include "FlipParticles.h"
#include "InitialConditions.h"
#include "Transform.h"

#include "CircularSim.h"

std::unique_ptr<Renderer> g_renderer;
MarkerParticles g_testparts;
LevelSet2D g_levelset;
Transform g_xform;

void keyboard(unsigned char key, int x, int y)
{
	if (key == 'n')
	{
		
		std::cout << "Reseeding" << std::endl;
		LevelSet2D rebuilt_surface;
		g_testparts.construct_surface(rebuilt_surface);
		g_testparts.reseed(rebuilt_surface);

		g_testparts.draw_points(*g_renderer.get(), Vec3f(1, 0, 0), 5);
		LevelSet2D rebuiltlevelset(g_levelset.xform(), g_levelset.size(), 5);
		g_testparts.construct_surface(rebuiltlevelset);


		g_renderer->clear();
		g_testparts.draw_points(*g_renderer.get(), Vec3f(1, 0, 0));
		rebuiltlevelset.draw_surface(*g_renderer.get(), Vec3f(1., 0., 0.));
	}
}

int main(int argc, char** argv)
{
	g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), Vec2R(-1), 1, &argc, argv));

	Mesh2D testmesh = circle_mesh();
	Mesh2D testmesh2 = circle_mesh(Vec2R(.5), 1., 10);
	Mesh2D testmesh3 = circle_mesh(Vec2R(.05), .5, 10);
	assert(testmesh.unit_test());
	assert(testmesh2.unit_test());
	testmesh.insert_mesh(testmesh2);
	testmesh.insert_mesh(testmesh3);

	g_xform = Transform(.05, Vec2R(0));
	g_levelset = LevelSet2D(g_xform, Vec2st(10), 5);
	g_levelset.init(testmesh);
	g_levelset.reinit();
	g_levelset.draw_grid(*g_renderer.get());
	g_levelset.draw_surface(*g_renderer.get(), Vec3f(0., 1.0, 1.));

	//levelset.draw_supersampled_values(*g_renderer.get(), .025, 5, 5);
	//levelset.draw_normals(*g_renderer.get());

	g_testparts = MarkerParticles(g_levelset.dx() * .5, 4, 2);
	g_testparts.init(g_levelset);



	CircularSim2D sim(Vec2R(0), 1);

	//testparts.set_velocity(sim);
	//testparts.draw_velocity(*g_renderer.get(), .05);

	LevelSet2D rebuilt_surface(g_levelset.xform(), g_levelset.size(), 5);
	g_testparts.construct_surface(rebuilt_surface);
	
	//
	//for (size_t i = 0; i < rebuilt_surface.size()[0]; ++i)
	//	for (size_t j = 0; j < rebuilt_surface.size()[1]; ++j)
	//	{
	//		Real phi = 0.;
	//		if (rebuilt_surface.phi(Vec2st(i, j)) > 0.) phi = 1.;
	//		else if (rebuilt_surface.phi(Vec2st(i, j)) < 0.) phi = -1.;

	//		rebuilt_surface.set_phi(Vec2st(i, j), phi);
	//	}

	//rebuilt_surface.draw_surface(*g_renderer.get(), Vec3f(0, 0, 1));
	rebuilt_surface.draw_supersampled_values(*g_renderer.get(), .5, 5, 3);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);

	g_renderer->run();
}


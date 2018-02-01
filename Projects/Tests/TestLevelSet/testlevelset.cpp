#include <memory>

#include "Core.h"
#include "Vec.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2D.h"
#include "InitialConditions.h"
#include "Transform.h"

std::unique_ptr<Renderer> g_renderer;

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
	//testmesh.draw_mesh(*g_renderer.get());

	Transform xform(.05, Vec2R(0));
	LevelSet2D levelset(xform, Vec2st(10), 5);
	levelset.init(testmesh);
	levelset.reinit();

	levelset.draw_surface(*g_renderer.get(), Vec3f(0., 1.0, 1.));
	levelset.draw_supersampled_values(*g_renderer.get(), .025, 5, 5);
	levelset.draw_normals(*g_renderer.get());

	g_renderer->run();
}

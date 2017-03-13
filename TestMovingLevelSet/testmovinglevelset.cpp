#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"
#include "Transform.h"

#include "CircularSim.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<LevelSet2D> g_surf;
std::unique_ptr<CircularSim2D> g_sim;

Real g_dt = 0.1;

Real g_dx = 0.05;
Vec2st g_size(150);

void display()
{
	typedef Integrator::forward_euler<Vec2R, CircularSim2D> integrator_functor;
	g_surf->backtrace_advect(g_dt, *g_sim.get(), integrator_functor());
	g_surf->reinit(true);
	
	g_renderer->clear();

	g_surf->draw_surface(*g_renderer.get(), Vec3f(1., 0., 0.));
	g_surf->draw_mesh_grid(*g_renderer.get());

	g_sim->draw_sim_vectors(*g_renderer.get(), .1, 1.);
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

	testmesh.reverse();
	Transform xform(g_dx, -Vec2R(g_dx * Vec2R(g_size) / 2.));
	g_surf = std::unique_ptr<LevelSet2D>(new LevelSet2D(xform, g_size, 5));
	g_surf->set_inverted();
	g_surf->init(testmesh, false);
	g_surf->reinit();

	g_surf->draw_surface(*g_renderer.get(), Vec3f(0., 1.0, 1.));
	g_surf->draw_mesh_grid(*g_renderer.get());

	// Set up simulation
	g_sim = std::unique_ptr<CircularSim2D>(new CircularSim2D(Vec2R(1., .5)));

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);
	g_renderer->run();
}
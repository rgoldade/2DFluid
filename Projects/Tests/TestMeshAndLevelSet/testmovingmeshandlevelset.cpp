#include <memory>

#include "Core.h"
#include "Vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2D.h"
#include "InitialConditions.h"
#include "Transform.h"

#include "Timer.h"

#include "TestVelocityFields.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<LevelSet2D> g_surf;
std::unique_ptr<SingleVortexSim2D> g_sim;

// Settings for single vortex
const static Real g_t_total = 12, g_dt = 0.01, g_t_reverse = 6, g_dx = .01;
const static Vec2st g_size(200);

static Real g_time = 0;
static int g_frame_count = 0;
static bool g_run = false;
static bool g_single_run = false;

static bool g_print_screen = false;

static Mesh2D *g_mesh;

void keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
		g_run = !g_run;
	else if (key == 'n')
		g_single_run = true;
	else if (key == 'p')
		g_print_screen = g_print_screen ? false : true;
}

void display()
{
	if (g_run || g_single_run)
	{
		g_renderer->clear();
		std::cout << "Time: " << g_time << std::endl;
		g_time += g_dt;

		////////////////////////////////////////////////////////////
		// SL level set advection
		////////////////////////////////////////////////////////////

		typedef Integrator::rk3<Vec2R, SingleVortexSim2D> integrator_functor;
		g_surf->backtrace_advect(g_dt, *g_sim.get(), integrator_functor());

		// Re-init level set	
		Timer reinitclock;
		g_surf->reinitFIM(true);

		std::cout << "FIM levelset reinit: " << reinitclock.stop() << "s" << std::endl;
		// Extract a mesh of the level set surface using DC
		g_surf->extract_dc_mesh(*g_mesh);
	
		//g_surf->draw_mesh_grid(*g_renderer.get());
		// Render out surface mesh
		g_mesh->draw_mesh(*g_renderer.get(), Vec3f(0, 0, 1));

		// For single vortex sim -> bump sim forward
		g_sim->advance(g_dt);
	}

	g_single_run = false;
}

int main(int argc, char** argv)
{
	g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), Vec2R(0), 1, &argc, argv));
	g_mesh = new Mesh2D();
		
	*g_mesh = vortex_mesh();
	
	Transform xform(g_dx, Vec2R(-.5));
	g_surf = std::unique_ptr<LevelSet2D>(new LevelSet2D(xform, g_size, 5));
	g_surf->init(*g_mesh, false);
	g_surf->reinit();

	g_mesh->draw_mesh(*g_renderer.get(), Vec3f(1,0,0), true);

	// Set up simulation
	g_sim = std::unique_ptr<SingleVortexSim2D>(new SingleVortexSim2D(0,g_t_reverse));

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);

	g_renderer->run();
}
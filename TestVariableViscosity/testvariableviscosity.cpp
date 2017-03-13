#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"

#include "EulerianFluid.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<EulerianFluid> g_sim;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;
Real g_dt = 1. / 30;

Real g_dx = 0.05;
Vec2st g_size(100);

Mesh2D g_static_solids;

void display()
{
	Transform xform(g_dx, Vec2R(0));
	if (g_run || g_single_run)
	{
		Real frame_time = 0.;
		while (frame_time < g_dt)
		{
			g_renderer->clear();

			// Set CFL condition
			Real velmag = g_sim->max_vel_mag();
			Real dt;
			if (velmag > 1E-6)
			{
				// CFL is allows the velocity to track 3 cells 
				dt = 3. * g_dx / velmag;
				if (dt > (g_dt - frame_time))
				{
					dt = g_dt - frame_time;
					std::cout << "Throttling timestep. CFL: " << dt << std::endl;
				}
			}
			else dt = g_dt - frame_time;
			// Store accumulated substep times
			frame_time += dt;

			// Safety checks
			if (dt <= 0.) break;

			g_sim->add_force(Vec2R(0., -9.8), dt);
			
			// Projection set unfortunately includes viscosity at the moment
			g_sim->run_simulation(dt, *g_renderer.get());
		}

		g_single_run = false;
		g_dirty_display = true;

	}
	if (g_dirty_display)
	{

		//g_sim->draw_grid(*g_renderer.get());
		g_sim->draw_surface(*g_renderer.get());
		g_sim->draw_collision(*g_renderer.get());
		g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt);
		g_dirty_display = false;
	}
}

void keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
		g_run = !g_run;
	else if (key == 'n')
		g_single_run = true;
}

int main(int argc, char** argv)
{
	// Scene settings
	Transform xform(g_dx, Vec2R(0));

	g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), xform.offset(), xform.dx() * (Real)(g_size[0]), &argc, argv));

	Vec2R center = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2);

	Mesh2D surface_mesh = square_mesh(center, Vec2R(1.5,1));
	assert(surface_mesh.unit_test());

	g_static_solids = square_mesh(center, Vec2R(2));
	g_static_solids.reverse();
	assert(g_static_solids.unit_test());

	LevelSet2D surface = LevelSet2D(xform, g_size, 10);
	surface.init(surface_mesh, false);

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(g_static_solids, false);

	// Set up simulation
	g_sim = std::unique_ptr<EulerianFluid>(new EulerianFluid(xform, g_size, 10));
	g_sim->set_surface_volume(surface);
	g_sim->set_collision_volume(solid);

	// Set variable viscosity
	ScalarGrid<Real> viscosity(xform, g_size);
	for (size_t x = 0; x < g_size[0]; ++x)
		for (size_t y = 0; y < g_size[1]; ++y)
		{
			if (surface(x, y) < g_dx)
			{
				Real ratio = (Real)x / (Real)g_size[0];
				viscosity(x, y) = 5. * sqr(sqr(ratio));
			}
		}
	g_sim->set_viscosity(viscosity);

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}
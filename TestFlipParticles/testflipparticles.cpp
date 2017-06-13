#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"

#include "FlipParticlesSimulation.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<FlipParticlesSimulation> g_sim;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;
Real g_dt = 1. / 24;
Real g_seed_time = 0;

Real g_dx = 0.025;
Vec2st g_size(200);
void display()
{
	if (g_run || g_single_run)
	{
		Real frame_time = 0.;
		while (frame_time < g_dt)
		{
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

			if (g_seed_time > 5.)
			{
				Transform xform(g_dx, Vec2R(0));
				Vec2R center = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2) + Vec2R(1.);
				Mesh2D seed_mesh = square_mesh(center, Vec2R(.5));

				LevelSet2D surface = LevelSet2D(xform, g_size, 5);
				surface.init(seed_mesh, false);
				g_sim->add_surface_volume(surface);
				g_seed_time = -100.;
			}

			g_sim->add_force(Vec2R(0., -9.8), dt);

			g_sim->run_simulation(dt, *g_renderer.get());

			g_seed_time += dt;
		}

		g_renderer->clear();
		g_single_run = false;
		g_dirty_display = true;

	}
	if (g_dirty_display)
	{
		//g_sim->draw_grid(*g_renderer.get());
		g_sim->draw_surface(*g_renderer.get());
		g_sim->draw_collision(*g_renderer.get());
		g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt);
		g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt, true);
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
	Mesh2D surface_mesh = square_mesh(center - Vec2R(0., 1.5), Vec2R(2, .5));
	Mesh2D surface_mesh2 = circle_mesh(center, .5, 20);
	surface_mesh.insert_mesh(surface_mesh2);

	assert(surface_mesh.unit_test());

	Mesh2D solid_mesh = square_mesh(center, Vec2R(2));
	solid_mesh.reverse();
	assert(solid_mesh.unit_test());

	LevelSet2D surface = LevelSet2D(xform, g_size, 10);
	surface.init(surface_mesh, false);

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(solid_mesh, false);

	// Set up simulation
	g_sim = std::unique_ptr<FlipParticlesSimulation>(new FlipParticlesSimulation(xform, g_size, 10));
	g_sim->set_surface_volume(surface);
	g_sim->set_collision_volume(solid);
	//g_sim->set_enforce_bubbles();

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}
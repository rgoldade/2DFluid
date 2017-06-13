#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"

#include "FlipParticlesSimulation.h"
//#include "MarkerParticlesSimulation.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<FlipParticlesSimulation> g_sim;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;
bool g_print_screen = false;
int g_frame_count = 0;
Real g_dt = 1. / 30.;

Real g_dx = 0.025;
Vec2st g_size(200);
void display()
{
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

			//g_sim->set_viscosity(.01);

			g_sim->add_force(Vec2R(0., -9.8), dt);

			std::cout << "Simulation loop. Timestep: " << dt << std::endl;
			g_sim->run_simulation(dt, *g_renderer.get());
		}
	
		g_single_run = false;
		g_dirty_display = true;

	}
	if (g_dirty_display)
	{
		//g_sim->draw_grid(*g_renderer.get());
		g_sim->draw_surface(*g_renderer.get());
		g_sim->draw_air(*g_renderer.get());
		g_sim->draw_collision(*g_renderer.get());
		//g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt, true);
		g_dirty_display = false;

		if (g_print_screen)
		{
			char * ppmfileformat;
			ppmfileformat = new char[strlen("d:/output/") + 50];

			sprintf_s(ppmfileformat, strlen("d:/output/") + 50, "%s/screenshot%%04d.sgi", "d:/output/");
			g_renderer->sgi_screenshot(ppmfileformat, g_frame_count);

			delete[] ppmfileformat;
		}
		++g_frame_count;
	}
}

void keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
		g_run = !g_run;
	else if (key == 'n')
		g_single_run = true;
	else if (key == 'p')
		g_print_screen = g_print_screen ? false : true;
}

int main(int argc, char** argv)
{
	// Scene settings
	Transform xform(g_dx, Vec2R(0));

	g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), xform.offset(), xform.dx() * (Real)(g_size[0]), &argc, argv));

	Vec2R center = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2);
	Vec2R centertop = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2) + Vec2R(xform.dx()) * Vec2R(0, g_size[1] / 4);
	Vec2R centerbottom = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2) - Vec2R(xform.dx()) * Vec2R(0, g_size[1] / 4);

	//
	// Construct water cooler collision
	// 

	// Top part of the watercooler
	Mesh2D watercooler_top = square_mesh(centertop, Vec2R(1.5, 1.1));
	
	// Bottom part of the watercooler
	Mesh2D watercooler_bottom = square_mesh(centerbottom, Vec2R(1.5, 1.1));
	
	// Pipe between bulbs
	Mesh2D pipe = square_mesh(center, Vec2R(xform.dx()) * Vec2R(g_size[0] * .075, g_size[1] * .1));

	watercooler_top.insert_mesh(watercooler_bottom);
	watercooler_top.insert_mesh(pipe);
	// Reverse mesh to make it an inverted volume
	watercooler_top.reverse();
	
	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(watercooler_top, false);

	solid.draw_surface(*g_renderer.get());

	//
	// Construct water cooler liquid 
	//
	Mesh2D fluid_top = square_mesh(centertop - Vec2R(0,.4), Vec2R(1.5, .7));

	//Mesh2D fluid_top = circle_mesh(centertop, 1., 50);

	//Mesh2D fluid_remover = square_mesh(centertop + Vec2R(xform.dx()) * Vec2R(0, g_size[1] / 8), Vec2R(1,.5));
	//fluid_remover.reverse();
	//fluid_top.insert_mesh(fluid_remover);


	LevelSet2D surface = LevelSet2D(xform, g_size, 10);
	surface.init(fluid_top, false);

	// Set up simulation
	g_sim = std::unique_ptr<FlipParticlesSimulation>(new FlipParticlesSimulation(xform, g_size, 10));
	g_sim->set_surface_volume(surface);
	g_sim->set_collision_volume(solid);
	//g_sim->set_air_volume();
	g_sim->set_enforce_bubbles();

	//g_sim->set_volume_correction();


	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}
#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"

#include "FlipParticlesSimulation.h"
#include "EulerianFluid.h"

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

			g_sim->add_force(Vec2R(0., -9.8), dt);
			
			g_sim->run_simulation(dt, *g_renderer.get());



				//g_sim->draw_surface(*g_renderer.get());
				//g_sim->draw_collision(*g_renderer.get());
				//g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt);
				//g_dirty_display = false;

				//// Print screenshot
				//if (g_print_screen)
				//{
				//	g_renderer->print_screen("test.png");

				//}
				//++g_frame_count;






		}
		
		g_single_run = false;
		g_dirty_display = true;

	}
	if (g_dirty_display)
	{
		//g_sim->draw_grid(*g_renderer.get());
		g_sim->draw_surface(*g_renderer.get());
		g_sim->draw_collision(*g_renderer.get());
		//g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt);
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
	
	//
	// Construct holy wall
	// 

	Mesh2D solid_mesh = square_mesh(center, Vec2R(2));
	Mesh2D top_wall = square_mesh(center + Vec2R(0,.5), Vec2R(.2,1.5));
	Mesh2D mid_wall = square_mesh(center - Vec2R(0,1.45), Vec2R(.2, .25));
	Mesh2D bottom_wall = square_mesh(center - Vec2R(0, 1.95), Vec2R(.2, .05));

	solid_mesh.reverse();
	solid_mesh.insert_mesh(top_wall);
	solid_mesh.insert_mesh(mid_wall);
	solid_mesh.insert_mesh(bottom_wall);
	
	solid_mesh.draw_mesh(*g_renderer.get());
	assert(solid_mesh.unit_test());

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(solid_mesh, false);

	solid.draw_surface(*g_renderer.get());
	solid.draw_normals(*g_renderer.get(), Vec3f(1,0,0), .05);
	//
	// Construct fluid side
	//
	Mesh2D waterside = square_mesh(center - Vec2R(1.1,.55), Vec2R(.9,1.45));

	LevelSet2D surface = LevelSet2D(xform, g_size, 10);
	surface.init(waterside, false);

	// Set up simulation
	g_sim = std::unique_ptr<FlipParticlesSimulation>(new FlipParticlesSimulation(xform, g_size, 10));
	g_sim->set_surface_volume(surface);
	g_sim->set_collision_volume(solid);
	g_sim->set_enforce_bubbles();
	//g_sim->set_volume_correction();

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}
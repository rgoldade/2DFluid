#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"
#include "Transform.h"
#include "MeshRebuilder.h"

#include "CircularSim.h"
#include "CurlNoiseSim.h"
#include "NotchedDiskSim.h"
#include "SingleVortex.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<LevelSet2D> g_surf;
std::unique_ptr<SingleVortexSim2D> g_sim;

//static Real g_dt = 0.05;
//static Real g_dx = 0.05;

//// Settings for notched disk
//static Real g_dt = 1.;
//static Real g_dx = 1.;

// Settings for single vortex
Real g_t_total = 12;
Real g_dt = 0.01;
Real g_t_reverse = 6;
Real g_dx = .01;

static Vec2st g_size(200);

static Real g_time = 0;
int g_frame_count = 0;
static bool g_run = false;
static bool g_single_run = false;

bool g_print_screen = false;

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

		//typedef Integrator::rk3<Vec2R, SingleVortexSim2D> integrator_functor;
		//g_surf->backtrace_advect(g_dt, *g_sim.get(), integrator_functor());

		//// Re-init level set
		//g_surf->reinit(1);

		//// Extract a mesh of the level set surface using DC
		//g_surf->extract_dc_mesh(*g_mesh);
	
		////g_surf->draw_mesh_grid(*g_renderer.get());
		//// Render out surface mesh
		//g_mesh->draw_mesh(*g_renderer.get(), Vec3f(0, 0, 1));

		////////////////////////////////////////////////////////////
		// MC contouring SL contouring
		////////////////////////////////////////////////////////////

		//// Extract a mesh of the level set surface using marching cubes
		//g_surf->extract_mesh(*g_mesh);

		//// Advect the mesh through a curl noise velocity field
		//typedef Integrator::rk3<Vec2R, SingleVortexSim2D> integrator_functor;
		//g_mesh->advect(g_dt, *g_sim.get(), integrator_functor());

		//// Re-init level set with updated surface mesh
		//g_surf->init(*g_mesh, false);

		//// Render out surface mesh
		//g_mesh->draw_mesh(*g_renderer.get(), Vec3f(0, 1, 0));

		//////////////////////////////////////////////////////////////
		//// Dual contouring SL contouring
		//////////////////////////////////////////////////////////////

		//////// Extract a mesh of the level set surface using DC
		//g_surf->extract_dc_mesh(*g_mesh);

		////// Advect the mesh through a curl noise velocity field
		//typedef Integrator::rk3<Vec2R, SingleVortexSim2D> integrator_functor;
		//g_mesh->advect(g_dt, *g_sim.get(), integrator_functor());

		//// Re-init level set with updated surface mesh
		//g_surf->init(*g_mesh, false);

		////g_surf->draw_mesh_grid(*g_renderer.get());
		//// Render out surface mesh
		//g_mesh->draw_mesh(*g_renderer.get(), Vec3f(1, 0, 0));

		//////////////////////////////////////////////////////////////
		//// Pure Muller + MS
		//////////////////////////////////////////////////////////////
		
		//// Advect the mesh through a curl noise velocity field
		//typedef Integrator::rk3<Vec2R, SingleVortexSim2D> integrator_functor;
		//g_mesh->advect(g_dt, *g_sim.get(), integrator_functor());

		//// Rebuild using the DC+Muller approach
		//MeshRebuilder rebuilder(g_dx);
		//rebuilder.rebuild(*g_mesh, false);

		//// Render out surface mesh
		//g_mesh->draw_mesh(*g_renderer.get(), Vec3f(.5, .5, .5), false, true, Vec3f(1., 0., 0.));

		//////////////////////////////////////////////////////////////
		//// Pure Muller + DCs
		//////////////////////////////////////////////////////////////

		// Advect the mesh through a curl noise velocity field
		typedef Integrator::rk3<Vec2R, SingleVortexSim2D> integrator_functor;
		g_mesh->advect(g_dt, *g_sim.get(), integrator_functor());

		// Rebuild using the DC+Muller approach
		MeshRebuilder rebuilder(g_dx);
		rebuilder.rebuild(*g_mesh, true);

		//rebuilder.draw_intersections(*g_renderer.get());

		// Render out surface mesh
		g_mesh->draw_mesh(*g_renderer.get(), Vec3f(.5, .5, .5), false, true, Vec3f(1., 0., 0.));

		// For single vortex sim -> bump sim forward
		g_sim->advance(g_dt);

		// Print screenshot
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

	g_single_run = false;
}

int main(int argc, char** argv)
{
	//g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), /*Vec2R(50.0, 75.0)*/ - Vec2R(g_dx * Vec2R(g_size) / 2.), 100, &argc, argv));
	g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), Vec2R(0), 1, &argc, argv));
	g_mesh = new Mesh2D();
		
	//*g_mesh = circle_mesh(Vec2R(0,1),1,80);
	//*g_mesh = notched_disk_mesh();
	*g_mesh = vortex_mesh();
	
	//Transform xform(g_dx, Vec2R(-.5));
	//g_surf = std::unique_ptr<LevelSet2D>(new LevelSet2D(xform, g_size, 5));
	//g_surf->init(*g_mesh, false);
	//g_surf->reinit();

	g_mesh->draw_mesh(*g_renderer.get(), Vec3f(1,0,0), true);

	// Set up simulation
	g_sim = std::unique_ptr<SingleVortexSim2D>(new SingleVortexSim2D(0,g_t_reverse));

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);

	g_renderer->run();
}
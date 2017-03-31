#include <memory>

#include "core.h"
#include "vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2d.h"
#include "InitialConditions.h"

#include "MarkerParticlesSimulation.h"
#include "EulerianFluid.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<MarkerParticlesSimulation> g_sim;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;
bool g_print_screen = false;
int g_frame_count = 0;
Real g_dt = 1. / 30.;

Real g_dx = 0.025;
Vec2st g_size(200);

Mesh2D g_moving_solids;
Mesh2D g_static_solids;


struct LinearSim2D
{
	LinearSim2D(Real time, Real scale, Real period) : m_time(time), m_scale(scale), m_period(period) {}
	inline void advance(Real dt) { m_time += dt; }
	inline Vec2R operator()(Real dt, const Vec2R& pos) const
	{
		return Vec2R(0, m_scale * std::sin(2*M_PI * (m_time + dt) / m_period));
	}

	Real m_time, m_scale, m_period;
};

std::unique_ptr<LinearSim2D> g_solid_sim;

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
				dt = 2 * g_dx / velmag;
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
			
			//
			// Update moving collision velocity
			//

			// Need moving solid volume to build sampled velocity
			LevelSet2D moving_solid = LevelSet2D(xform, g_size, 10);
			moving_solid.init(g_moving_solids, false);

			VectorGrid<Real> solid_vel(xform, g_size, 0, VectorGridSettings::STAGGERED);
			// Simple solid velocity updater
			for (size_t x = 0; x < solid_vel.size(0)[0]; ++x)
				for (size_t y = 0; y < solid_vel.size(0)[1]; ++y)
				{
					Vec2R wpos = solid_vel.idx_to_ws(Vec2R(x, y), 0);
					if (moving_solid.interp(wpos) < xform.dx()) solid_vel(x, y, 0) = (*g_solid_sim.get())(0, wpos)[0];
				}

			for (size_t x = 0; x < solid_vel.size(1)[0]; ++x)
				for (size_t y = 0; y < solid_vel.size(1)[1]; ++y)
				{
					Vec2R wpos = solid_vel.idx_to_ws(Vec2R(x, y), 1);
					if (moving_solid.interp(wpos) < xform.dx()) solid_vel(x, y, 1) = (*g_solid_sim.get())(0, wpos)[1];
				}

			LevelSet2D solid = LevelSet2D(xform, g_size, 10);
			solid.set_inverted();
			solid.init(g_static_solids, false);
			solid.surface_union(moving_solid);

			g_sim->set_collision_volume(solid);
			g_sim->set_collision_velocity(solid_vel);

			//
			// Run simulation
			//

			g_sim->run_simulation(dt, *g_renderer.get());

			// Update moving solid
			typedef Integrator::forward_euler<Vec2R, LinearSim2D> integrator_functor;
			g_moving_solids.advect(dt, *g_solid_sim.get(), integrator_functor());
			g_solid_sim->advance(dt);
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
		//g_sim->draw_collision_vel(*g_renderer.get(), 50 * g_dt);
		//g_sim->draw_velocity(*g_renderer.get(), 5 * g_dt);
		g_dirty_display = false;

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
	// Construct moving plunger
	//
		
	g_moving_solids = square_mesh(center - Vec2R(1.1, -.2), Vec2R(1, .2));
	g_solid_sim = std::unique_ptr<LinearSim2D>(new LinearSim2D(0, -1.2, 5));
	
	//
	// Construct middle wall
	// 
	
	g_static_solids = square_mesh(center, Vec2R(2));
	Mesh2D mid_wall = square_mesh(center - Vec2R(0,.2), Vec2R(.2, 1.1));

	g_static_solids.reverse();
	g_static_solids.insert_mesh(mid_wall);

	g_static_solids.draw_mesh(*g_renderer.get());
	assert(g_static_solids.unit_test());

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(g_static_solids, false);

	solid.draw_surface(*g_renderer.get());
	
	//
	// Construct fluid side
	//

	Mesh2D leftwaterside = square_mesh(center - Vec2R(1, 1.15), Vec2R(1, .85));
	Mesh2D rightwaterside = square_mesh(center - Vec2R(-1, .55), Vec2R(1, 1.45));
	mid_wall.reverse();

	leftwaterside.insert_mesh(rightwaterside);
	leftwaterside.insert_mesh(mid_wall); 

	LevelSet2D surface = LevelSet2D(xform, g_size, 10);
	surface.init(leftwaterside, false);

	//
	// Set up simulation
	//

	g_sim = std::unique_ptr<MarkerParticlesSimulation>(new MarkerParticlesSimulation(xform, g_size, 10));
	g_sim->set_surface_volume(surface);
	g_sim->set_collision_volume(solid);
	g_sim->set_air_volume();
	g_sim->set_enforce_bubbles();

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}
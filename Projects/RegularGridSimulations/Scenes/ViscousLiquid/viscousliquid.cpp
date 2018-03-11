#include <memory>

#include "Core.h"
#include "Vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2D.h"
#include "InitialConditions.h"
#include "TestVelocityFields.h"

#include "EulerianLiquid.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<EulerianLiquid> g_sim;
std::unique_ptr<CircularSim2D> g_solid_sim;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;
Real g_dt = 1. / 30;
Real g_seed_time = 0;

Real g_dx = 0.025;
Vec2st g_size(200);

Vec2R g_seed_center;
Mesh2D g_seed_mesh;
LevelSet2D g_seed_surface;

Mesh2D g_moving_solids;
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
				dt = 2. * g_dx / velmag;
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

			// Seed every frame
			g_sim->add_force(Vec2R(0., -9.8), dt);

			// Update moving solid
			typedef Integrator::forward_euler<Vec2R, CircularSim2D> integrator_functor;
			g_moving_solids.advect(dt, *g_solid_sim.get(), integrator_functor());

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


			g_sim->set_viscosity(30.);
			// Projection set unfortunately includes viscosity at the moment
			g_sim->run_simulation(dt, *g_renderer.get());

			g_sim->add_surface_volume(g_seed_surface);
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
	g_seed_center = center;

	g_moving_solids = circle_mesh(center + Vec2R(1.2, 0), .25, 20);

	g_static_solids = square_mesh(center, Vec2R(2));
	g_static_solids.reverse();
	assert(g_static_solids.unit_test());
	
	Mesh2D dropping_beam_mesh = square_mesh(center - Vec2R(.8, 0), Vec2R(1.5, .2));
	assert(dropping_beam_mesh.unit_test());

	LevelSet2D dropping_beam_surface = LevelSet2D(xform, g_size, 10);
	dropping_beam_surface.init(dropping_beam_mesh, false);

	g_seed_mesh = square_mesh(center + Vec2R(0, .6), Vec2R(.075, .25));
	assert(g_seed_mesh.unit_test());

	g_seed_surface = LevelSet2D(xform, g_size, 10);
	g_seed_surface.init(g_seed_mesh, false);

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(g_static_solids, false);

	// Set up simulation
	g_sim = std::unique_ptr<EulerianLiquid>(new EulerianLiquid(xform, g_size, 10));
	g_sim->set_surface_volume(dropping_beam_surface);
	g_sim->add_surface_volume(g_seed_surface);
	g_sim->set_collision_volume(solid);

	// Set up simulation
	g_solid_sim = std::unique_ptr<CircularSim2D>(new CircularSim2D(center, .5));

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}

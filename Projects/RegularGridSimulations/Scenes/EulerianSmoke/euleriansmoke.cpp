#include <memory>

#include "Core.h"
#include "Vec.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2D.h"
#include "InitialConditions.h"

#include "ScalarGrid.h"

#include "EulerianSmoke.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<EulerianSmoke> g_sim;

ScalarGrid<Real> g_smokedensity;
ScalarGrid<Real> g_smoketemperature;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;
Real g_dt = 1./24;
Real g_seed_time = 0;

Real g_tambient = 300;

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
				dt = 5. * g_dx / velmag;
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

			// Add smoke density and temperature source to simulation frame
			g_sim->set_smoke_source(g_smokedensity, g_smoketemperature);
			
			g_sim->run_simulation(dt, *g_renderer.get());

			g_seed_time += dt;
		}

		g_renderer->clear();
		g_single_run = false;
		g_dirty_display = true;

	}
	if (g_dirty_display)
	{
		g_sim->draw_smoke(*g_renderer.get(), 1);
		g_sim->draw_collision(*g_renderer.get());
		//g_sim->draw_velocity(*g_renderer.get(), 5*g_dt);
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


void set_smoke_source(const LevelSet2D &source_volume,
						Real defaultdensity, ScalarGrid<Real> &smokedensity,
						Real defaulttemp, ScalarGrid<Real> &smoketemperature)
{
	Vec2st size = source_volume.size();

	int samples = 2;
	Real sample_dx = 1. / (Real)samples;

	Real dx = source_volume.dx();

	for (int i = 0; i < size[0]; ++i)
		for (int j = 0; j < size[1]; ++j)
		{
			// Super sample to determine 
			if (fabs(source_volume.interp(source_volume.idx_to_ws(Vec2R(i, j))) < dx * 2.))
			{
				// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
				// first sample point is the .5 * sample_dx closer to (i,j).
				int volcount = 0;
				for (Real x = ((Real)i - .5) + (.5 * sample_dx); x < (Real)i + .5; x += sample_dx)
					for (Real y = ((Real)j - .5) + (.5 * sample_dx); y < (Real)j + .5; y += sample_dx)
					{
						if (source_volume.interp(source_volume.idx_to_ws(Vec2R(i, j))) <= 0.) ++volcount;
					}

				smokedensity(i, j) = defaultdensity * (Real)volcount * sample_dx * sample_dx;
				smoketemperature(i, j) = defaulttemp * (Real)volcount * sample_dx * sample_dx;
			}
		}
}

int main(int argc, char** argv)
{
	// Scene settings
	Transform xform(g_dx, Vec2R(0));

	g_renderer = std::unique_ptr<Renderer>(new Renderer("Mesh test", Vec2i(1000), xform.offset(), xform.dx() * (Real)(g_size[0]), &argc, argv));

	Vec2R center = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2);
	
	Mesh2D solid_mesh = square_mesh(center, Vec2R(2.4));
	solid_mesh.reverse();
	assert(solid_mesh.unit_test());

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(solid_mesh, false);

	g_sim = std::unique_ptr<EulerianSmoke>(new EulerianSmoke(xform, g_size, 300));
	g_sim->set_collision_volume(solid);

	// Set up source for smoke density and smoke temperature
	Mesh2D source_mesh = circle_mesh(center - Vec2R(0, 1.5), .5, 20);
	LevelSet2D source_volume = LevelSet2D(xform, g_size, 10);
	source_volume.init(source_mesh, false);
	
	// Super sample source volume to get a smooth volumetric representation.
	g_smokedensity = ScalarGrid<Real>(xform, g_size, 0);
	g_smoketemperature = ScalarGrid<Real>(xform, g_size, g_tambient);

	set_smoke_source(source_volume, .1, g_smokedensity, 310, g_smoketemperature);

	g_sim->set_smoke_source(g_smokedensity, g_smoketemperature);

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);
	g_renderer->run();
}
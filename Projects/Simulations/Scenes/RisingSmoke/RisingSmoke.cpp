#include <memory>
#include <iostream>

#include "Common.h"
#include "Integrator.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2D.h"
#include "InitialConditions.h"

#include "ScalarGrid.h"

#include "EulerianSmoke.h"

std::unique_ptr<Renderer> g_renderer;
std::unique_ptr<EulerianSmoke> g_sim;

ScalarGrid<Real> g_smoke_density;
ScalarGrid<Real> g_smoke_temperature;

bool g_run = false;
bool g_single_run = false;
bool g_dirty_display = true;

const Real g_dt = 1./24;
const Real g_ambient = 300;
const Real g_dx = 0.025;
const Vec2ui g_size(200);

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
					dt = g_dt - frame_time;
			}
			else dt = g_dt - frame_time;
			// Store accumulated substep times
			frame_time += dt;

			// Safety checks
			if (dt <= 0.) break;

			g_sim->run_simulation(dt, *g_renderer);

			// Add smoke density and temperature source to simulation frame
			g_sim->set_smoke_source(g_smoke_density, g_smoke_temperature);
		}

		g_single_run = false;
		g_dirty_display = true;
	}
	if (g_dirty_display)
	{
		g_renderer->clear();

		g_sim->draw_smoke(*g_renderer, 1);
		g_sim->draw_collision(*g_renderer);

		g_dirty_display = false;

		glutPostRedisplay();
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
	Vec2ui size = source_volume.size();

	Real samples = 2;
	Real sample_dx = 1. / samples;

	Real dx = source_volume.dx();

	for_each_voxel_range(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		// Super sample to determine 
		if (fabs(source_volume.interp(source_volume.idx_to_ws(Vec2R(cell))) < dx * 2.))
		{
			// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
			// first sample point is the .5 * sample_dx closer to (i,j).
			int volcount = 0;
			for (Real x = (Real(cell[0]) - .5) + (.5 * sample_dx); x < Real(cell[0]) + .5; x += sample_dx)
				for (Real y = (Real(cell[1]) - .5) + (.5 * sample_dx); y < Real(cell[1]) + .5; y += sample_dx)
				{
					if (source_volume.interp(source_volume.idx_to_ws(Vec2R(cell))) <= 0.) ++volcount;
				}

			if (volcount > 0)
			{
				Real temp_density = defaultdensity +.05 * randhashd(cell[0] + cell[1] * source_volume.size()[0]);
				Real temp_temperature = defaulttemp + 50 * randhashd(cell[0] + cell[1] * source_volume.size()[0]);
				smokedensity(cell) = temp_density * Real(volcount) * sample_dx * sample_dx;
				smoketemperature(cell) = temp_temperature * Real(volcount) * sample_dx * sample_dx;
			}
		}
	});
}

int main(int argc, char** argv)
{
	// Scene settings
	Transform xform(g_dx, Vec2R(0));

	g_renderer = std::make_unique<Renderer>("Mesh test", Vec2ui(1000), xform.offset(), xform.dx() * Real(g_size[0]), &argc, argv);

	Vec2R center = xform.offset() + Vec2R(xform.dx()) * Vec2R(g_size / 2);
	
	Mesh2D solid_mesh = square_mesh(center, Vec2R(2.4));
	solid_mesh.reverse();
	assert(solid_mesh.unit_test());

	LevelSet2D solid = LevelSet2D(xform, g_size, 10);
	solid.set_inverted();
	solid.init(solid_mesh, false);

	g_sim = std::make_unique<EulerianSmoke>(xform, g_size, 300);
	g_sim->set_collision_volume(solid);

	// Set up source for smoke density and smoke temperature
	//Mesh2D source_mesh = circle_mesh(center - Vec2R(0, 1.5), .5, 20);
	Mesh2D source_mesh = circle_mesh(center - Vec2R(0, 2.), .25, 40);
	LevelSet2D source_volume = LevelSet2D(xform, g_size, 10);
	source_volume.init(source_mesh, false);
	
	// Super sample source volume to get a smooth volumetric representation.
	g_smoke_density = ScalarGrid<Real>(xform, g_size, 0);
	g_smoke_temperature = ScalarGrid<Real>(xform, g_size, g_ambient);

	set_smoke_source(source_volume, .2, g_smoke_density, 400, g_smoke_temperature);

	g_sim->set_smoke_source(g_smoke_density, g_smoke_temperature);

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);

	g_renderer->run();
}
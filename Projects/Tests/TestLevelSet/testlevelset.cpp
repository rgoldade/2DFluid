#include <memory>

#include "Common.h"

#include "Renderer.h"
#include "Mesh2D.h"
#include "LevelSet2D.h"
#include "TestVelocityFields.h"
#include "InitialConditions.h"
#include "Transform.h"

static std::unique_ptr<Renderer> g_renderer;
static std::unique_ptr<LevelSet2D> g_levelset;
static std::unique_ptr<Mesh2D> g_mesh;
static std::unique_ptr<CurlNoise2D> g_velocity;

static Real g_dt = 1./24.;

static bool g_run = false;
static bool g_single_run = false;

// TODO: advect level set through a velocity field....
void keyboard(unsigned char key, int x, int y)
{
	if (key = ' ')
		g_run = !g_run;
	else if (key == 'n')
		g_single_run = true;
}

void display()
{
	if (g_run || g_single_run)
	{
		g_renderer->clear();

		// Semi-Lagrangian level set advection

		g_levelset->advect(g_dt, *g_velocity, IntegrationOrder::FORWARDEULER);

		g_levelset->reinitFIM();

		g_levelset->extract_dc_mesh(*g_mesh);

		g_mesh->draw_mesh(*g_renderer, Vec3f(0, 0, 1));
	}

	g_single_run = false;

	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	g_renderer = std::make_unique<Renderer>("Mesh test", Vec2ui(1000), Vec2R(-1), 1, &argc, argv);

	g_velocity = std::make_unique<CurlNoise2D>();

	Mesh2D circle = circle_mesh();
	
	g_mesh = std::make_unique<Mesh2D>();
	g_mesh->insert_mesh(circle);

	Mesh2D test_mesh2 = circle_mesh(Vec2R(.5), 1., 10);
	Mesh2D test_mesh3 = circle_mesh(Vec2R(.05), .5, 10);
	assert(g_mesh->unit_test());
	assert(test_mesh2.unit_test());
	assert(test_mesh3.unit_test());
	g_mesh->insert_mesh(test_mesh2);
	g_mesh->insert_mesh(test_mesh3);

	Transform xform(.05, Vec2R(0));
	g_levelset = std::make_unique<LevelSet2D>(xform, Vec2ui(100), 5);
	g_levelset->init(*g_mesh, true);
	g_levelset->reinitFIM();

	g_levelset->draw_surface(*g_renderer, Vec3f(0., 1.0, 1.));

	std::function<void()> display_func = display;
	g_renderer->set_user_display(display_func);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	g_renderer->set_user_keyboard(keyboard);

	g_renderer->run();

	g_renderer->run();
}

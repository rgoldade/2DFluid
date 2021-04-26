#include <memory>

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim2D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<LevelSet> surface;
static std::unique_ptr<CurlNoiseField> velocityField;

static constexpr double dt = 1./24.;

static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

void keyboard(unsigned char key, int, int)
{
	if (key == ' ')
		runSimulation = !runSimulation;
	else if (key == 'n')
		runSingleTimestep = true;
}

void display()
{
	if (runSimulation || runSingleTimestep)
	{
		renderer->clear();

		EdgeMesh surfaceMesh = surface->buildDCMesh();

		surfaceMesh.advectMesh(dt, *velocityField, IntegrationOrder::RK3);
		surface->initFromMesh(surfaceMesh, true);

		isDisplayDirty = true;
	}

	runSingleTimestep = false;

	if (isDisplayDirty)
	{
		surface->drawGrid(*renderer, true);
		surface->drawDCSurface(*renderer, Vec3d(1., 0., 0.), 2.);
		surface->drawNormals(*renderer, Vec3d(.5, .5, .5), .1);

		glutPostRedisplay();
	}
}

int main(int argc, char** argv)
{
	EdgeMesh initialMesh = makeCircleMesh();
	
	EdgeMesh testMesh2 = makeCircleMesh(Vec2d(.5, .5), 1., 10.);
	EdgeMesh testMesh3 = makeCircleMesh(Vec2d(.05, .05), .5, 10.);
	
	assert(initialMesh.unitTestMesh());
	assert(testMesh2.unitTestMesh());
	assert(testMesh3.unitTestMesh());
	
	initialMesh.insertMesh(testMesh2);
	initialMesh.insertMesh(testMesh3);

	assert(initialMesh.unitTestMesh());

	AlignedBox2d bbox;

	for (int vertexIndex = 0; vertexIndex < initialMesh.vertexCount(); ++vertexIndex)
	{
		bbox.extend(initialMesh.vertex(vertexIndex));
	}

	Vec2d boundingBoxSize = bbox.max() - bbox.min();

	double dx = .05;

	Vec2d origin = bbox.min() - boundingBoxSize;
	Vec2i size = (3. * boundingBoxSize / dx).cast<int>();

	Transform xform(dx, origin);
	surface = std::make_unique<LevelSet>(xform, size, 5);
	surface->initFromMesh(initialMesh, true /* resize grid */);
	
	renderer = std::make_unique<Renderer>("Levelset Test", Vec2i(1000), origin, double(size[1]) * dx, &argc, argv);

	velocityField = std::make_unique<CurlNoiseField>();

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}

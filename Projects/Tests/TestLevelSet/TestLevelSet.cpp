#include <memory>

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::SurfaceTrackers;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<LevelSet> surface;
static std::unique_ptr<CurlNoiseField> velocityField;

static constexpr float dt = 1./24.;

static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

void keyboard(unsigned char key, int x, int y)
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
		surface->drawDCSurface(*renderer, Vec3f(1,0,0), 2);
		surface->drawNormals(*renderer, Vec3f(.5), .1);

		glutPostRedisplay();
	}
}

int main(int argc, char** argv)
{
	EdgeMesh initialMesh = makeCircleMesh();
	
	EdgeMesh testMesh2 = makeCircleMesh(Vec2f(.5), 1., 10);
	EdgeMesh testMesh3 = makeCircleMesh(Vec2f(.05), .5, 10);
	
	assert(initialMesh.unitTestMesh());
	assert(testMesh2.unitTestMesh());
	assert(testMesh3.unitTestMesh());
	
	initialMesh.insertMesh(testMesh2);
	initialMesh.insertMesh(testMesh3);

	assert(initialMesh.unitTestMesh());

	Vec2f minBoundingBox(std::numeric_limits<float>::max());
	Vec2f maxBoundingBox(std::numeric_limits<float>::lowest());

	for (int vertexIndex = 0; vertexIndex < initialMesh.vertexCount(); ++vertexIndex)
		updateMinAndMax(minBoundingBox, maxBoundingBox, initialMesh.vertex(vertexIndex).point());

	Vec2f boundingBoxSize(maxBoundingBox - minBoundingBox);

	Vec2f origin = minBoundingBox - boundingBoxSize;

	float dx = .05;
	Vec2i size = Vec2i(3. * boundingBoxSize / dx);

	Transform xform(dx, origin);
	surface = std::make_unique<LevelSet>(xform, size, 5);
	surface->initFromMesh(initialMesh, true /* resize grid */);
	
	renderer = std::make_unique<Renderer>("Levelset Test", Vec2i(1000), origin, float(size[1]) * dx, &argc, argv);

	velocityField = std::make_unique<CurlNoiseField>();

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}

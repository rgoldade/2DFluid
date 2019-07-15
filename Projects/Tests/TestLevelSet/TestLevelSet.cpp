#include <memory>

#include "Common.h"
#include "EdgeMesh.h"
#include "InitialConditions.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<LevelSet> surface;

static constexpr Real dt = 1./24.;

static bool runSimulation = false;
static bool runSingleStep = false;

void keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
		runSimulation = !runSimulation;
	else if (key == 'n')
		runSingleStep = true;
}

void display()
{
	if (runSimulation || runSingleStep)
	{
		renderer->clear();

		// Semi-Lagrangian level set advection
		CurlNoise2D velocityField;

		EdgeMesh surfaceMesh = surface->buildDCMesh();
		surfaceMesh.advect(dt, velocityField, IntegrationOrder::FORWARDEULER);
		
		surface->initFromMesh(surfaceMesh, false);

		surfaceMesh.drawMesh(*renderer, Vec3f(0, 0, 1), true);
		surface->drawGrid(*renderer);
	}

	runSingleStep = false;

	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	EdgeMesh initialMesh = circleMesh();
	
	EdgeMesh testMesh2 = circleMesh(Vec2R(.5), 1., 10);
	EdgeMesh testMesh3 = circleMesh(Vec2R(.05), .5, 10);
	
	assert(initialMesh.unitTest());
	assert(testMesh2.unitTest());
	assert(testMesh3.unitTest());
	
	initialMesh.insertMesh(testMesh2);
	initialMesh.insertMesh(testMesh3);

	assert(initialMesh.unitTest());

	Vec2R minBoundingBox(std::numeric_limits<Real>::max());
	Vec2R maxBoundingBox(std::numeric_limits<Real>::lowest());

	for (int vertexIndex = 0; vertexIndex < initialMesh.vertexListSize(); ++vertexIndex)
		updateMinAndMax(minBoundingBox, maxBoundingBox, initialMesh.vertex(vertexIndex).point());

	Vec2R boundingBoxSize(maxBoundingBox - minBoundingBox);

	Vec2R origin = minBoundingBox - boundingBoxSize;

	Real dx = .05;
	Vec2i size = Vec2i(3. * boundingBoxSize / dx);

	Transform xform(dx, origin);
	surface = std::make_unique<LevelSet>(xform, size, 5);
	surface->initFromMesh(initialMesh, false);
	surface->reinitFIM();
	
	renderer = std::make_unique<Renderer>("Levelset Test", Vec2i(1000), origin, Real(size[1]) * dx, &argc, argv);

	surface->drawSurface(*renderer, Vec3f(0., 1.0, 1.));

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	renderer->setUserKeyboard(keyboard);

	renderer->run();
}

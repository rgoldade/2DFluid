#include <memory>

#include "EdgeMesh.h"
#include "FluidParticles.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim2D;

static std::unique_ptr<FluidParticles> particles;
static std::unique_ptr<Renderer> renderer;

static bool isDisplayDirty = true;

static Transform xform;
static Vec2i gridSize;

void display()
{
	if (isDisplayDirty)
	{
		renderer->clear();

		LevelSet sphereUnionSurface = particles->surfaceParticles(xform, gridSize, 5);
		
		sphereUnionSurface.drawGrid(*renderer, true);
		sphereUnionSurface.drawSupersampledValues(*renderer, .5, 5, 3);
		sphereUnionSurface.drawDCSurface(*renderer, Vec3d(1., 0., 0.), 2.);
		sphereUnionSurface.drawNormals(*renderer, Vec3d(.5, .5, .5), .1);

		particles->drawPoints(*renderer, Vec3d(1., 0., 0.), 5.);

		glutPostRedisplay();
	}
}

void keyboard(unsigned char key, int, int)
{
	if (key == 'n')
	{
		LevelSet reseedSurface = particles->surfaceParticles(xform, gridSize, 5);
		particles->reseed(reseedSurface);
	}
}

int main(int argc, char** argv)
{
	EdgeMesh initialMesh = makeCircleMesh();
	
	EdgeMesh tempMesh = makeCircleMesh(Vec2d(.5, .5), 1., 10);
	assert(tempMesh.unitTestMesh());
	initialMesh.insertMesh(tempMesh);

	tempMesh = makeCircleMesh(Vec2d(.05, .05), .5, 10);
	assert(tempMesh.unitTestMesh());
	initialMesh.insertMesh(tempMesh);

	assert(initialMesh.unitTestMesh());

	double dx = .125;
	Vec2d topRightCorner(2.25, 2.25);
	Vec2d bottomLeftCorner(-1.5, -1.5);
	gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Particle Surfacing Test", Vec2i(1000, 1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	LevelSet initialSurface(xform, gridSize, 5);
	initialSurface.initFromMesh(initialMesh, false);

	particles = std::make_unique<FluidParticles>(dx * .75, 8, 2);
	particles->init(initialSurface);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
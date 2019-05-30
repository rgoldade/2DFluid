#include <memory>

#include "Common.h"
#include "EdgeMesh.h"
#include "FluidParticles.h"
#include "InitialConditions.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"
#include "Transform.h"

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<FluidParticles> particles;
static Transform xform;
static Vec2i gridSize;

void keyboard(unsigned char key, int x, int y)
{
	if (key == 'n')
	{	
		renderer->clear();

		LevelSet rebuiltSurface = particles->surfaceParticles(xform, gridSize, 5);

		particles->reseed(rebuiltSurface);

		rebuiltSurface = particles->surfaceParticles(xform, gridSize, 5);
		rebuiltSurface.drawGrid(*renderer);
		rebuiltSurface.drawSupersampledValues(*renderer, .5, 5, 3);
		rebuiltSurface.drawSurface(*renderer, Vec3f(1., 0., 0.));


		particles->drawPoints(*renderer, Vec3f(1, 0, 0), 5);
	}
}

int main(int argc, char** argv)
{
	EdgeMesh initialMesh = circleMesh();
	
	EdgeMesh tempMesh = circleMesh(Vec2R(.5), 1., 10);
	assert(tempMesh.unitTest());
	initialMesh.insertMesh(tempMesh);

	tempMesh = circleMesh(Vec2R(.05), .5, 10);
	assert(tempMesh.unitTest());
	initialMesh.insertMesh(tempMesh);

	assert(initialMesh.unitTest());

	Real dx = .125;
	Vec2R topRightCorner(2.25);
	Vec2R bottomLeftCorner(-1.5);
	gridSize = Vec2i((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Particle Surfacing Test", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	LevelSet initialSurface(xform, gridSize, 5);
	initialSurface.initFromMesh(initialMesh, false);
	initialSurface.reinit();
	initialSurface.drawGrid(*renderer);
	initialSurface.drawSurface(*renderer, Vec3f(0., 1.0, 1.));
	initialSurface.drawSupersampledValues(*renderer, .5, 5, 3);

	particles = std::make_unique<FluidParticles>(dx * .75, 8, 2);
	particles->init(initialSurface);

	//LevelSet2D rebuiltSurface = particles->surfaceParticles(xform, gridSize, 5);
	//rebuiltSurface.drawSupersampledValues(*renderer, .5, 5, 3);

	particles->drawPoints(*renderer, Vec3f(1, 0, 0), 4);

	std::function<void(unsigned char, int, int)> keyboard_func = keyboard;
	renderer->setUserKeyboard(keyboard);

	renderer->run();
}
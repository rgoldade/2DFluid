#include <memory>

#include "Common.h"
#include "EdgeMesh.h"
#include "EulerianLiquid.h"
#include "InitialConditions.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "TestVelocityFields.h"

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianLiquid> simulator;
static std::unique_ptr<CircularSim2D> solidSimulator;

static bool runSimulation = false;
static bool runSingleStep = false;
static bool isDisplayDirty = true;

static constexpr Real dt = 1. / 30;

static Transform xform;
static Vec2i gridSize;

static EdgeMesh seedLiquidMesh;
static EdgeMesh movingSolidsMesh;
static EdgeMesh staticSolidsMesh;

static LevelSet seedLiquidSurface;

static int frameCount = 0;

void display()
{
	if (runSimulation || runSingleStep)
	{
		Real frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			Real speed = simulator->maxVelocityMagnitude();

			Real localDt = dt - frameTime;
			assert(localDt >= 0);

			if (speed > 1E-6)
			{
				Real cflDt = 3. * xform.dx() / speed;
				if (localDt > cflDt)
				{
					localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			// Add gravity
			simulator->addForce(localDt, Vec2R(0., -9.8));

			// Update moving solid
			movingSolidsMesh.advect(localDt, *solidSimulator, IntegrationOrder::FORWARDEULER);

			// Need moving solid volume to build sampled velocity
			LevelSet movindSolidsSurface = LevelSet(xform, gridSize, 10);
			movindSolidsSurface.initFromMesh(movingSolidsMesh, false);

			VectorGrid<Real> movingSolidVelocity(xform, gridSize, 0, VectorGridSettings::SampleType::STAGGERED);
	
			// Set moving solid velocity
			for (int axis : {0, 1})
			{
				forEachVoxelRange(Vec2i(0), movingSolidVelocity.size(axis), [&](const Vec2i& face)
				{
					Vec2R worldPosition = movingSolidVelocity.indexToWorld(Vec2R(face), axis);
					
					if (movindSolidsSurface.interp(worldPosition) < xform.dx())
						movingSolidVelocity(face, axis) = (*solidSimulator)(0, worldPosition)[axis];
				});
			}

			LevelSet combinedSolidSurface = LevelSet(xform, gridSize, 10);
			combinedSolidSurface.setBoundaryNegative();
			combinedSolidSurface.initFromMesh(staticSolidsMesh, false);
			combinedSolidSurface.unionSurface(movindSolidsSurface);

			simulator->setSolidSurface(combinedSolidSurface);
			simulator->setSolidVelocity(movingSolidVelocity);

			simulator->setViscosity(30.);
			// Projection set unfortunately includes viscosity at the moment
			simulator->runTimestep(localDt, *renderer);

			simulator->unionLiquidSurface(seedLiquidSurface);

			frameTime += localDt;
		}

		runSingleStep = false;
		isDisplayDirty = true;

	}
	if (isDisplayDirty)
	{
		renderer->clear();

		simulator->drawLiquidSurface(*renderer);
		simulator->drawSolidSurface(*renderer);
		simulator->drawLiquidVelocity(*renderer, .5);

		isDisplayDirty = false;

		glutPostRedisplay();
	}
}

void keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
		runSimulation = !runSimulation;
	else if (key == 'n')
		runSingleStep = true;
}

int main(int argc, char** argv)
{
	// Scene settings
	Real dx = .025;
	Vec2R topRightCorner(2.5);
	Vec2R bottomLeftCorner(-2.5);
	gridSize = Vec2i((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Viscous Liquid Simulator", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	movingSolidsMesh = circleMesh(center + Vec2R(1.2, 0), .25, 20);
	
	staticSolidsMesh = squareMesh(center, Vec2R(2));
	staticSolidsMesh.reverse();
	assert(staticSolidsMesh.unitTest());
	
	EdgeMesh beamLiquidMesh = squareMesh(center - Vec2R(.8, 0), Vec2R(1.5, .2));
	assert(beamLiquidMesh.unitTest());

	LevelSet beamLiquidSurface = LevelSet(xform, gridSize, 10);
	beamLiquidSurface.initFromMesh(beamLiquidMesh, false);

	seedLiquidMesh = squareMesh(center + Vec2R(0, .6), Vec2R(.075, .25));
	assert(seedLiquidMesh.unitTest());

	seedLiquidSurface = LevelSet(xform, gridSize, 10);
	seedLiquidSurface.initFromMesh(seedLiquidMesh, false);

	LevelSet solidSurface(xform, gridSize, 10);
	solidSurface.setBoundaryNegative();
	solidSurface.initFromMesh(staticSolidsMesh, false);

	// Set up simulation
	simulator = std::unique_ptr<EulerianLiquid>(new EulerianLiquid(xform, gridSize, 10));
	simulator->setLiquidSurface(beamLiquidSurface);
	simulator->unionLiquidSurface(seedLiquidSurface);
	simulator->setSolidSurface(solidSurface);

	// Set up simulation
	solidSimulator = std::unique_ptr<CircularSim2D>(new CircularSim2D(center, .5));

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);
	renderer->run();
}

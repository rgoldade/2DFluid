#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "Transform.h"
#include "TestVelocityFields.h"
#include "Utilities.h"
#include "Vec.h"

using namespace FluidSim2D::RegularGridSim;
using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::Utilities;

static std::unique_ptr<Renderer> renderer;

static std::unique_ptr<EulerianLiquidSimulator> simulator;
static std::unique_ptr<CircularField> solidVelocityField;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr float dt = 1. / 30;

static constexpr float cfl = 5.;

static Transform xform;
static Vec2i gridSize;

static EdgeMesh movingSolidsMesh;
static EdgeMesh staticSolidsMesh;

static LevelSet seedLiquidSurface;

void display()
{
	if (runSimulation || runSingleTimestep)
	{
		float frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			float speed = simulator->maxVelocityMagnitude();

			float localDt = dt - frameTime;
			assert(localDt >= 0);

			if (speed > 1E-6)
			{
				float cflDt = cfl * xform.dx() / speed;
				if (localDt > cflDt)
				{
					localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			if (localDt <= 0)
				break;

			// Add gravity
			simulator->addForce(localDt, Vec2f(0., -9.8));

			// Update moving solid
			movingSolidsMesh.advectMesh(localDt, *solidVelocityField, IntegrationOrder::RK3);

			// Need moving solid volume to build sampled velocity
			LevelSet movingSolidSurface(xform, gridSize, 5);
			movingSolidSurface.initFromMesh(movingSolidsMesh, false);

			VectorGrid<float> movingSolidVelocity(xform, gridSize, 0, VectorGridSettings::SampleType::STAGGERED);
	
			// Set moving solid velocity
			for (int axis : {0, 1})
			{
				tbb::parallel_for(tbb::blocked_range<int>(0, movingSolidVelocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
				{
					for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
					{
						Vec2i face = movingSolidVelocity.grid(axis).unflatten(faceIndex);

						Vec2f worldPosition = movingSolidVelocity.indexToWorld(Vec2f(face), axis);

						if (movingSolidSurface.biLerp(worldPosition) < xform.dx())
							movingSolidVelocity(face, axis) = (*solidVelocityField)(0, worldPosition)[axis];
					}
				});
			}

			LevelSet combinedSolidSurface(xform, gridSize, 5);
			combinedSolidSurface.setBackgroundNegative();
			combinedSolidSurface.initFromMesh(staticSolidsMesh, false);
			combinedSolidSurface.unionSurface(movingSolidSurface);

			simulator->setSolidSurface(combinedSolidSurface);
			simulator->setSolidVelocity(movingSolidVelocity);

			// Projection set unfortunately includes viscosity at the moment
			simulator->runTimestep(localDt);

			simulator->unionLiquidSurface(seedLiquidSurface);

			frameTime += localDt;
		}

		runSingleTimestep = false;
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
		runSingleTimestep = true;
}

int main(int argc, char** argv)
{
	// Scene settings
	float dx = .025;
	Vec2f topRightCorner(2.5);
	Vec2f bottomLeftCorner(-2.5);
	gridSize = Vec2i((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2f center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Viscous Liquid Simulator", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	movingSolidsMesh = makeCircleMesh(center + Vec2f(1.2, 0), .25, 20);
	
	staticSolidsMesh = makeSquareMesh(center, Vec2f(2));
	staticSolidsMesh.reverse();
	assert(staticSolidsMesh.unitTestMesh());
	
	EdgeMesh beamLiquidMesh = makeSquareMesh(center - Vec2f(.8, 0), Vec2f(1.5, .2));
	assert(beamLiquidMesh.unitTestMesh());

	LevelSet beamLiquidSurface(xform, gridSize, 5);
	beamLiquidSurface.initFromMesh(beamLiquidMesh, false);

	EdgeMesh seedLiquidMesh = makeSquareMesh(center + Vec2f(0, .6), Vec2f(.075, .25));
	assert(seedLiquidMesh.unitTestMesh());

	seedLiquidSurface = LevelSet(xform, gridSize, 5);
	seedLiquidSurface.initFromMesh(seedLiquidMesh, false);

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(staticSolidsMesh, false);

	// Set up simulation
	simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 10);
	simulator->setLiquidSurface(beamLiquidSurface);
	simulator->unionLiquidSurface(seedLiquidSurface);
	simulator->setSolidSurface(solidSurface);

	simulator->setViscosity(30);

	// Set up simulation
	solidVelocityField = std::make_unique<CircularField>(center, .5);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);
	renderer->run();
}

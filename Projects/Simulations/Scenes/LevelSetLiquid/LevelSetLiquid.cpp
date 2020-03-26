#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"

using namespace FluidSim2D::RegularGridSim;
using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::Utilities;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianLiquidSimulator> simulator;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static bool drawLiquidVelocities = false;

static LevelSet seedSurface;

static bool printFrame = false;

static float seedTime = 0;
static constexpr float seedPeriod = 2;
static constexpr float dt = 1. / 60;

static constexpr float cfl = 5;

static Transform xform;

static LevelSet testLevelSet;

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

			if (seedTime > seedPeriod)
			{
				simulator->unionLiquidSurface(seedSurface);
				seedTime = 0;
			}
			
			simulator->addForce(localDt, Vec2f(0., -9.8));

			simulator->runTimestep(localDt);

			seedTime += localDt;
			
			// Store accumulated substep times
			frameTime += localDt;
		}
		std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
		++frameCount;

		runSingleTimestep = false;
		isDisplayDirty = true;
	}
	if (isDisplayDirty)
	{
		renderer->clear();

		simulator->drawVolumetricSurface(*renderer);
		simulator->drawLiquidSurface(*renderer);
		simulator->drawSolidSurface(*renderer);

		if (drawLiquidVelocities)
			simulator->drawLiquidVelocity(*renderer, .5);

		isDisplayDirty = false;

		if (printFrame)
		{
			std::string frameCountString = std::to_string(frameCount);
			std::string renderFilename = "freeSurfaceLiquidSimulation_" + std::string(4 - frameCountString.length(), '0') + frameCountString;
			renderer->printImage(renderFilename);
		}

		glutPostRedisplay();
	}
}

void keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
		runSimulation = !runSimulation;
	else if (key == 'n')
		runSingleTimestep = true;
	else if (key == 'p')
	{
		printFrame = !printFrame;
		isDisplayDirty = true;
	}
	else if (key == 'v')
	{
		drawLiquidVelocities = !drawLiquidVelocities;
		isDisplayDirty = true;
	}
}

int main(int argc, char** argv)
{
	// Scene settings
	float dx = .0125;
	Vec2f topRightCorner(2.5);
	Vec2f bottomLeftCorner(-2.5);
	Vec2i gridSize((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2f center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Levelset Liquid Simulator", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	EdgeMesh liquidMesh = makeCircleMesh(center - Vec2f(0,.65), 1, 100);
	assert(liquidMesh.unitTestMesh());

	EdgeMesh solidMesh = makeCircleMesh(center, 2, 100);
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());
	
	LevelSet liquidSurface(xform, gridSize, 5);
	liquidSurface.initFromMesh(liquidMesh, false);

	testLevelSet = liquidSurface;

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(solidMesh, false);
	
	Vec2f seedCenter = xform.offset() + Vec2f(xform.dx()) * Vec2f(gridSize / 2) + Vec2f(.8);
	EdgeMesh seedMesh = makeSquareMesh(seedCenter, Vec2f(.5));

	seedSurface = LevelSet(xform, gridSize, 5);
	seedSurface.initFromMesh(seedMesh, false);

	// Set up simulation

	simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 5);
	simulator->setLiquidSurface(liquidSurface);
	simulator->setSolidSurface(solidSurface);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}

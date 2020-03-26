#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "MultiMaterialLiquidSimulator.h"
#include "Renderer.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"

using namespace FluidSim2D::RegularGridSim;
using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::Utilities;

std::unique_ptr<MultiMaterialLiquidSimulator> multiMaterialSimulator;
std::unique_ptr<Renderer> renderer;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr float dt = 1. / 60.;

static constexpr float cfl = 5;

static bool printFrame = false;

static int liquidMaterialCount;
static int currentMaterial = 0;

static Transform xform;
static Vec2i gridSize;

static const float bubbleDensity = 1;
static const float liquidDensity = 1000;

void display()
{
	if (runSimulation || runSingleTimestep)
	{
		float frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			float speed = multiMaterialSimulator->maxVelocityMagnitude();
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

			for (int material = 0; material < liquidMaterialCount; ++material)
				multiMaterialSimulator->addForce(localDt, material, Vec2f(0., -9.8));

			multiMaterialSimulator->runTimestep(localDt, *renderer, frameCount);

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
		for (int material = 0; material < liquidMaterialCount; ++material)
			multiMaterialSimulator->drawMaterialSurface(*renderer, material);

		multiMaterialSimulator->drawSolidSurface(*renderer);

		for (int material = 0; material < liquidMaterialCount; ++material)
		{
			if (currentMaterial == material)
				multiMaterialSimulator->drawMaterialVelocity(*renderer, .25, material);
		}
		
		if (printFrame)
		{
			std::string frameCountString = std::to_string(frameCount);
			std::string renderFilename = "multimaterial_" + std::to_string(unsigned(bubbleDensity)) + "_" + std::string(4 - frameCountString.length(), '0') + frameCountString;
			renderer->printImage(renderFilename);
		}

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
	else if (key == 'm')
	{
		currentMaterial = (currentMaterial + 1) % liquidMaterialCount;
		isDisplayDirty = true;
	}
	else if (key == 'p')
	{
		printFrame = !printFrame;
		isDisplayDirty = true;
	}
}

int main(int argc, char** argv)
{
	// Scene settings
	float dx = .015;
	float boundaryPadding = 10.;

	Vec2f topRightCorner(1.5, 2.5);
	topRightCorner += dx * boundaryPadding;

	Vec2f bottomLeftCorner(-1.5, -2.5);
	bottomLeftCorner -= dx * boundaryPadding;

	Vec2f simulationSize = topRightCorner - bottomLeftCorner;
	gridSize = Vec2i(simulationSize / dx);

	xform = Transform(dx, bottomLeftCorner);
	Vec2f center = .5 * (topRightCorner + bottomLeftCorner);

	int pixelHeight = 1080;
	int pixelWidth = pixelHeight * (topRightCorner[0] - bottomLeftCorner[0]) / (topRightCorner[1] - bottomLeftCorner[1]);
	renderer = std::make_unique<Renderer>("Multimaterial Liquid Simulator", Vec2i(pixelWidth, pixelHeight), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Build outer boundary grid.
	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - Vec2f(boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(solidMesh, false);

	multiMaterialSimulator = std::make_unique<MultiMaterialLiquidSimulator>(xform, gridSize, 2, 5);

	multiMaterialSimulator->setSolidSurface(solidSurface);

	EdgeMesh bubbleMesh = makeCircleMesh(center, .75, 40);

	LevelSet bubbleSurface = LevelSet(xform, gridSize, 5);
	bubbleSurface.initFromMesh(bubbleMesh, false);
	bubbleMesh.reverse();

	EdgeMesh liquidMesh = solidMesh;
	liquidMesh.reverse();
	liquidMesh.insertMesh(bubbleMesh);

	LevelSet liquidSurface = LevelSet(xform, gridSize, 5);
	liquidSurface.initFromMesh(liquidMesh, false);

	multiMaterialSimulator->setMaterial(liquidSurface, liquidDensity, 0);
	multiMaterialSimulator->setMaterial(bubbleSurface, bubbleDensity, 1);

	liquidMaterialCount = 2;

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
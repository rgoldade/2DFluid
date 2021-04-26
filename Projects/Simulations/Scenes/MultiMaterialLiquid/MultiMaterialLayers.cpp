#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "MultiMaterialLiquidSimulator.h"
#include "Renderer.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim2D;

std::unique_ptr<MultiMaterialLiquidSimulator> multiMaterialSimulator;
std::unique_ptr<Renderer> renderer;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr double dt = 1. / 60.;

static constexpr double cfl = 5;

static bool printFrame = false;

static int liquidMaterialCount;
static int currentMaterial = 0;

static Transform xform;
static Vec2i gridSize;

static const double lowDensity = 1;
static const double mediumDensity = 1000;
static const double highDensity = 10000;

void display()
{
    if (runSimulation || runSingleTimestep)
    {
		double frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			double speed = multiMaterialSimulator->maxVelocityMagnitude();
			double localDt = dt - frameTime;
			assert(localDt >= 0);

			if (speed > 1E-6)
			{
				double cflDt = cfl * xform.dx() / speed;
				if (localDt > cflDt)
				{
					localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			if (localDt <= 0)
				break;

			for (int material = 0; material < liquidMaterialCount; ++material)
				multiMaterialSimulator->addForce(localDt, material, Vec2d(0., -9.8));

			multiMaterialSimulator->runTimestep(localDt);

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
			std::string renderFilename = "multimaterial_" + std::string(4 - frameCountString.length(), '0') + frameCountString;
			renderer->printImage(renderFilename);
		}

		isDisplayDirty = false;

		glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int, int)
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
	double dx = .015;
	double boundaryPadding = 10;

	Vec2d topRightCorner(1.5, 2.5);
	topRightCorner.array() += dx * boundaryPadding;

	Vec2d bottomLeftCorner(-1.5, -2.5);
	bottomLeftCorner.array() -= dx * boundaryPadding;

	Vec2d simulationSize = topRightCorner - bottomLeftCorner;
	gridSize = (simulationSize.array() / dx).matrix().cast<int>();

	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5f * (topRightCorner + bottomLeftCorner);

	unsigned pixelHeight = 1080;
	unsigned pixelWidth = pixelHeight * int((topRightCorner[0] - bottomLeftCorner[0]) / (topRightCorner[1] - bottomLeftCorner[1]));
	renderer = std::make_unique<Renderer>("Multimaterial Liquid Simulator", Vec2i(pixelWidth, pixelHeight), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Build outer boundary grid.
	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - Vec2d(boundaryPadding * xform.dx(), boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(solidMesh, false);

	multiMaterialSimulator = std::make_unique<MultiMaterialLiquidSimulator>(xform, gridSize, 3, 5.);

	multiMaterialSimulator->setSolidSurface(solidSurface);

	// Build three material surfaces

	EdgeMesh lowDensityMesh = makeSquareMesh(Vec2d(center[0], bottomLeftCorner[1] + dx * boundaryPadding + 1. / 6. * (topRightCorner[1] - bottomLeftCorner[1] - 2 * dx * boundaryPadding)), .5 * Vec2d(simulationSize[0] - 2. * boundaryPadding * xform.dx(), .33 * (simulationSize[1] - 2. * boundaryPadding * xform.dx())));

	LevelSet lowDensitySurface(xform, gridSize, 5);
	lowDensitySurface.initFromMesh(lowDensityMesh, false);
	lowDensityMesh.reverse();

	EdgeMesh highDensityMesh = makeSquareMesh(Vec2d(center[0], bottomLeftCorner[1] + dx * boundaryPadding + 5. / 6. * (topRightCorner[1] - bottomLeftCorner[1] - 2 * dx * boundaryPadding)), .5 * Vec2d(simulationSize[0] - 2. * boundaryPadding * xform.dx(), .33 * (simulationSize[1] - 2. * boundaryPadding * xform.dx())));

	LevelSet highDensitySurface(xform, gridSize, 5);
	highDensitySurface.initFromMesh(highDensityMesh, false);
	highDensityMesh.reverse();

	EdgeMesh mediumDensityMesh = solidMesh;
	mediumDensityMesh.reverse();
	mediumDensityMesh.insertMesh(lowDensityMesh);
	mediumDensityMesh.insertMesh(highDensityMesh);

	LevelSet mediumDensitySurface(xform, gridSize, 5);
	mediumDensitySurface.initFromMesh(mediumDensityMesh, false);

	multiMaterialSimulator->setMaterial(lowDensitySurface, lowDensity, 0);
	multiMaterialSimulator->setMaterial(mediumDensitySurface, mediumDensity, 1);
	multiMaterialSimulator->setMaterial(highDensitySurface, highDensity, 2);

	liquidMaterialCount = 3;

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
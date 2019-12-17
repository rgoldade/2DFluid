#include <memory>

#include "Common.h"
#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "MultiMaterialLiquid.h"
#include "Renderer.h"
#include "ScalarGrid.h"

std::unique_ptr<MultiMaterialLiquid> multiMaterialSimulator;
std::unique_ptr<Renderer> renderer;

static bool runSimulation = false;
static bool runSingleStep = false;
static bool isDisplayDirty = true;
static constexpr Real dt = 1. / 60.;

static bool printFrame = false;

static unsigned liquidMaterialCount;
static unsigned currentMaterial = 0;

static Transform xform;
static Vec2i gridSize;

static int frameCount = 0;
static const Real bubbleDensity = 10;
static const Real liquidDensity = 1000;

void display()
{
    if (runSimulation || runSingleStep)
    {
		Real frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			Real speed = multiMaterialSimulator->maxVelocityMagnitude();

			Real localDt = dt - frameTime;
			assert(localDt >= 0);

			if (speed > 1E-6)
			{
				Real cflDt = 5. * xform.dx() / speed;
				if (localDt > cflDt)
				{
					if (cflDt < dt / 20.)
						localDt = dt / 20.;
					else
						localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			if (localDt <= 0)
				break;

			for (int material = 0; material < liquidMaterialCount; ++material)
				multiMaterialSimulator->addForce(localDt, material, Vec2R(0., -9.8));

			multiMaterialSimulator->runTimestep(localDt, *renderer, frameCount);

			// Store accumulated substep times
			frameTime += localDt;
		}

		++frameCount;

		runSingleStep = false;
		isDisplayDirty = true;
    }
    if (isDisplayDirty)
    {
		renderer->clear();
		multiMaterialSimulator->drawMaterialSurface(*renderer, currentMaterial);
		multiMaterialSimulator->drawSolidSurface(*renderer);

		for (int material = 0; material < liquidMaterialCount; ++material)
			multiMaterialSimulator->drawMaterialVelocity(*renderer, .25, material);
		
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
		runSingleStep = true;
	else if (key == 'm')
	{
		currentMaterial = (currentMaterial + 1) % 2;
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
	Real dx = .015;
	Real boundaryPadding = 10.;

	Vec2R topRightCorner(1.5, 2.5);
	topRightCorner += dx * boundaryPadding;

	Vec2R bottomLeftCorner(-1.5, -2.5);
	bottomLeftCorner -= dx * boundaryPadding;

	Vec2R simulationSize = topRightCorner - bottomLeftCorner;
	gridSize = Vec2i(simulationSize / dx);

	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	unsigned pixelHeight = 1080;
	unsigned pixelWidth = pixelHeight * (topRightCorner[0] - bottomLeftCorner[0]) / (topRightCorner[1] - bottomLeftCorner[1]);
	renderer = std::make_unique<Renderer>("Multimaterial Liquid Simulator", Vec2i(pixelWidth, pixelHeight), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Build outer boundary grid.
	EdgeMesh solidMesh = InitialGeometry::makeSquareMesh(center, .5 * simulationSize - Vec2R(boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solidSurface = LevelSet(xform, gridSize, 10);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(solidMesh, false);

	// Build two-material level set.
	// Circle centered in the grid.

	EdgeMesh bubbleMesh = InitialGeometry::makeCircleMesh(center, .75, 40);

	LevelSet bubbleSurface = LevelSet(xform, gridSize, 10);
	bubbleSurface.initFromMesh(bubbleMesh, false);
	bubbleMesh.reverse();

	EdgeMesh liquidMesh = solidMesh;
	liquidMesh.reverse();
	liquidMesh.insertMesh(bubbleMesh);

	LevelSet liquidSurface = LevelSet(xform, gridSize, 10);
	liquidSurface.initFromMesh(liquidMesh, false);

	multiMaterialSimulator = std::make_unique<MultiMaterialLiquid>(xform, gridSize, 2, 5);

	multiMaterialSimulator->setSolidSurface(solidSurface);

	multiMaterialSimulator->setMaterial(liquidSurface, liquidDensity, 0);
	multiMaterialSimulator->setMaterial(bubbleSurface, bubbleDensity, 1);

	liquidMaterialCount = 2;

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
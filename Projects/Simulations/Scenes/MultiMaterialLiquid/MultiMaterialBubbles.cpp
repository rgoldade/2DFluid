#include <memory>

#include "Common.h"
#include "EdgeMesh.h"
#include "InitialConditions.h"
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
static constexpr Real dt = 1. / 30.;

static unsigned liquidMaterialCount;
static unsigned currentMaterial = 0;

static Transform xform;
static Vec2i gridSize;

static int frameCount = 0;
static const Real bubbleDensity = 10000;

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
				Real cflDt = 3. * xform.dx() / speed;
				if (localDt > cflDt)
				{
					localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			for (int material = 0; material < liquidMaterialCount; ++material)
				multiMaterialSimulator->addForce(localDt, material, Vec2R(0., -9.8));

			multiMaterialSimulator->runTimestep(localDt, *renderer);

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
		multiMaterialSimulator->drawMaterialVelocity(*renderer, .5, currentMaterial);

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
}

int main(int argc, char** argv)
{
	// Scene settings
	Real dx = .025;
	Real boundaryPadding = 10.;

	Vec2R topRightCorner(2.0, 2.5);
	topRightCorner += dx * boundaryPadding;

	Vec2R bottomLeftCorner(-2.0, -2.5);
	bottomLeftCorner -= dx * boundaryPadding;

	Vec2R simulationSize = topRightCorner - bottomLeftCorner;
	gridSize = Vec2i(simulationSize / dx);

	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	unsigned pixelHeight = 1080;
	unsigned pixelWidth = pixelHeight * (topRightCorner[0] - bottomLeftCorner[0]) / (topRightCorner[1] - bottomLeftCorner[1]);
	renderer = std::make_unique<Renderer>("Multimaterial Liquid Simulator", Vec2i(pixelWidth, pixelHeight), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Build outer boundary grid.
	EdgeMesh solidMesh = squareMesh(center, .5 * simulationSize - Vec2R(boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTest());

	LevelSet solidSurface = LevelSet(xform, gridSize, 10);
	solidSurface.setBoundaryNegative();
	solidSurface.initFromMesh(solidMesh, false);

	// Build two-material level set.
	// Circle centered in the grid.

	Vec2R bubbleOffset(0, 1.);
	EdgeMesh bubbleMesh = circleMesh(center - bubbleOffset, .75, 40);

	Real surfaceHeight = 1.;
	Vec2R surfaceCenter(center[0], topRightCorner[1] - .5 * surfaceHeight - boundaryPadding * dx);
	EdgeMesh surfaceMesh = squareMesh(surfaceCenter, Vec2R(.5 * simulationSize[0] - boundaryPadding * dx, .5*surfaceHeight));
	bubbleMesh.insertMesh(surfaceMesh);

	LevelSet bubbleSurface = LevelSet(xform, gridSize, 10);
	bubbleSurface.initFromMesh(bubbleMesh, false);
	bubbleMesh.reverse();

	/*Vec2R liquidSurfaceOffset(0., .5);
	Mesh2D liquidMesh = squareMesh(center - liquidSurfaceOffset, .5 * simulationSize - Vec2R(boundaryPadding * xform.dx()) - liquidSurfaceOffset);
    liquidMesh.insertMesh(bubbleMesh);*/

	EdgeMesh liquidMesh = solidMesh;

	liquidMesh.reverse();
	liquidMesh.insertMesh(bubbleMesh);
	LevelSet liquidSurface = LevelSet(xform, gridSize, 10);
	liquidSurface.initFromMesh(liquidMesh, false);

	multiMaterialSimulator = std::make_unique<MultiMaterialLiquid>(xform, gridSize, 2, 5);

	multiMaterialSimulator->setSolidSurface(solidSurface);

	multiMaterialSimulator->setMaterial(liquidSurface, 1000, 0);
	multiMaterialSimulator->setMaterial(bubbleSurface, bubbleDensity, 1);

	liquidMaterialCount = 2;

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	forEachVoxelRange(Vec2i(0), gridSize, [&](const Vec2i& cell)
	{
		if (liquidSurface(cell) > 0 && solidSurface(cell) > 0 && bubbleSurface(cell) > 0)
			renderer->addPoint(liquidSurface.indexToWorld(Vec2R(cell)), Vec3f(0,1,0), 4);
	});

	renderer->run();
}
#include <memory>

#include "Common.h"
#include "InitialConditions.h"
#include "Integrator.h"
#include "LevelSet2D.h"
#include "Mesh2D.h"
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
static Vec2ui gridSize;

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
		multiMaterialSimulator->drawCollisionSurface(*renderer);
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
	gridSize = Vec2ui(simulationSize / dx);

	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	unsigned pixelHeight = 1080;
	unsigned pixelWidth = pixelHeight * (topRightCorner[0] - bottomLeftCorner[0]) / (topRightCorner[1] - bottomLeftCorner[1]);
	renderer = std::make_unique<Renderer>("Multimaterial Liquid Simulator", Vec2ui(pixelWidth, pixelHeight), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	// Build outer boundary grid.
	Mesh2D solidMesh = squareMesh(center, .5 * simulationSize - Vec2R(boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTest());

	LevelSet2D solidSurface = LevelSet2D(xform, gridSize, 10);
	solidSurface.setInverted();
	solidSurface.init(solidMesh, false);

	// Build two-material level set.
	// Circle centered in the grid.

	Vec2R bubbleOffset(0, 1.);
	Mesh2D bubbleMesh = circleMesh(center - bubbleOffset, .75, 40);

	Real surfaceHeight = 1.;
	Vec2R surfaceCenter(center[0], topRightCorner[1] - .5 * surfaceHeight - boundaryPadding * dx);
	Mesh2D surfaceMesh = squareMesh(surfaceCenter, Vec2R(.5 * simulationSize[0] - boundaryPadding * dx, .5*surfaceHeight));
	bubbleMesh.insertMesh(surfaceMesh);

	LevelSet2D bubbleSurface = LevelSet2D(xform, gridSize, 10);
	bubbleSurface.init(bubbleMesh, false);
	bubbleMesh.reverse();

	/*Vec2R liquidSurfaceOffset(0., .5);
	Mesh2D liquidMesh = squareMesh(center - liquidSurfaceOffset, .5 * simulationSize - Vec2R(boundaryPadding * xform.dx()) - liquidSurfaceOffset);
    liquidMesh.insertMesh(bubbleMesh);*/

	Mesh2D liquidMesh = solidMesh;

	liquidMesh.reverse();
	liquidMesh.insertMesh(bubbleMesh);
	LevelSet2D liquidSurface = LevelSet2D(xform, gridSize, 10);
	liquidSurface.init(liquidMesh, false);

	multiMaterialSimulator = std::make_unique<MultiMaterialLiquid>(xform, gridSize, 2, 5);

	multiMaterialSimulator->setCollisionVolume(solidSurface);

	multiMaterialSimulator->setMaterial(liquidSurface, 1000, 0);
	multiMaterialSimulator->setMaterial(bubbleSurface, bubbleDensity, 1);

	liquidMaterialCount = 2;

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	forEachVoxelRange(Vec2ui(0), gridSize, [&](const Vec2ui& cell)
	{
		if (liquidSurface(cell) > 0 && solidSurface(cell) > 0 && bubbleSurface(cell) > 0)
			renderer->addPoint(liquidSurface.indexToWorld(Vec2R(cell)), Vec3f(0,1,0), 4);
	});

	renderer->run();
}
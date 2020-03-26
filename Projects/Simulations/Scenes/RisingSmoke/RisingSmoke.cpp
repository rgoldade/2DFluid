#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "EulerianSmokeSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "Transform.h"
#include "EdgeMesh.h"
#include "Utilities.h"
#include "Vec.h"

using namespace FluidSim2D::RegularGridSim;
using namespace FluidSim2D::RenderTools;
using namespace FluidSim2D::SurfaceTrackers;
using namespace FluidSim2D::SimTools;
using namespace FluidSim2D::Utilities;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianSmokeSimulator> simulator;

static ScalarGrid<float> seedSmokeDensity;
static ScalarGrid<float> seedSmokeTemperature;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr float dt = 1./30.;
static constexpr float ambientTemperature = 300;

static constexpr float cfl = 5.;
static Transform xform;

void setSmokeSource(const LevelSet& sourceVolume,
					float defaultDensity,
					ScalarGrid<float>& smokeDensity,
					float defaultTemperature,
					ScalarGrid<float>& smokeTemperature)
{
	Vec2i size = sourceVolume.size();

	float samples = 2;
	float sampleDx = 1. / samples;

	float dx = sourceVolume.dx();

	tbb::parallel_for(tbb::blocked_range<int>(0, smokeDensity.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = smokeDensity.unflatten(cellIndex);
			// Super sample to determine 
			if (sourceVolume(cell) < dx * 2.)
			{
				// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
				// first sample point is the .5 * sample_dx closer to (i,j).
				int insideVolumeCount = 0;
				for (float x = (float(cell[0]) - .5) + (.5 * sampleDx); x <= float(cell[0]) + .5; x += sampleDx)
					for (float y = (float(cell[1]) - .5) + (.5 * sampleDx); y <= float(cell[1]) + .5; y += sampleDx)
					{
						if (sourceVolume.biLerp(sourceVolume.indexToWorld(Vec2f(cell))) <= 0.) ++insideVolumeCount;
					}

				if (insideVolumeCount > 0)
				{
					float tempDensity = defaultDensity;// +.05 * Util::randhashd(cell[0] + cell[1] * sourceVolume.size()[0]);
					float tempTemperature = defaultTemperature;// +50. * Util::randhashd(cell[0] + cell[1] * sourceVolume.size()[0]);
					smokeDensity(cell) = tempDensity * float(insideVolumeCount) * sampleDx * sampleDx;
					smokeTemperature(cell) = tempTemperature * float(insideVolumeCount) * sampleDx * sampleDx;
				}
			}
		}
	});
}

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

			simulator->runTimestep(localDt);

			// Add smoke density and temperature source to simulation frame
			simulator->setSmokeSource(seedSmokeDensity, seedSmokeTemperature);

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

		simulator->drawFluidDensity(*renderer, 1);
		simulator->drawSolidSurface(*renderer);

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
	float boundaryPadding = 10.;

	Vec2f topRightCorner(2.5);
	topRightCorner += dx * boundaryPadding;

	Vec2f bottomLeftCorner(-2.5);
	bottomLeftCorner -= dx * boundaryPadding;

	Vec2f simulationSize = topRightCorner - bottomLeftCorner;
	Vec2i gridSize(simulationSize / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2f center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Smoke Simulator", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - Vec2f(boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solid(xform, gridSize, 5);
	solid.setBackgroundNegative();
	solid.initFromMesh(solidMesh, false);

	simulator = std::make_unique<EulerianSmokeSimulator>(xform, gridSize, 300);
	simulator->setSolidSurface(solid);

	// Set up source for smoke density and smoke temperature
	EdgeMesh sourceMesh = makeCircleMesh(center - Vec2f(0, 2.), .25, 40);
	LevelSet sourceVolume(xform, gridSize, 5);
	sourceVolume.initFromMesh(sourceMesh, false);
	
	// Super sample source volume to get a smooth volumetric representation.
	seedSmokeDensity = ScalarGrid<float>(xform, gridSize, 0);
	seedSmokeTemperature = ScalarGrid<float>(xform, gridSize, ambientTemperature);

	setSmokeSource(sourceVolume, .2, seedSmokeDensity, 350, seedSmokeTemperature);

	simulator->setSmokeSource(seedSmokeDensity, seedSmokeTemperature);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
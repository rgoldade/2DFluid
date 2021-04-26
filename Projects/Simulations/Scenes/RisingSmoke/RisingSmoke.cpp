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

using namespace FluidSim2D;

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianSmokeSimulator> simulator;

static ScalarGrid<double> seedSmokeDensity;
static ScalarGrid<double> seedSmokeTemperature;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr double dt = 1./30.;
static constexpr double ambientTemperature = 300;

static constexpr double cfl = 5.;
static Transform xform;

void setSmokeSource(const LevelSet& sourceVolume,
					double defaultDensity,
					ScalarGrid<double>& smokeDensity,
					double defaultTemperature,
					ScalarGrid<double>& smokeTemperature)
{
	Vec2i size = sourceVolume.size();

	double samples = 2;
	double sampleDx = 1. / samples;

	double dx = sourceVolume.dx();

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
				for (double x = (double(cell[0]) - .5) + (.5 * sampleDx); x <= double(cell[0]) + .5; x += sampleDx)
					for (double y = (double(cell[1]) - .5) + (.5 * sampleDx); y <= double(cell[1]) + .5; y += sampleDx)
					{
						if (sourceVolume.biLerp(sourceVolume.indexToWorld(cell.cast<double>())) <= 0.) ++insideVolumeCount;
					}

				if (insideVolumeCount > 0)
				{
					double tempDensity = defaultDensity;// +.05 * Util::randhashd(cell[0] + cell[1] * sourceVolume.size()[0]);
					double tempTemperature = defaultTemperature;// +50. * Util::randhashd(cell[0] + cell[1] * sourceVolume.size()[0]);
					smokeDensity(cell) = tempDensity * double(insideVolumeCount) * sampleDx * sampleDx;
					smokeTemperature(cell) = tempTemperature * double(insideVolumeCount) * sampleDx * sampleDx;
				}
			}
		}
	});
}

void display()
{
	if (runSimulation || runSingleTimestep)
	{
		double frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			double speed = simulator->maxVelocityMagnitude();

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

void keyboard(unsigned char key, int, int)
{
	if (key == ' ')
		runSimulation = !runSimulation;
	else if (key == 'n')
		runSingleTimestep = true;
}

int main(int argc, char** argv)
{
	// Scene settings
	double dx = .025;
	double boundaryPadding = 10.;

	Vec2d topRightCorner(2.5, 2.5);
	topRightCorner.array() += dx * boundaryPadding;

	Vec2d bottomLeftCorner(-2.5, -2.5);
	bottomLeftCorner.array() -= dx * boundaryPadding;

	Vec2d simulationSize = topRightCorner - bottomLeftCorner;
	Vec2i gridSize = (simulationSize / dx).cast<int>();
	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Smoke Simulator", Vec2i(1000, 1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - Vec2d(boundaryPadding * xform.dx(), boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solid(xform, gridSize, 5);
	solid.setBackgroundNegative();
	solid.initFromMesh(solidMesh, false);

	simulator = std::make_unique<EulerianSmokeSimulator>(xform, gridSize, 300);
	simulator->setSolidSurface(solid);

	// Set up source for smoke density and smoke temperature
	EdgeMesh sourceMesh = makeCircleMesh(center - Vec2d(0, 2.), .25, 40);
	LevelSet sourceVolume(xform, gridSize, 5);
	sourceVolume.initFromMesh(sourceMesh, false);
	
	// Super sample source volume to get a smooth volumetric representation.
	seedSmokeDensity = ScalarGrid<double>(xform, gridSize, 0);
	seedSmokeTemperature = ScalarGrid<double>(xform, gridSize, ambientTemperature);

	setSmokeSource(sourceVolume, .2, seedSmokeDensity, 350, seedSmokeTemperature);

	simulator->setSmokeSource(seedSmokeDensity, seedSmokeTemperature);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
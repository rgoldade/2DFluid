#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "EulerianSmokeSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Transform.h"
#include "Utilities.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static std::unique_ptr<EulerianSmokeSimulator> simulator;

static ScalarGrid<double> seedSmokeDensity;
static ScalarGrid<double> seedSmokeTemperature;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr double dt = 1. / 30.;
static constexpr double ambientTemperature = 300;
static constexpr double cfl = 5.;

static Transform xform;

static void setSmokeSource(const LevelSet& sourceVolume,
							double defaultDensity,
							ScalarGrid<double>& smokeDensity,
							double defaultTemperature,
							ScalarGrid<double>& smokeTemperature)
{
	double samples = 2;
	double sampleDx = 1. / samples;
	double dx = sourceVolume.dx();

	tbb::parallel_for(tbb::blocked_range<int>(0, smokeDensity.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = smokeDensity.unflatten(cellIndex);

			if (sourceVolume(cell) < dx * 2.)
			{
				int insideVolumeCount = 0;
				for (double x = (double(cell[0]) - .5) + (.5 * sampleDx); x <= double(cell[0]) + .5; x += sampleDx)
					for (double y = (double(cell[1]) - .5) + (.5 * sampleDx); y <= double(cell[1]) + .5; y += sampleDx)
					{
						if (sourceVolume.biLerp(sourceVolume.indexToWorld(cell.cast<double>())) <= 0.) ++insideVolumeCount;
					}

				if (insideVolumeCount > 0)
				{
					smokeDensity(cell) = defaultDensity * double(insideVolumeCount) * sampleDx * sampleDx;
					smokeTemperature(cell) = defaultTemperature * double(insideVolumeCount) * sampleDx * sampleDx;
				}
			}
		}
	});
}

int main()
{
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

	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - Vec2d(boundaryPadding * xform.dx(), boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solid(xform, gridSize, 5);
	solid.setBackgroundNegative();
	solid.initFromMesh(solidMesh, false);

	simulator = std::make_unique<EulerianSmokeSimulator>(xform, gridSize, 300);
	simulator->setSolidSurface(solid);

	EdgeMesh sourceMesh = makeCircleMesh(center - Vec2d(0, 2.), .25, 40);
	LevelSet sourceVolume(xform, gridSize, 5);
	sourceVolume.initFromMesh(sourceMesh, false);

	seedSmokeDensity = ScalarGrid<double>(xform, gridSize, 0);
	seedSmokeTemperature = ScalarGrid<double>(xform, gridSize, ambientTemperature);

	setSmokeSource(sourceVolume, .2, seedSmokeDensity, 350, seedSmokeTemperature);
	simulator->setSmokeSource(seedSmokeDensity, seedSmokeTemperature);

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	simulator->drawFluidDensity("simulator", 1);
	simulator->drawSolidSurface("simulator");

	polyscope::state::userCallback = [&]()
	{
		if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
		ImGui::SameLine();
		if (ImGui::Button("Step")) runSingleTimestep = true;

		if (runSimulation || runSingleTimestep)
		{
			double frameTime = 0.;
			std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

			while (frameTime < dt)
			{
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

				if (localDt <= 0) break;

				simulator->runTimestep(localDt);
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
			simulator->drawFluidDensity("simulator", 1);
			simulator->drawSolidSurface("simulator");

			isDisplayDirty = false;
		}
	};

	polyscope::show();
}

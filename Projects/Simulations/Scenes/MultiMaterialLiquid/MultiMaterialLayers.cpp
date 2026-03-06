#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "MultiMaterialLiquidSimulator.h"
#include "Transform.h"
#include "Utilities.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static std::unique_ptr<MultiMaterialLiquidSimulator> multiMaterialSimulator;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr double dt = 1. / 60.;
static constexpr double cfl = 5;

static int liquidMaterialCount;
static int currentMaterial = 0;

static Transform xform;
static Vec2i gridSize;

static const double lowDensity = 1;
static const double mediumDensity = 1000;
static const double highDensity = 10000;

int main()
{
	double dx = .015;
	double boundaryPadding = 10;

	Vec2d topRightCorner(1.5, 2.5);
	topRightCorner.array() += dx * boundaryPadding;

	Vec2d bottomLeftCorner(-1.5, -2.5);
	bottomLeftCorner.array() -= dx * boundaryPadding;

	Vec2d simulationSize = topRightCorner - bottomLeftCorner;
	gridSize = (simulationSize.array() / dx).matrix().cast<int>();

	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - Vec2d(boundaryPadding * xform.dx(), boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(solidMesh, false);

	multiMaterialSimulator = std::make_unique<MultiMaterialLiquidSimulator>(xform, gridSize, 3, 5.);
	multiMaterialSimulator->setSolidSurface(solidSurface);

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

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	for (int material = 0; material < liquidMaterialCount; ++material)
		multiMaterialSimulator->drawMaterialSurface("simulator", material);
	multiMaterialSimulator->drawSolidSurface("simulator");

	polyscope::state::userCallback = [&]()
	{
		if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
		ImGui::SameLine();
		if (ImGui::Button("Step")) runSingleTimestep = true;
		ImGui::SameLine();
		if (ImGui::Button("Next material"))
		{
			currentMaterial = (currentMaterial + 1) % liquidMaterialCount;
			isDisplayDirty = true;
		}
		ImGui::Text("Velocity material: %d", currentMaterial);

		if (runSimulation || runSingleTimestep)
		{
			double frameTime = 0.;
			std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

			while (frameTime < dt)
			{
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

				if (localDt <= 0) break;

				for (int material = 0; material < liquidMaterialCount; ++material)
					multiMaterialSimulator->addForce(localDt, material, Vec2d(0., -9.8));

				multiMaterialSimulator->runTimestep(localDt);
				frameTime += localDt;
			}

			std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
			++frameCount;

			runSingleTimestep = false;
			isDisplayDirty = true;
		}

		if (isDisplayDirty)
		{
			for (int material = 0; material < liquidMaterialCount; ++material)
				multiMaterialSimulator->drawMaterialSurface("simulator", material);

			multiMaterialSimulator->drawSolidSurface("simulator");
			multiMaterialSimulator->drawMaterialVelocity("simulator", .25, currentMaterial);

			isDisplayDirty = false;
		}
	};

	polyscope::show();
}

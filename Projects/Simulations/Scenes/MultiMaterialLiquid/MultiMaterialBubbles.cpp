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

static const double bubbleDensity = 1;
static const double liquidDensity = 1000;

int main()
{
	double dx = .015;
	double boundaryPadding = 10;

	Vec2d topRightCorner(1.5, 2.5);
	topRightCorner.array() += dx * boundaryPadding;

	Vec2d bottomLeftCorner(-1.5, -2.5);
	bottomLeftCorner.array() -= dx * boundaryPadding;

	Vec2d simulationSize = topRightCorner - bottomLeftCorner;
	gridSize = (simulationSize.array() / dx).cast<int>();

	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	EdgeMesh solidMesh = makeSquareMesh(center, .5 * simulationSize - xform.dx() * Vec2d(boundaryPadding, boundaryPadding));
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

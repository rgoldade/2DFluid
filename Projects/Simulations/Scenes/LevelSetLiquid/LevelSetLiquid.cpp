#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Transform.h"
#include "Utilities.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static std::unique_ptr<EulerianLiquidSimulator> simulator;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static bool drawLiquidVelocities = false;

static LevelSet seedSurface;

static double seedTime = 0;
static constexpr double seedPeriod = 2;
static constexpr double dt = 1. / 60.;
static constexpr double cfl = 5;

static Transform xform;

int main()
{
	double dx = .0125;
	Vec2d topRightCorner(2.5, 2.5);
	Vec2d bottomLeftCorner(-2.5, -2.5);
	Vec2i gridSize = ((topRightCorner - bottomLeftCorner).array() / dx).matrix().cast<int>();
	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	EdgeMesh liquidMesh = makeCircleMesh(center - Vec2d(0, .65), 1, 100);
	assert(liquidMesh.unitTestMesh());

	EdgeMesh solidMesh = makeCircleMesh(center, 2, 100);
	solidMesh.reverse();
	assert(solidMesh.unitTestMesh());

	LevelSet liquidSurface(xform, gridSize, 5);
	liquidSurface.initFromMesh(liquidMesh, false);

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(solidMesh, false);

	Vec2d seedCenter = xform.offset() + xform.dx() * (gridSize.array() / 2).matrix().cast<double>() + Vec2d(.8, .8);
	EdgeMesh seedMesh = makeSquareMesh(seedCenter, Vec2d(.5, .5));

	seedSurface = LevelSet(xform, gridSize, 5);
	seedSurface.initFromMesh(seedMesh, false);

	simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 5);
	simulator->setLiquidSurface(liquidSurface);
	simulator->setSolidSurface(solidSurface);

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	simulator->drawVolumetricSurface("simulator");
	simulator->drawLiquidSurface("simulator");
	simulator->drawSolidSurface("simulator");

	polyscope::state::userCallback = [&]()
	{
		if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
		ImGui::SameLine();
		if (ImGui::Button("Step")) runSingleTimestep = true;
		ImGui::SameLine();
		if (ImGui::Checkbox("Velocity", &drawLiquidVelocities))
			isDisplayDirty = true;

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

				if (seedTime > seedPeriod)
				{
					simulator->unionLiquidSurface(seedSurface);
					seedTime = 0;
				}

				simulator->addForce(localDt, Vec2d(0., -9.8));
				simulator->runTimestep(localDt);

				seedTime += localDt;
				frameTime += localDt;
			}

			std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
			++frameCount;

			runSingleTimestep = false;
			isDisplayDirty = true;
		}

		if (isDisplayDirty)
		{
			simulator->drawVolumetricSurface("simulator");
			simulator->drawLiquidSurface("simulator");
			simulator->drawSolidSurface("simulator");

			if (drawLiquidVelocities)
				simulator->drawLiquidVelocity("simulator", .5);

			isDisplayDirty = false;
		}
	};

	polyscope::show();
}

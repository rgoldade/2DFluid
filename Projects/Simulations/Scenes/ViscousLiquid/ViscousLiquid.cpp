#include <iostream>
#include <memory>

#include "EdgeMesh.h"
#include "EulerianLiquidSimulator.h"
#include "InitialGeometry.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "Transform.h"
#include "TestVelocityFields.h"
#include "Utilities.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static std::unique_ptr<EulerianLiquidSimulator> simulator;
static std::unique_ptr<CircularField> solidVelocityField;

static int frameCount = 0;
static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

static constexpr double dt = 1. / 30;
static constexpr double cfl = 5.;

static Transform xform;
static Vec2i gridSize;

static EdgeMesh movingSolidsMesh;
static EdgeMesh staticSolidsMesh;

static LevelSet seedLiquidSurface;

int main()
{
	double dx = .025;
	Vec2d topRightCorner(2.5, 2.5);
	Vec2d bottomLeftCorner(-2.5, -2.5);
	gridSize = ((topRightCorner - bottomLeftCorner).array() / dx).matrix().cast<int>();
	xform = Transform(dx, bottomLeftCorner);
	Vec2d center = .5 * (topRightCorner + bottomLeftCorner);

	movingSolidsMesh = makeCircleMesh(center + Vec2d(1.2, 0), .25, 20);

	staticSolidsMesh = makeSquareMesh(center, Vec2d(2, 2));
	staticSolidsMesh.reverse();
	assert(staticSolidsMesh.unitTestMesh());

	EdgeMesh beamLiquidMesh = makeSquareMesh(center - Vec2d(.8, 0), Vec2d(1.5, .2));
	assert(beamLiquidMesh.unitTestMesh());

	LevelSet beamLiquidSurface(xform, gridSize, 5);
	beamLiquidSurface.initFromMesh(beamLiquidMesh, false);

	EdgeMesh seedLiquidMesh = makeSquareMesh(center + Vec2d(0, .6), Vec2d(.075, .25));
	assert(seedLiquidMesh.unitTestMesh());

	seedLiquidSurface = LevelSet(xform, gridSize, 5);
	seedLiquidSurface.initFromMesh(seedLiquidMesh, false);

	LevelSet solidSurface(xform, gridSize, 5);
	solidSurface.setBackgroundNegative();
	solidSurface.initFromMesh(staticSolidsMesh, false);

	simulator = std::make_unique<EulerianLiquidSimulator>(xform, gridSize, 10);
	simulator->setLiquidSurface(beamLiquidSurface);
	simulator->unionLiquidSurface(seedLiquidSurface);
	simulator->setSolidSurface(solidSurface);
	simulator->setViscosity(30);

	solidVelocityField = std::make_unique<CircularField>(center, .5);

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	simulator->drawLiquidSurface("simulator");
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

				simulator->addForce(localDt, Vec2d(0., -9.8));

				movingSolidsMesh.advectMesh(localDt, *solidVelocityField, IntegrationOrder::RK3);

				LevelSet movingSolidSurface(xform, gridSize, 5);
				movingSolidSurface.initFromMesh(movingSolidsMesh, false);

				VectorGrid<double> movingSolidVelocity(xform, gridSize, 0, VectorGridSettings::SampleType::STAGGERED);

				for (int axis : {0, 1})
				{
					tbb::parallel_for(tbb::blocked_range<int>(0, movingSolidVelocity.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
					{
						for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
						{
							Vec2i face = movingSolidVelocity.grid(axis).unflatten(faceIndex);
							Vec2d worldPosition = movingSolidVelocity.indexToWorld(face.cast<double>(), axis);

							if (movingSolidSurface.biLerp(worldPosition) < xform.dx())
								movingSolidVelocity(face, axis) = (*solidVelocityField)(0, worldPosition)[axis];
						}
					});
				}

				LevelSet combinedSolidSurface(xform, gridSize, 5);
				combinedSolidSurface.setBackgroundNegative();
				combinedSolidSurface.initFromMesh(staticSolidsMesh, false);
				combinedSolidSurface.unionSurface(movingSolidSurface);

				simulator->setSolidSurface(combinedSolidSurface);
				simulator->setSolidVelocity(movingSolidVelocity);

				simulator->runTimestep(localDt);
				simulator->unionLiquidSurface(seedLiquidSurface);

				frameTime += localDt;
			}

			std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
			++frameCount;

			runSingleTimestep = false;
			isDisplayDirty = true;
		}

		if (isDisplayDirty)
		{
			simulator->drawLiquidSurface("simulator");
			simulator->drawSolidSurface("simulator");
			simulator->drawLiquidVelocity("simulator", .5);

			isDisplayDirty = false;
		}
	};

	polyscope::show();
}

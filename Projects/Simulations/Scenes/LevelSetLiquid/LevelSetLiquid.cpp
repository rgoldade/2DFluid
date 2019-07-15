#include <memory>

#include "Common.h"
#include "EdgeMesh.h"
#include "EulerianLiquid.h"
#include "InitialConditions.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "Renderer.h"

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianLiquid> simulator;

static bool runSimulation = false;
static bool runSingleStep = false;
static bool isDisplayDirty = true;
static constexpr Real dt = 1./30.;
static Real seedTime = 0;

static Transform xform;
static Vec2i gridSize;
static int frameCount = 0;

void display()
{
	if (runSimulation || runSingleStep)
	{
		Real frameTime = 0.;
		std::cout << "\nStart of frame: " << frameCount << ". Timestep: " << dt << std::endl;

		while (frameTime < dt)
		{
			// Set CFL condition
			Real speed = simulator->maxVelocityMagnitude();
			
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

			if (seedTime > 2.)
			{
				Vec2R center = xform.offset() + Vec2R(xform.dx()) * Vec2R(gridSize / 2) + Vec2R(.8);
				EdgeMesh seedMesh = squareMesh(center, Vec2R(.5));

				LevelSet seedSurface = LevelSet(xform, gridSize, 5);
				seedSurface.initFromMesh(seedMesh, false);
				simulator->unionLiquidSurface(seedSurface);
				seedTime = 0.;
			}
			
			simulator->addForce(localDt, Vec2R(0., -9.8));

			simulator->runTimestep(localDt, *renderer);

			seedTime += localDt;
			
			// Store accumulated substep times
			frameTime += localDt;
		}
		std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;
		++frameCount;
		runSingleStep = false;
		isDisplayDirty = true;
	}
	if (isDisplayDirty)
	{
		renderer->clear();

		simulator->drawLiquidSurface(*renderer);
		simulator->drawSolidSurface(*renderer);
		simulator->drawLiquidVelocity(*renderer, .5);

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
}

int main(int argc, char** argv)
{
	// Scene settings
	Real dx = .025;
	Vec2R topRightCorner(2.5);
	Vec2R bottomLeftCorner(-2.5);
	gridSize = Vec2i((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Levelset Liquid Simulator", Vec2i(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	EdgeMesh liquidMesh = circleMesh(center - Vec2R(0,.65), 1, 100);
	
	assert(liquidMesh.unitTest());

	EdgeMesh solidMesh = circleMesh(center, 2, 100);
	solidMesh.reverse();
	assert(solidMesh.unitTest());
	
	LevelSet liquidSurface = LevelSet(xform, gridSize, 10);
	liquidSurface.initFromMesh(liquidMesh, false);

	LevelSet solidSurface = LevelSet(xform, gridSize, 10);
	solidSurface.setBoundaryNegative();
	solidSurface.initFromMesh(solidMesh, false);
	
	// Set up simulation
	simulator = std::unique_ptr<EulerianLiquid>(new EulerianLiquid(xform, gridSize, 10));
	simulator->unionLiquidSurface(liquidSurface);
	simulator->setSolidSurface(solidSurface);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}

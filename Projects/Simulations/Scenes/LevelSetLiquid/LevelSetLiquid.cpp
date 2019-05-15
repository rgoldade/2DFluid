#include <memory>

#include "Common.h"
#include "EulerianLiquid.h"
#include "InitialConditions.h"
#include "Integrator.h"
#include "LevelSet2D.h"
#include "Mesh2D.h"
#include "Renderer.h"

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianLiquid> simulator;

static bool runSimulation = false;
static bool runSingleStep = false;
static bool isDisplayDirty = true;
static constexpr Real dt = 1./30.;
static Real seedTime = 0;

static Transform xform;
static Vec2ui gridSize;
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
				Mesh2D seedMesh = squareMesh(center, Vec2R(.5));

				LevelSet2D seedSurface = LevelSet2D(xform, gridSize, 5);
				seedSurface.init(seedMesh, false);
				simulator->unionLiquidSurface(seedSurface);
				seedTime = 0.;
			}
			
			simulator->addForce(localDt, Vec2R(0., -9.8));

			simulator->runTimestep(localDt, *renderer);

			seedTime += localDt;
			
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

		simulator->drawLiquidSurface(*renderer);
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
		runSingleStep = true;
}

int main(int argc, char** argv)
{
	// Scene settings
	Real dx = .025;
	Vec2R topRightCorner(2.5);
	Vec2R bottomLeftCorner(-2.5);
	gridSize = Vec2ui((topRightCorner - bottomLeftCorner) / dx);
	xform = Transform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Levelset Liquid Simulator", Vec2ui(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	Mesh2D liquidMesh = circleMesh(center - Vec2R(0,.65), 1, 40);
	
	assert(liquidMesh.unitTest());

	Mesh2D solidMesh = circleMesh(center, 2, 40);
	solidMesh.reverse();
	assert(solidMesh.unitTest());
	
	LevelSet2D liquidSurface = LevelSet2D(xform, gridSize, 10);
	liquidSurface.init(liquidMesh, false);

	LevelSet2D solidSurface = LevelSet2D(xform, gridSize, 10);
	solidSurface.setInverted();
	solidSurface.init(solidMesh, false);
	
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

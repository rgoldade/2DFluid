#include <memory>

#include "Common.h"
#include "EulerianSmoke.h"
#include "InitialConditions.h"
#include "Integrator.h"
#include "LevelSet2D.h"
#include "Mesh2D.h"
#include "Renderer.h"
#include "ScalarGrid.h"

static std::unique_ptr<Renderer> renderer;
static std::unique_ptr<EulerianSmoke> simulator;

static ScalarGrid<Real> smokeDensity;
static ScalarGrid<Real> smokeTemperature;

static bool runSimulation = false;
static bool runSingleStep = false;
static bool isDisplayDirty = true;
static bool printFrame = false;

static constexpr Real dt = 1./30.;
static constexpr Real ambientTemperature = 300;
static int frameCount = 0;

void display()
{
	if (runSimulation || runSingleStep)
	{
		Real frameTime = 0.;
		std::cout << "\n\nStart of frame: " << frameCount << ". Timestep: " << dt << "\n" << std::endl;

		int substep = 0;
		while (frameTime < dt)
		{
			// Set CFL condition
			Real speed = simulator->maxVelocityMagnitude();

			Real localDt = dt - frameTime;
			assert(localDt >= 0);

			if (speed > 1E-6)
			{
				Real cflDt = 5. * smokeDensity.dx() / speed;
				if (localDt > cflDt)
				{
					localDt = cflDt;
					std::cout << "\n  Throttling frame with substep: " << localDt << "\n" << std::endl;
				}
			}

			simulator->runTimestep(localDt, *renderer);

			// Add smoke density and temperature source to simulation frame
			simulator->setSmokeSource(smokeDensity, smokeTemperature);

			frameTime += localDt;

			++substep;
		}

		std::cout << "\n\nEnd of frame: " << frameCount << "\n" << std::endl;

		++frameCount;

		runSingleStep = false;
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
		runSingleStep = true;
}


void setSmokeSource(const LevelSet2D &sourceVolume,
						Real defaultDensity, ScalarGrid<Real> &smokeDensity,
						Real defaultTemperature, ScalarGrid<Real> &smokeTemperature)
{
	Vec2ui size = sourceVolume.size();

	Real samples = 2;
	Real sampleDx = 1. / samples;

	Real dx = sourceVolume.dx();

	forEachVoxelRange(Vec2ui(0), size, [&](const Vec2ui& cell)
	{
		// Super sample to determine 
		if (fabs(sourceVolume.interp(sourceVolume.indexToWorld(Vec2R(cell))) < dx * 2.))
		{
			// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
			// first sample point is the .5 * sample_dx closer to (i,j).
			int insideVolumeCount = 0;
			for (Real x = (Real(cell[0]) - .5) + (.5 * sampleDx); x < Real(cell[0]) + .5; x += sampleDx)
				for (Real y = (Real(cell[1]) - .5) + (.5 * sampleDx); y < Real(cell[1]) + .5; y += sampleDx)
				{
					if (sourceVolume.interp(sourceVolume.indexToWorld(Vec2R(cell))) <= 0.) ++insideVolumeCount;
				}

			if (insideVolumeCount > 0)
			{
				Real temp_density = defaultDensity;// +.05 * Util::randhashd(cell[0] + cell[1] * sourceVolume.size()[0]);
				Real temp_temperature = defaultTemperature;// +50. * Util::randhashd(cell[0] + cell[1] * sourceVolume.size()[0]);
				smokeDensity(cell) = temp_density * Real(insideVolumeCount) * sampleDx * sampleDx;
				smokeTemperature(cell) = temp_temperature * Real(insideVolumeCount) * sampleDx * sampleDx;
			}
		}
	});
}

int main(int argc, char** argv)
{
	// Scene settings
	Real dx = .025;
	Real boundaryPadding = 10.;

	Vec2R topRightCorner(2.5);
	topRightCorner += dx * boundaryPadding;

	Vec2R bottomLeftCorner(-2.5);
	bottomLeftCorner -= dx * boundaryPadding;

	Vec2R simulationSize = topRightCorner - bottomLeftCorner;
	Vec2ui gridSize(simulationSize / dx);
	Transform xform(dx, bottomLeftCorner);
	Vec2R center = .5 * (topRightCorner + bottomLeftCorner);

	renderer = std::make_unique<Renderer>("Smoke Simulator", Vec2ui(1000), bottomLeftCorner, topRightCorner[1] - bottomLeftCorner[1], &argc, argv);

	Mesh2D solidMesh = squareMesh(center, .5 * simulationSize - Vec2R(boundaryPadding * xform.dx()));
	solidMesh.reverse();
	assert(solidMesh.unitTest());

	LevelSet2D solid = LevelSet2D(xform, gridSize, 10);
	solid.setInverted();
	solid.init(solidMesh, false);

	simulator = std::make_unique<EulerianSmoke>(xform, gridSize, 300);
	simulator->setSolidSurface(solid);

	// Set up source for smoke density and smoke temperature
	Mesh2D sourceMesh = circleMesh(center - Vec2R(0, 2.), .25, 40);
	LevelSet2D sourceVolume = LevelSet2D(xform, gridSize, 10);
	sourceVolume.init(sourceMesh, false);
	
	// Super sample source volume to get a smooth volumetric representation.
	smokeDensity = ScalarGrid<Real>(xform, gridSize, 0);
	smokeTemperature = ScalarGrid<Real>(xform, gridSize, ambientTemperature);

	setSmokeSource(sourceVolume, .2, smokeDensity, 350, smokeTemperature);

	simulator->setSmokeSource(smokeDensity, smokeTemperature);

	std::function<void()> displayFunc = display;
	renderer->setUserDisplay(displayFunc);

	std::function<void(unsigned char, int, int)> keyboardFunc = keyboard;
	renderer->setUserKeyboard(keyboardFunc);

	renderer->run();
}
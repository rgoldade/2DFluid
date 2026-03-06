#include <memory>

#include "EdgeMesh.h"
#include "FluidParticles.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "Utilities.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static std::unique_ptr<FluidParticles> particles;

static bool isDisplayDirty = true;

static Transform xform;
static Vec2i gridSize;

int main()
{
	EdgeMesh initialMesh = makeCircleMesh();

	EdgeMesh tempMesh = makeCircleMesh(Vec2d(.5, .5), 1., 10);
	assert(tempMesh.unitTestMesh());
	initialMesh.insertMesh(tempMesh);

	tempMesh = makeCircleMesh(Vec2d(.05, .05), .5, 10);
	assert(tempMesh.unitTestMesh());
	initialMesh.insertMesh(tempMesh);

	assert(initialMesh.unitTestMesh());

	double dx = .125;
	Vec2d topRightCorner(2.25, 2.25);
	Vec2d bottomLeftCorner(-1.5, -1.5);
	gridSize = ((topRightCorner - bottomLeftCorner) / dx).cast<int>();
	xform = Transform(dx, bottomLeftCorner);

	LevelSet initialSurface(xform, gridSize, 5);
	initialSurface.initFromMesh(initialMesh, false);

	particles = std::make_unique<FluidParticles>(dx * .75, 8, 2);
	particles->init(initialSurface);

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	LevelSet sphereUnionSurface = particles->surfaceParticles(xform, gridSize, 5);
	sphereUnionSurface.drawGrid("sphereUnionSurface", true);
	sphereUnionSurface.drawSupersampledValues("sphereUnionSurface", .5, 5, 3);
	sphereUnionSurface.drawDCSurface("sphereUnionSurface", Vec3d(1., 0., 0.));
	sphereUnionSurface.drawNormals("sphereUnionSurface", Vec3d(.5, .5, .5), .1);
	particles->drawPoints("particles", Vec3d(1., 0., 0.), 1.);

	polyscope::state::userCallback = [&]()
	{
		if (ImGui::Button("Reseed"))
		{
			LevelSet reseedSurface = particles->surfaceParticles(xform, gridSize, 5);
			particles->reseed(reseedSurface);
			isDisplayDirty = true;
		}

		if (isDisplayDirty)
		{
			LevelSet sphereUnionSurface = particles->surfaceParticles(xform, gridSize, 5);
			sphereUnionSurface.drawGrid("sphereUnionSurface", true);
			sphereUnionSurface.drawSupersampledValues("sphereUnionSurface", .5, 5, 3);
			sphereUnionSurface.drawDCSurface("sphereUnionSurface", Vec3d(1., 0., 0.));
			sphereUnionSurface.drawNormals("sphereUnionSurface", Vec3d(.5, .5, .5), .1);
			particles->drawPoints("particles", Vec3d(1., 0., 0.), 1.);

			isDisplayDirty = false;
		}
	};

	polyscope::show();
}

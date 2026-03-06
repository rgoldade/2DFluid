#include <memory>

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "TestVelocityFields.h"
#include "Transform.h"
#include "Utilities.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

using namespace FluidSim2D;

static std::unique_ptr<LevelSet> surface;
static std::unique_ptr<CurlNoiseField> velocityField;

static constexpr double dt = 1. / 24.;

static bool runSimulation = false;
static bool runSingleTimestep = false;
static bool isDisplayDirty = true;

int main()
{
	EdgeMesh initialMesh = makeCircleMesh();

	EdgeMesh testMesh2 = makeCircleMesh(Vec2d(.5, .5), 1., 10.);
	EdgeMesh testMesh3 = makeCircleMesh(Vec2d(.05, .05), .5, 10.);

	assert(initialMesh.unitTestMesh());
	assert(testMesh2.unitTestMesh());
	assert(testMesh3.unitTestMesh());

	initialMesh.insertMesh(testMesh2);
	initialMesh.insertMesh(testMesh3);

	assert(initialMesh.unitTestMesh());

	AlignedBox2d bbox;
	for (int vertexIndex = 0; vertexIndex < initialMesh.vertexCount(); ++vertexIndex)
		bbox.extend(initialMesh.vertex(vertexIndex));

	Vec2d boundingBoxSize = bbox.max() - bbox.min();
	double dx = .05;
	Vec2d origin = bbox.min() - boundingBoxSize;
	Vec2i size = (3. * boundingBoxSize / dx).cast<int>();

	Transform xform(dx, origin);
	surface = std::make_unique<LevelSet>(xform, size, 5);
	surface->initFromMesh(initialMesh, true);

	velocityField = std::make_unique<CurlNoiseField>();

	polyscope::view::style = polyscope::NavigateStyle::Planar;
	polyscope::init();

	surface->drawGrid("surface", true);
	surface->drawDCSurface("surface", Vec3d(1., 0., 0.));
	surface->drawNormals("surface", Vec3d(.5, .5, .5), .05);

	polyscope::state::userCallback = [&]()
	{
		if (ImGui::Button("Run/Pause")) runSimulation = !runSimulation;
		ImGui::SameLine();
		if (ImGui::Button("Step")) runSingleTimestep = true;

		if (runSimulation || runSingleTimestep)
		{
			EdgeMesh surfaceMesh = surface->buildDCMesh();
			surfaceMesh.advectMesh(dt, *velocityField, IntegrationOrder::RK3);
			surface->initFromMesh(surfaceMesh, true);

			runSingleTimestep = false;
			isDisplayDirty = true;
		}

		if (isDisplayDirty)
		{
			surface->drawGrid("surface", true);
			surface->drawDCSurface("surface", Vec3d(1., 0., 0.));
			surface->drawNormals("surface", Vec3d(.5, .5, .5), .05);

			isDisplayDirty = false;
		}
	};

	polyscope::show();
}

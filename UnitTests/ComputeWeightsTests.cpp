#include <Eigen/Sparse>

#include "gtest/gtest.h"

#include "ComputeWeights.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D;

//
// Ghost fluid weights tests
//

void testGhostFluidWeights(const LevelSet& surface)
{
	// Build ghost fluid weights
	VectorGrid<double> weights = computeGhostFluidWeights(surface);

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), weights.size(axis), [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(face, axis, 0);
			Vec2i forwardCell = faceToCell(face, axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] == surface.size()[axis])
                EXPECT_EQ(weights(face, axis), 0);
            else if (surface(backwardCell) <= 0 && surface(forwardCell) <= 0)
                EXPECT_EQ(weights(face, axis), 1);
			else if (surface(backwardCell) > 0 && surface(forwardCell) > 0)
				EXPECT_EQ(weights(face, axis), 0);
			else
			{
				EXPECT_GE(weights(face, axis), 0);
				EXPECT_LE(weights(face, axis), 1);
			}
		});
	}
}

TEST(COMPUTE_WEIGHTS_TESTS, CIRCLE_GHOST_FLUID_WEIGHT_TEST)
{
    double radius = 1;
    Vec2d center = Vec2d::Zero();
	EdgeMesh mesh = makeCircleMesh(center, radius, 40);

	double dx = .01;
	Vec2d bottomLeft = center - 1.5 * Vec2d(radius, radius);
	Vec2d topRight = center + 1.5 * Vec2d(radius, radius);
	Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

	Transform xform(dx, bottomLeft);
	LevelSet surface(xform, gridSize, 10);

	surface.initFromMesh(mesh);

	testGhostFluidWeights(surface);
}

TEST(COMPUTE_WEIGHTS_TESTS, SQUARE_GHOST_FLUID_WEIGHT_TEST)
{
	Vec2d radius = Vec2d::Ones();
	Vec2d center = Vec2d::Zero();
	EdgeMesh mesh = makeSquareMesh(center, radius);

	double dx = .01;
	Vec2d bottomLeft = center - 1.5 * radius;
	Vec2d topRight = center + 1.5 * radius;
	Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

	Transform xform(dx, bottomLeft);
	LevelSet surface(xform, gridSize, 10);

	surface.initFromMesh(mesh);

	testGhostFluidWeights(surface);
}

//
// Cut-cell weights tests
//

void testCutCellWeights(const LevelSet& surface, bool invert)
{
    // Build ghost fluid weights
    VectorGrid<double> weights = computeCutCellWeights(surface, invert);

    for (int axis : {0, 1})
    {
        forEachVoxelRange(Vec2i::Zero(), weights.size(axis), [&](const Vec2i& face)
        {
            Vec2d offset = Vec2d::Zero();
            offset[(axis + 1) % 2] = .5;
            Vec2d backwardNode = weights.indexToWorld(face.cast<double>() - offset, axis);
            Vec2d forwardNode = weights.indexToWorld(face.cast<double>() + offset, axis);

            double backwardPhi = surface.biLerp(backwardNode);
            double forwardPhi = surface.biLerp(forwardNode);

            if (backwardPhi <= 0 && forwardPhi <= 0)
            {
                if (invert)
                    EXPECT_EQ(weights(face, axis), 0);
                else
                    EXPECT_EQ(weights(face, axis), 1);
            }
            else if (backwardPhi <= 0 && forwardPhi > 0 || backwardPhi > 0 && forwardPhi <= 0)
            {
                EXPECT_GT(weights(face, axis), 0);
                EXPECT_LT(weights(face, axis), 1);
            }
            else
            {
                EXPECT_GT(backwardPhi, 0);
                EXPECT_GT(forwardPhi, 0);

                if (invert)
                    EXPECT_EQ(weights(face, axis), 1);
                else
                    EXPECT_EQ(weights(face, axis), 0);
            }
        });
    }
}

TEST(COMPUTE_WEIGHTS_TESTS, CIRCLE_CUTCELL_WEIGHT_TEST)
{
    double radius = 1;
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    double dx = .01;
    Vec2d bottomLeft = center - 1.5 * Vec2d(radius, radius);
    Vec2d topRight = center + 1.5 * Vec2d(radius, radius);
    Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testCutCellWeights(surface, false);
    testCutCellWeights(surface, true);
}

TEST(COMPUTE_WEIGHTS_TESTS, SQUARE_CUTCELL_WEIGHT_TEST)
{
    Vec2d radius = Vec2d::Ones();
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeSquareMesh(center, radius);

    double dx = .01;
    Vec2d bottomLeft = center - 1.5 * radius;
    Vec2d topRight = center + 1.5 * radius;
    Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testCutCellWeights(surface, false);
    testCutCellWeights(surface, true);
}

//
// Super-sample areas test
//

void testSupersampledAreas(const LevelSet& surface, ScalarGridSettings::SampleType sampleType)
{
    // Build ghost fluid weights
    int samples = 3;
    ScalarGrid<double> weights = computeSupersampledAreas(surface, sampleType, samples);

    double sampleDx = double(1) / double(samples);
    forEachVoxelRange(Vec2i::Zero(), weights.size(), [&](const Vec2i& coord)
    {
        int accum = 0;
        Vec2d startPoint = coord.cast<double>() - .5 * Vec2d::Ones() + .5 * Vec2d(sampleDx, sampleDx);

        for (int i = 0; i < samples; ++i)
            for (int j = 0; j < samples; ++j)
            {
                Vec2d point = startPoint + Vec2d(i * sampleDx, j * sampleDx);
                Vec2d samplePoint = weights.indexToWorld(point);

                if (surface.biLerp(samplePoint) <= 0)
                    ++accum;
            }

        double weight = double(accum) / double(samples * samples);

        EXPECT_EQ(weights(coord), weight);
    });
}

TEST(COMPUTE_WEIGHTS_TESTS, CIRCLE_SUPERSAMPLE_WEIGHT_TEST)
{
    double radius = 1;
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    double dx = .01;
    Vec2d bottomLeft = center - 1.5 * Vec2d(radius, radius);
    Vec2d topRight = center + 1.5 * Vec2d(radius, radius);
    Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledAreas(surface, ScalarGridSettings::SampleType::CENTER);
    testSupersampledAreas(surface, ScalarGridSettings::SampleType::NODE);
}

TEST(COMPUTE_WEIGHTS_TESTS, SQUARE_SUPERSAMPLE_WEIGHT_TEST)
{
    Vec2d radius = Vec2d::Ones();
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeSquareMesh(center, radius);

    double dx = .01;
    Vec2d bottomLeft = center - 1.5 * radius;
    Vec2d topRight = center + 1.5 * radius;
    Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledAreas(surface, ScalarGridSettings::SampleType::CENTER);
    testSupersampledAreas(surface, ScalarGridSettings::SampleType::NODE);
}

//
// Super-sample face areas test
//

void testSupersampledFacesAreas(const LevelSet& surface)
{
    // Build ghost fluid weights
    int samples = 3;
    VectorGrid<double> weights = computeSupersampledFaceAreas(surface, samples);

    double sampleDx = double(1) / double(samples);
    for (int axis : {0, 1})
    {
        forEachVoxelRange(Vec2i::Zero(), weights.size(axis), [&](const Vec2i& face)
        {
            int accum = 0;

            Vec2d start = face.cast<double>() - .5 * Vec2d::Ones() + .5 * Vec2d(sampleDx, sampleDx);

            for (int i = 0; i < samples; ++i)
                for (int j = 0; j < samples; ++j)
                {
                    Vec2d point = start + Vec2d(i * sampleDx, j * sampleDx);
                    Vec2d samplePoint = weights.indexToWorld(point, axis);

                    if (surface.biLerp(samplePoint) <= 0.)
                        ++accum;
                }

            double weight = double(accum) / double(samples * samples);

            EXPECT_EQ(weights(face, axis), weight) << "Computed weights: " << weights(face, axis) << " Expected weight: " << weight;
        });
    }
}

TEST(COMPUTE_WEIGHTS_TESTS, CIRCLE_SUPERSAMPLE_FACE_WEIGHT_TEST)
{
    double radius = 1;
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeCircleMesh(center, radius, 40);

    double dx = .01;
    Vec2d bottomLeft = center - 1.5 * Vec2d(radius, radius);
    Vec2d topRight = center + 1.5 * Vec2d(radius, radius);
    Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledFacesAreas(surface);
}

TEST(COMPUTE_WEIGHTS_TESTS, SQUARE_SUPERSAMPLE_FACE_WEIGHT_TEST)
{
    Vec2d radius = Vec2d::Ones();
    Vec2d center = Vec2d::Zero();
    EdgeMesh mesh = makeSquareMesh(center, radius);

    double dx = .01;
    Vec2d bottomLeft = center - 1.5 * radius;
    Vec2d topRight = center + 1.5 * radius;
    Vec2i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledFacesAreas(surface);
}
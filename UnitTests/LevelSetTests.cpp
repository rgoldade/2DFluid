#include "gtest/gtest.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "EdgeMesh.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim2D;

TEST(LEVEL_SET_TESTS, INITIALIZE_TESTS)
{
    Vec2d origin = Vec2d::Random();
    double dx = .01;
    Transform xform(dx, origin);
    Vec2i gridSize(100, 100);

    LevelSet surfaceGrid(xform, gridSize);

    EXPECT_EQ(surfaceGrid.dx(), dx);
    EXPECT_EQ(surfaceGrid.offset()[0], origin[0]);
    EXPECT_EQ(surfaceGrid.offset()[1], origin[1]);
    EXPECT_EQ(surfaceGrid.xform(), xform);
    EXPECT_EQ(surfaceGrid.size()[0], gridSize[0]);
    EXPECT_EQ(surfaceGrid.size()[1], gridSize[1]);

    forEachVoxelRange(Vec2i::Zero(), gridSize, [&](const Vec2i& cell) { EXPECT_GT(surfaceGrid(cell), 0.); });

    LevelSet negativeBackgroundGrid(xform, gridSize, 10., true);

    EXPECT_TRUE(surfaceGrid.isGridMatched(negativeBackgroundGrid));

    forEachVoxelRange(Vec2i::Zero(), gridSize, [&](const Vec2i& cell) { EXPECT_LT(negativeBackgroundGrid(cell), 0.); });
}

template<typename IsoFunc>
static void testLevelSet(const IsoFunc& isoFunc, const EdgeMesh& mesh, double dx, bool rebuild)
{
    // Verify mesh lines up with the isosurface function
    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(isoFunc(vertex), 0., 1e-5, false));
    }

    // Get bounding box for mesh
    AlignedBox2d meshBBox = mesh.boundingBox();

    double bandwidth = 20.;

    Vec2d origin = meshBBox.min() - 2. * bandwidth * dx * Vec2d::Ones();
    Vec2d topRight = meshBBox.max() + 2. * bandwidth * dx * Vec2d::Ones();

    Transform xform(dx, origin);

    Vec2i gridSize = ((topRight - origin) / dx).cast<int>();

    LevelSet surfaceGrid(xform, gridSize, bandwidth);

    // Set iso surface values to the grid
    tbb::parallel_for(tbb::blocked_range<int>(0, surfaceGrid.voxelCount(), tbbLightGrainSize),[&](const tbb::blocked_range<int>& range) {
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec2i cell = surfaceGrid.unflatten(cellIndex);

            double phi = isoFunc(surfaceGrid.indexToWorld(cell.cast<double>()));

            surfaceGrid(cell) = phi;
        }
    });

    if (rebuild)
    {
        // Rebuild surface using fast marching
        surfaceGrid.reinit(true);

        // Verify narrow band enforced
        forEachVoxelRange(Vec2i::Zero(), surfaceGrid.size(), [&](const Vec2i& cell) { EXPECT_LE(std::fabs(surfaceGrid(cell)), bandwidth * dx); });
    }

    // Verify mesh points are on the zero isosurface
    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., 1e-5, false));
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., 1e-5, false));
    }

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec2i::Ones(), surfaceGrid.size() - Vec2i::Ones(), [&](const Vec2i& cell) {
            if (std::fabs(surfaceGrid(cell)) > narrowBand)
                return;

            Vec2d lerpNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();

            Vec2d cubicNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(cubicNorm.norm(), 1.));
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = surfaceGrid.buildMSMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., dx * 1e-2, false)) << "Bilerp value: " << surfaceGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., dx * 1e-2, false)) << "Bicubic value: " << surfaceGrid.biCubicInterp(vertex);
            EXPECT_TRUE(isNearlyEqual(isoFunc(vertex), 0., 1e-5, false));
        }
    }

    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = surfaceGrid.buildDCMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., dx * 1e-2, false)) << "Bilerp value: " << surfaceGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., dx * 1e-2, false)) << "Bicubic value: " << surfaceGrid.biCubicInterp(vertex);
            EXPECT_TRUE(isNearlyEqual(isoFunc(vertex), 0., 1e-5, false));
        }
    }
}

TEST(LEVEL_SET_TESTS, CIRCLE_ISO_SURFACE_TEST)
{
    Vec2d center = Vec2d::Random();
    double radius = 1;
    EdgeMesh mesh = makeCircleMesh(center, radius, 1000);

    auto isoSurface = [&center, &radius](const Vec2d& point) -> double { return std::sqrt(std::pow(point[0] - center[0], 2) + std::pow(point[1] - center[1], 2)) - radius; };

    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(isoSurface(vertex), 0., 1e-5, false));
    }

    double dx = .005;

    testLevelSet(isoSurface, mesh, dx, false);
}

TEST(LEVEL_SET_TESTS, REBUILD_CIRCLE_ISO_SURFACE_TEST)
{
    Vec2d center = Vec2d::Random();
    double radius = 1;
    EdgeMesh mesh = makeCircleMesh(center, radius, 1000);

    auto isoSurface = [&center, &radius](const Vec2d& point) -> double { return std::sqrt(std::pow(point[0] - center[0], 2) + std::pow(point[1] - center[1], 2)) - radius; };

    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(isoSurface(vertex), 0., 1e-5, false));
    }

    double dx = .005;

    testLevelSet(isoSurface, mesh, dx, true);
}

static void testUnionFunctions(const std::vector<std::function<double(const Vec2d&)>>& isoFuncs, const AlignedBox2d& bbox, double dx)
{
    // Build individual grids for each iso function
    Transform xform(dx, bbox.min());
    Vec2i gridSize = ceil(((bbox.max() - bbox.min()) / dx).eval()).cast<int>();
    double bandwidth = 20;
    std::vector<LevelSet> surfaceGrids(isoFuncs.size(), LevelSet(xform, gridSize, bandwidth));

    for (int gridIndex = 0; gridIndex < surfaceGrids.size(); ++gridIndex)
    {
        LevelSet& grid = surfaceGrids[gridIndex];
        const auto& isoFunc = isoFuncs[gridIndex];

        tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec2i cell = grid.unflatten(cellIndex);
                grid(cell) = isoFunc(grid.indexToWorld(cell.cast<double>()));
            }
        });
    }

    // Build union level set directly from lambdas
    LevelSet unionFuncGrid(xform, gridSize, bandwidth);

    for (int gridIndex = 0; gridIndex < surfaceGrids.size(); ++gridIndex)
    {
        ASSERT_TRUE(unionFuncGrid.isGridMatched(surfaceGrids[gridIndex]));
    }

    tbb::parallel_for(tbb::blocked_range<int>(0, unionFuncGrid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec2i cell = unionFuncGrid.unflatten(cellIndex);

            double minVal = std::numeric_limits<double>::max();
            for (const auto& isoFunc : isoFuncs)
            {
                minVal = std::min(isoFunc(unionFuncGrid.indexToWorld(cell.cast<double>())), minVal);
            }

            unionFuncGrid(cell) = minVal;
        }
    });

    LevelSet unionGrid(xform, gridSize, bandwidth);

    ASSERT_TRUE(unionFuncGrid.isGridMatched(unionGrid));

    for (int gridIndex = 0; gridIndex < surfaceGrids.size(); ++gridIndex)
    {
        unionGrid.unionSurface(surfaceGrids[gridIndex]);
    }

    // Verify both unions match
    forEachVoxelRange(Vec2i::Zero(), unionGrid.size(), [&](const Vec2i& cell) { EXPECT_EQ(unionGrid(cell), unionFuncGrid(cell)); });

    // Verify outer band of cells are "outside"
    for (int axis : {0, 1})
        for (int direction : {0, 1})
        {
            Vec2i start = Vec2i::Zero();
            Vec2i end = unionGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = unionGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec2i& cell) {
                ASSERT_GT(unionGrid(cell), 0.);
                ASSERT_GT(unionFuncGrid(cell), 0.);
            });
        }

    unionFuncGrid.reinit(true);
    unionGrid.reinit(true);

    // Verify both unions match after rebuilding
    forEachVoxelRange(Vec2i::Zero(), unionGrid.size(), [&](const Vec2i& cell) { EXPECT_EQ(unionGrid(cell), unionFuncGrid(cell)); });

    // Verify inside / outside of union grid against iso functions
    forEachVoxelRange(Vec2i::Zero(), unionGrid.size(), [&](const Vec2i& cell) {
        Vec2d worldPoint = unionGrid.indexToWorld(cell.cast<double>());
        if (unionGrid(cell) > 0)
        {
            for (const auto& isoFunc : isoFuncs)
            {
                EXPECT_GT(isoFunc(worldPoint), 0.);
            }
        }
        else
        {
            int insideCount = 0;
            for (const auto& isoFunc : isoFuncs)
            {
                if (isoFunc(worldPoint) <= 0)
                    ++insideCount;
            }
            EXPECT_GT(insideCount, 0);
        }
    });

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec2i::Ones(), unionGrid.size() - Vec2i::Ones(), [&](const Vec2i& cell) {
            if (std::fabs(unionGrid(cell)) > narrowBand)
                return;

            Vec2d lerpNorm = unionGrid.normal(unionGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();

            Vec2d cubicNorm = unionGrid.normal(unionGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(cubicNorm.norm(), 1.));
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = unionGrid.buildMSMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(unionGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp value: " << unionGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(unionGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic value: " << unionGrid.biCubicInterp(vertex);

            double minVal = std::numeric_limits<double>::max();
            for (const auto& isoFunc : isoFuncs)
            {
                minVal = std::min(minVal, isoFunc(vertex));
            }

            EXPECT_TRUE(isNearlyEqual(minVal, 0., 1e-5, false));
        }
    }

    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = unionGrid.buildDCMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(unionGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp value: " << unionGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(unionGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic value: " << unionGrid.biCubicInterp(vertex);

            double minVal = std::numeric_limits<double>::max();
            for (const auto& isoFunc : isoFuncs)
            {
                minVal = std::min(minVal, isoFunc(vertex));
            }

            EXPECT_TRUE(isNearlyEqual(minVal, 0., 1e-5, false));
        }
    }
}

TEST(LEVEL_SET_TESTS, CIRCLE_UNION_TEST)
{
    auto circleIso = [](const Vec2d& center, const Vec2d& point, double radius) { return std::sqrt(std::pow(point[0] - center[0], 2) + std::pow(point[1] - center[1], 2)) - radius; };
    auto iso0 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(0, 0);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso1 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(.5, .5);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso2 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(.75, 0);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso3 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(.5, -.5);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    std::vector<std::function<double(const Vec2d&)>> isoFuncs;
    isoFuncs.push_back(iso0);
    isoFuncs.push_back(iso1);
    isoFuncs.push_back(iso2);
    isoFuncs.push_back(iso3);

    AlignedBox2d bbox;
    bbox.extend(Vec2d(-2, -2));
    bbox.extend(Vec2d(2, 2));
}

static void testInitFromMesh(const EdgeMesh& mesh, double dx)
{
    double bandwidth = 10;
    Transform xform(dx, Vec2d::Zero());
    LevelSet surfaceGrid(xform, Vec2i(100, 100), bandwidth);

    surfaceGrid.initFromMesh(mesh, true);

    // Verify the mesh falls entirely inside the grid
    for (const Vec2d& vertex : mesh.vertices())
    {
        Vec2d indexPoint = surfaceGrid.worldToIndex(vertex);

        EXPECT_GT(indexPoint[0], 0.);
        EXPECT_GT(indexPoint[1], 0.);
        EXPECT_LT(indexPoint[0], double(surfaceGrid.size()[0] - 1));
        EXPECT_LT(indexPoint[1], double(surfaceGrid.size()[1] - 1));
    }

    // Verify outer band of cells are "outside"
    for (int axis : {0, 1})
        for (int direction : {0, 1})
        {
            Vec2i start = Vec2i::Zero();
            Vec2i end = surfaceGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = surfaceGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec2i& cell) { ASSERT_GT(surfaceGrid(cell), 0.); });
        }

    // Verify mesh points are on the zero isosurface
    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp: " << surfaceGrid.biLerp(vertex);
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic: " << surfaceGrid.biLerp(vertex);
        ;
    }

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec2i::Ones(), surfaceGrid.size() - Vec2i::Ones(), [&](const Vec2i& cell) {
            if (std::fabs(surfaceGrid(cell)) > narrowBand)
                return;

            Vec2d lerpNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();

            Vec2d cubicNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(cubicNorm.norm(), 1.));
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = surfaceGrid.buildMSMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp value: " << surfaceGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic value: " << surfaceGrid.biCubicInterp(vertex);
        }
    }

    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = surfaceGrid.buildDCMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp value: " << surfaceGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic value: " << surfaceGrid.biCubicInterp(vertex);
        }
    }
}

TEST(LEVEL_SET_TESTS, INIT_FROM_MESH)
{
    Vec2d origin = Vec2d::Random();
    double radius = 1.5;
    EdgeMesh mesh = makeCircleMesh(origin, radius, 1000);

    double dx = .005;
    testInitFromMesh(mesh, dx);
}

static void testInitFromMeshUnion(const std::vector<EdgeMesh>& meshes, const std::vector<std::function<double(const Vec2d&)>>& isoFuncs, double dx)
{
    EdgeMesh unionMesh = meshes[0];
    unionMesh.insertMesh(meshes[1]);
    unionMesh.insertMesh(meshes[2]);
    unionMesh.insertMesh(meshes[3]);

    double bandwidth = 10;
    Transform xform(dx, Vec2d::Zero());
    LevelSet surfaceGrid(xform, Vec2i(100, 100), bandwidth);

    surfaceGrid.initFromMesh(unionMesh, true);

    // Verify the mesh falls entirely inside the grid
    for (const Vec2d& vertex : unionMesh.vertices())
    {
        Vec2d indexPoint = surfaceGrid.worldToIndex(vertex);

        EXPECT_GT(indexPoint[0], 0.);
        EXPECT_GT(indexPoint[1], 0.);
        EXPECT_LT(indexPoint[0], double(surfaceGrid.size()[0] - 1));
        EXPECT_LT(indexPoint[1], double(surfaceGrid.size()[1] - 1));
    }

    // Verify outer band of cells are "outside"
    for (int axis : {0, 1})
        for (int direction : {0, 1})
        {
            Vec2i start = Vec2i::Zero();
            Vec2i end = surfaceGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = surfaceGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec2i& cell) { ASSERT_GT(surfaceGrid(cell), 0.); });
        }

    auto unionIsoFunc = [&isoFuncs](const Vec2d& point) {
        double minVal = std::numeric_limits<double>::max();

        for (auto& isoFunc : isoFuncs)
            minVal = std::min(minVal, isoFunc(point));

        return minVal;
    };

    // Verify mesh points fall on their own iso surface
    for (int meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
    {
        for (const Vec2d& vertex : meshes[meshIndex].vertices())
        {
            EXPECT_TRUE(isNearlyEqual(isoFuncs[meshIndex](vertex), 0., dx * 1e-2, false));
            EXPECT_TRUE(isNearlyEqual(isoFuncs[meshIndex](vertex), 0., dx * 1e-2, false));
        }
    }

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec2i::Ones(), surfaceGrid.size() - Vec2i::Ones(), [&](const Vec2i& cell) {
            if (std::fabs(surfaceGrid(cell)) > narrowBand)
                return;

            Vec2d lerpNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();

            Vec2d cubicNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()), true);

            EXPECT_TRUE(isNearlyEqual(cubicNorm.norm(), 1.));
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = surfaceGrid.buildMSMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., dx * 1e-2, false)) << "Bilerp value: " << surfaceGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., dx * 1e-2, false)) << "Bicubic value: " << surfaceGrid.biCubicInterp(vertex);
        }
    }

    {
        // Verify output mesh is close to the initial mesh
        EdgeMesh outputMesh = surfaceGrid.buildDCMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec2d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., dx * 1e-2, false)) << "Bilerp value: " << surfaceGrid.biLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., dx * 1e-2, false)) << "Bicubic value: " << surfaceGrid.biCubicInterp(vertex);
        }
    }
}

TEST(LEVEL_SET_TESTS, UNION_INIT_FROM_MESH)
{
    double radius = 1;
    std::vector<EdgeMesh> meshes;
    meshes.push_back(makeCircleMesh(Vec2d::Zero(), radius, 5000));
    meshes.push_back(makeCircleMesh(Vec2d(.5, .5), radius, 5000));
    meshes.push_back(makeCircleMesh(Vec2d(.75, 0), radius, 5000));
    meshes.push_back(makeCircleMesh(Vec2d(.5, -.5), radius, 5000));

    auto circleIso = [](const Vec2d& center, const Vec2d& point, double radius) { return std::sqrt(std::pow(point[0] - center[0], 2) + std::pow(point[1] - center[1], 2)) - radius; };
    auto iso0 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(0, 0);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso1 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(.5, .5);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso2 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(.75, 0);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso3 = [&circleIso](const Vec2d& point) -> double {
        Vec2d center(.5, -.5);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    std::vector<std::function<double(const Vec2d&)>> isoFuncs;
    isoFuncs.push_back(iso0);
    isoFuncs.push_back(iso1);
    isoFuncs.push_back(iso2);
    isoFuncs.push_back(iso3);

    double dx = .005;
    testInitFromMeshUnion(meshes, isoFuncs, dx);
}

// Test jittering mesh and iterating to the surface using gradients

static void testJitterMesh(const EdgeMesh& mesh, double dx)
{
    double bandwidth = 10;
    Transform xform(dx, Vec2d::Zero());
    LevelSet surfaceGrid(xform, Vec2i(100, 100), bandwidth);

    surfaceGrid.initFromMesh(mesh, true);

    // Jitter vertex and then iterate it back to the zero isosurface
    for (Vec2d vertex : mesh.vertices())
    {
        vertex += 2. * dx * Vec2d::Random();

        vertex = surfaceGrid.findSurface(vertex, 100, 1e-5);

        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp: " << surfaceGrid.biLerp(vertex);
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic: " << surfaceGrid.biCubicInterp(vertex);
    }
}

TEST(LEVEL_SET_TESTS, FIND_SURFACE_TEST)
{
    Vec2d origin = Vec2d::Random();
    double radius = 1.5;
    EdgeMesh mesh = makeCircleMesh(origin, radius, 1000);

    double dx = .005;
    testJitterMesh(mesh, dx);
}

// Test re-initializing with a mesh that extends outside of the grid

static void testOutOfBoundsMesh(const EdgeMesh& mesh, double dx)
{
    double bandwidth = 10;

    AlignedBox2d bbox = mesh.boundingBox();

    Vec2d origin = bbox.min() + 10. * dx * Vec2d::Ones();
    Vec2d topRight = bbox.max() - 10. * dx * Vec2d::Ones();
    Transform xform(dx, origin);

    Vec2i gridSize = ceil(((topRight - origin) / dx).eval()).cast<int>();
    LevelSet surfaceGrid(xform, gridSize, bandwidth);

    surfaceGrid.initFromMesh(mesh, false);

    // Verify the mesh falls entirely inside the grid
    bool outOfBounds = false;
    for (const Vec2d& vertex : mesh.vertices())
    {
        Vec2d indexPoint = surfaceGrid.worldToIndex(vertex);

        if (indexPoint[0] <= 0 || indexPoint[1] <= 0 || indexPoint[0] >= surfaceGrid.size()[0] - 1 || indexPoint[1] >= surfaceGrid.size()[1] - 1)
            outOfBounds = true;
    }

    EXPECT_TRUE(outOfBounds);

    // Verify outer band of cells are "outside" even though
    // the mesh falls outside of the grid
    for (int axis : {0, 1})
        for (int direction : {0, 1})
        {
            Vec2i start = Vec2i::Zero();
            Vec2i end = surfaceGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = surfaceGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec2i& cell) { ASSERT_GT(surfaceGrid(cell), 0.); });
        }
}

TEST(LEVEL_SET_TESTS, OUT_OF_BOUNDS_TEST)
{
    Vec2d origin = Vec2d::Random();
    double radius = 1.5;
    EdgeMesh mesh = makeCircleMesh(origin, radius, 1000);

    double dx = .005;
    testOutOfBoundsMesh(mesh, dx);
}

TEST(LEVEL_SET_TESTS, ADVECT_TEST)
{
    Vec2d origin = Vec2d::Random();
    double radius = 1.5;
    EdgeMesh mesh = makeCircleMesh(origin, radius, 1000);

    double dx = .005;
    LevelSet surfaceGrid(Transform(dx, Vec2d::Zero()), Vec2i(50, 50), 20);
    surfaceGrid.initFromMesh(mesh);

    auto velFunc = [](double, const Vec2d&) { return Vec2d(-1, 1); };

    surfaceGrid.advect(4. * dx, velFunc, IntegrationOrder::FORWARDEULER);
    surfaceGrid.reinit();

    mesh.advectMesh(4. * dx, velFunc, IntegrationOrder::FORWARDEULER);

    // Verify mesh points are on the zero isosurface or inside the unioned surface
    for (const Vec2d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biLerp(vertex), 0., 1e-5, false)) << "Bilerp: " << surfaceGrid.biLerp(vertex);
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.biCubicInterp(vertex), 0., 1e-5, false)) << "Bicubic: " << surfaceGrid.biCubicInterp(vertex);
    }
}

#include "gtest/gtest.h"

#include <Eigen/Sparse>

#include "GeometricMultigridOperators.h"
#include "GeometricMultigridPoissonSolver.h"
#include "InitialMultigridTestDomains.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim2D;
using namespace FluidSim2D::GeometricMultigridOperators;

static void testBoundaryCells(const UniformGrid<CellLabels>& cellLabels, const VectorGrid<double>* boundaryWeights)
{
    if (boundaryWeights != nullptr)
    {
        ASSERT_TRUE(boundaryWeights->sampleType() == VectorGridSettings::SampleType::STAGGERED);
        ASSERT_TRUE(boundaryWeights->size(0)[0] == cellLabels.size()[0] + 1);
        ASSERT_TRUE(boundaryWeights->size(0)[1] == cellLabels.size()[1]);
        ASSERT_TRUE(boundaryWeights->size(1)[0] == cellLabels.size()[0]);
        ASSERT_TRUE(boundaryWeights->size(1)[1] == cellLabels.size()[1] + 1);
    }

    // Verfify that cells along the boundary are not active
    for (int axis : {0,1})
        for (int direction : {0,1})
        {
            Vec2i startCell = Vec2i::Zero();
            Vec2i endCell = cellLabels.size();

            if (direction == 0)
                endCell[axis] = 1;
            else
                startCell[axis] = cellLabels.size()[axis] - 1;

            forEachVoxelRange(startCell, endCell, [&](const Vec2i& cell) { 
                ASSERT_TRUE(cellLabels(cell) == CellLabels::EXTERIOR_CELL || cellLabels(cell) == CellLabels::DIRICHLET_CELL);
            });            
        }

    forEachVoxelRange(Vec2i::Zero(), cellLabels.size(), [&](const Vec2i& cell) {

        // Verify that an interior cell only has interior and boundary cell neighbours
        if (cellLabels(cell) == CellLabels::INTERIOR_CELL)
        {
            ASSERT_TRUE(cell[0] > 0);
            ASSERT_TRUE(cell[1] > 0);
            ASSERT_TRUE(cell[0] < cellLabels.size()[0] - 1);
            ASSERT_TRUE(cell[1] < cellLabels.size()[1] - 1);

            for (int axis : {0, 1})
                for (int direction : {0, 1})
                {
                    Vec2i adjacentCell = cellToCell(cell, axis, direction);                  

                    EXPECT_TRUE(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL || cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

                    if (boundaryWeights != nullptr)
                    {
                        Vec2i face = cellToFace(cell, axis, direction);
                        EXPECT_TRUE((*boundaryWeights)(face, axis) == 1);
                    }
                }
        }
        else if (cellLabels(cell) == CellLabels::BOUNDARY_CELL)
        {
            ASSERT_TRUE(cell[0] > 0);
            ASSERT_TRUE(cell[1] > 0);
            ASSERT_TRUE(cell[0] < cellLabels.size()[0] - 1);
            ASSERT_TRUE(cell[1] < cellLabels.size()[1] - 1);

            bool hasValidBoundary = false;

            for (int axis : {0, 1})
                for (int direction : {0, 1})
                {
                    Vec2i adjacentCell = cellToCell(cell, axis, direction);

                    // A boundary cell should have at least one neighbour that is not active in the solver OR a boundary weight
                    // between an adjacent boundary cell that is less that one
                    if (!(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL || cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL))
                        hasValidBoundary = true;
                    else if (boundaryWeights != nullptr)
                    {
                        Vec2i face = cellToFace(cell, axis, direction);
                        if ((*boundaryWeights)(face, axis) < 1 && cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
                            hasValidBoundary = true;
                    }
                }

            EXPECT_TRUE(hasValidBoundary);
        }
    });
}

std::pair<UniformGrid<CellLabels>, VectorGrid<double>> buildTestDomain(bool useComplexDomain, bool useSolidSphere, int gridSize)
{
    UniformGrid<CellLabels> baseDomainCellLabels;
    VectorGrid<double> baseBoundaryWeights;
    // Complex domain set up
    if (useComplexDomain)
        buildComplexDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, useSolidSphere);
    // Simple domain set up
    else
        buildSimpleDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, 1 /*dirichlet band*/);

    std::pair<UniformGrid<CellLabels>, VectorGrid<double>> grids;

    std::pair<Vec2i, int> mgSettings = buildExpandedDomain(grids.first, grids.second, baseDomainCellLabels, baseBoundaryWeights);

    return grids;
}

TEST(GEOMETRIC_MULTIGRID_TESTS, BOUNDARY_CELLS_TEST)
{
    std::vector<bool> complexDomainSettings = {false, true};
    std::vector<bool> solidSphereSettings = {false, true};

    const int startGrid = 16;
    const int endGrid = startGrid * int(pow(2, 5));

    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
        for (bool useComplexDomain : complexDomainSettings)
        {
            std::pair<UniformGrid<CellLabels>, VectorGrid<double>> grids;
            if (useComplexDomain)
            {
                for (bool useSolidSphere : solidSphereSettings)
                {
                    grids = buildTestDomain(useComplexDomain, useSolidSphere, gridSize);
                }
            }
            else
            {
                grids = buildTestDomain(useComplexDomain, false, gridSize);
            }

            testBoundaryCells(grids.first, &grids.second);
        }
}

static void testExteriorCells(const UniformGrid<CellLabels>& cellLabels)
{
    for (int axis : {0, 1})
        for (int direction : {0, 1})
        {
            Vec2i startCell = Vec2i::Zero();
            Vec2i endCell = cellLabels.size();

            if (direction == 0)
                endCell[axis] = 1;
            else
                startCell[axis] = endCell[axis] - 1;

            forEachVoxelRange(startCell, endCell, [&](const Vec2i& cell) {
                EXPECT_TRUE(cellLabels(cell) == CellLabels::EXTERIOR_CELL);
            });
        }
}

TEST(GEOMETRIC_MULTIGRID_TESTS, EXTERIOR_CELLS_TEST)
{
    std::vector<bool> complexDomainSettings = {false, true};
    std::vector<bool> solidSphereSettings = {false, true};

    const int startGrid = 16;
    const int endGrid = startGrid * int(pow(2, 5));

    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
        for (bool useComplexDomain : complexDomainSettings)
        {
            std::pair<UniformGrid<CellLabels>, VectorGrid<double>> grids;
            if (useComplexDomain)
            {
                for (bool useSolidSphere : solidSphereSettings)
                {
                    grids = buildTestDomain(useComplexDomain, useSolidSphere, gridSize);
                }
            }
            else
            {
                grids = buildTestDomain(useComplexDomain, false, gridSize);
            }

            testExteriorCells(grids.first);
        }
}

static void testCoarsening(const UniformGrid<CellLabels>& coarseCellLabels, const UniformGrid<CellLabels>& fineCellLabels)
{
    // The coarse cell grid must be exactly have the size of the fine cell grid.
    ASSERT_TRUE((2 * coarseCellLabels.size()).eval() == fineCellLabels.size());

    ASSERT_TRUE(coarseCellLabels.size()[0] % 2 == 0);
    ASSERT_TRUE(coarseCellLabels.size()[1] % 2 == 0);
    ASSERT_TRUE(fineCellLabels.size()[0] % 2 == 0);
    ASSERT_TRUE(fineCellLabels.size()[1] % 2 == 0);

    forEachVoxelRange(Vec2i::Zero(), fineCellLabels.size(), [&](const Vec2i& fineCell) {
        Vec2i coarseCell = fineCell / 2;

        // If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
        if (fineCellLabels(fineCell) == CellLabels::DIRICHLET_CELL)
        {
            EXPECT_TRUE(coarseCellLabels(coarseCell) == CellLabels::DIRICHLET_CELL);
        }
        else if (fineCellLabels(fineCell) == CellLabels::INTERIOR_CELL || fineCellLabels(fineCell) == CellLabels::BOUNDARY_CELL)
        {
            // If the fine cell is active, the coarse cell cannot be exterior
            EXPECT_TRUE(coarseCellLabels(coarseCell) != CellLabels::EXTERIOR_CELL);
        }
    });

    forEachVoxelRange(Vec2i::Zero(), coarseCellLabels.size(), [&](const Vec2i& coarseCell)
    {
        bool foundDirichletChild = false;
        bool foundActiveChild = false;
        bool foundExteriorChild = false;

        for (int childIndex = 0; childIndex < 4; ++childIndex)
        {
            Vec2i fineCell = getChildCell(coarseCell, childIndex);

            CellLabels fineLabel = fineCellLabels(fineCell);

            if (fineLabel == CellLabels::DIRICHLET_CELL)
                foundDirichletChild = true;
            else if (fineLabel == CellLabels::INTERIOR_CELL || fineLabel == CellLabels::BOUNDARY_CELL)
                foundActiveChild = true;
            else if (fineLabel == CellLabels::EXTERIOR_CELL)
                foundExteriorChild = true;
        }

        CellLabels coarseLabel = coarseCellLabels(coarseCell);
        if (coarseLabel == CellLabels::DIRICHLET_CELL)
        {
            EXPECT_TRUE(foundDirichletChild);
        }
        else if (coarseLabel == CellLabels::INTERIOR_CELL || coarseLabel == CellLabels::BOUNDARY_CELL)
        {
            EXPECT_FALSE(foundDirichletChild);
            EXPECT_TRUE(foundActiveChild);
        }
        else if (coarseLabel == CellLabels::EXTERIOR_CELL)
        {
            EXPECT_FALSE(foundDirichletChild);
            EXPECT_FALSE(foundActiveChild);
            EXPECT_TRUE(foundExteriorChild);
        }
    });
}

TEST(GEOMETRIC_MULTIGRID_TESTS, COARSENING_TEST)
{
    std::vector<bool> complexDomainSettings = {false, true};
    std::vector<bool> solidSphereSettings = {false, true};

    const int startGrid = 32;
    const int endGrid = startGrid * int(pow(2, 5));

    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
        for (bool useComplexDomain : complexDomainSettings)
        {
            std::pair<UniformGrid<CellLabels>, VectorGrid<double>> grids;
            if (useComplexDomain)
            {
                for (bool useSolidSphere : solidSphereSettings)
                {
                    grids = buildTestDomain(useComplexDomain, useSolidSphere, gridSize);
                }
            }
            else
            {
                grids = buildTestDomain(useComplexDomain, false, gridSize);
            }

            UniformGrid<CellLabels> coarseCell = buildCoarseCellLabels(grids.first);

            testCoarsening(coarseCell, grids.first);
        }
}

static void geometricVectorTest(const UniformGrid<CellLabels>& cellLabels)
{
    VectorXd vecA = VectorXd::Random(cellLabels.voxelCount()); 
    VectorXd vecB = VectorXd::Random(cellLabels.voxelCount()); 

    UniformGrid<double> gridA(cellLabels.size(), 0);
    UniformGrid<double> gridB(cellLabels.size(), 0);

    tbb::parallel_for(tbb::blocked_range<int>(0, cellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec2i cell = cellLabels.unflatten(cellIndex);

            if (cellLabels(cell) == CellLabels::INTERIOR_CELL || cellLabels(cell) == CellLabels::BOUNDARY_CELL)
            {
                gridA(cell) = vecA(cellIndex);
                gridB(cell) = vecB(cellIndex);
            }
            else
            {
                vecA(cellIndex) = 0;
                vecB(cellIndex) = 0;
            }
        }
    });

    // Test dot product
    {
        double vecDot = vecA.dot(vecB);
        double gridDot = dotProduct(gridA, gridB, cellLabels);

        EXPECT_TRUE(isNearlyEqual(vecDot, gridDot, 1e-10, true));
    }

    // Test L2 norm
    {
        double vecAL2 = vecA.squaredNorm();
        double gridAL2 = squaredl2Norm(gridA, cellLabels);

        EXPECT_TRUE(isNearlyEqual(vecAL2, gridAL2, 1e-10, true));
    }

    {
        double vecBL2 = vecB.squaredNorm();
        double gridBL2 = squaredl2Norm(gridB, cellLabels);

        EXPECT_TRUE(isNearlyEqual(vecBL2, gridBL2, 1e-10, true));
    }

    // Test L-infinity norm
    {
        double vecALinf = vecA.lpNorm<Eigen::Infinity>();
        double gridALinf = lInfinityNorm(gridA, cellLabels);

        EXPECT_EQ(vecALinf, gridALinf);
    }

    {
        double vecBLinf = vecB.lpNorm<Eigen::Infinity>();
        double gridBLinf = lInfinityNorm(gridB, cellLabels);

        EXPECT_EQ(vecBLinf, gridBLinf);
    }

    // Add vectors
    {
        double scale = 10;
        VectorXd vecC = vecA + scale * vecB;
        UniformGrid<double> gridC(cellLabels.size(), 0);

        addVectors(gridC, gridA, gridB, cellLabels, scale);

        forEachVoxelRange(Vec2i::Zero(), cellLabels.size(), [&](const Vec2i& cell) { 
            
            int cellIndex = cellLabels.flatten(cell);

            EXPECT_EQ(vecC(cellIndex), gridC(cell));
        });
    }
  
    // Add to vector
    {
        double scale = 20;
        vecA += scale * vecB;

        addToVector(gridA, gridB, cellLabels, scale);

        forEachVoxelRange(Vec2i::Zero(), cellLabels.size(), [&](const Vec2i& cell) {
            int cellIndex = cellLabels.flatten(cell);

            EXPECT_EQ(vecA(cellIndex), gridA(cell));
        });
    }
}

TEST(GEOMETRIC_MULTIGRID_TESTS, GEOMETRIC_VECTOR_TEST)
{
    std::vector<bool> complexDomainSettings = {false, true};
    std::vector<bool> solidSphereSettings = {false, true};

    const int startGrid = 32;
    const int endGrid = startGrid * int(pow(2, 5));

    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
        for (bool useComplexDomain : complexDomainSettings)
        {
            std::pair<UniformGrid<CellLabels>, VectorGrid<double>> grids;
            if (useComplexDomain)
            {
                for (bool useSolidSphere : solidSphereSettings)
                {
                    grids = buildTestDomain(useComplexDomain, useSolidSphere, gridSize);
                }
            }
            else
            {
                grids = buildTestDomain(useComplexDomain, false, gridSize);
            }

            geometricVectorTest(grids.first);
        }
}

static void symmetryTest(bool useComplexDomain, bool useSolidSphere, int gridSize)
{
    UniformGrid<CellLabels> domainCellLabels;
    VectorGrid<double> boundaryWeights;
    int mgLevels;
    {
        UniformGrid<CellLabels> baseDomainCellLabels;
        VectorGrid<double> baseBoundaryWeights;

        // Complex domain set up
        if (useComplexDomain)
            buildComplexDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, useSolidSphere);
        // Simple domain set up
        else
            buildSimpleDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, 1 /*dirichlet band*/);

        // Build expanded domain
        std::pair<Vec2i, int> mgSettings = buildExpandedDomain(domainCellLabels, boundaryWeights, baseDomainCellLabels, baseBoundaryWeights);

        mgLevels = mgSettings.second;
    }

    double dx = boundaryWeights.dx();

    UniformGrid<double> rhsA(domainCellLabels.size(), 0);
    UniformGrid<double> rhsB(domainCellLabels.size(), 0);

    {
        VectorXd randVecA = VectorXd::Random(domainCellLabels.voxelCount());
        VectorXd randVecB = VectorXd::Random(domainCellLabels.voxelCount());

        ASSERT_GT((randVecA - randVecB).norm(), 0.);

        tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {

            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec2i cell = domainCellLabels.unflatten(cellIndex);

                if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
                {
                    rhsA(cell) = randVecA(cellIndex);
                    rhsB(cell) = randVecB(cellIndex);
                }
            }
        });
    }

    Transform xform(dx, Vec2d::Zero());
    
    // Jacobi smoother symmetry test
    {
        UniformGrid<double> solutionA(domainCellLabels.size(), 0);
        UniformGrid<double> solutionB(domainCellLabels.size(), 0);

        VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

        boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
        boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

        interiorJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
        interiorJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

        boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
        boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);

        double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
        double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

        EXPECT_TRUE(isNearlyEqual(dotA, dotB, 1e-10, true));
    }

    // Test direct solve symmetry
    {
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
        Eigen::SparseMatrix<double> sparseMatrix;

        int interiorCellCount = 0;
        UniformGrid<int> directSolverIndices(domainCellLabels.size(), -1);
        {
            forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i& cell) {
                if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
                    directSolverIndices(cell) = interiorCellCount++;
            });

            // Build rows
            std::vector<Eigen::Triplet<double>> sparseElements;

            double gridScale = 1. / std::pow(dx, 2);
            forEachVoxelRange(Vec2i::Zero(), domainCellLabels.size(), [&](const Vec2i& cell) {
                if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
                {
                    int index = directSolverIndices(cell);
                    ASSERT_TRUE(index >= 0);
                    for (int axis : {0, 1})
                        for (int direction : {0, 1})
                        {
                            Vec2i adjacentCell = cellToCell(cell, axis, direction);

                            ASSERT_TRUE(domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL || domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

                            Vec2i face = cellToFace(cell, axis, direction);
                            ASSERT_TRUE(boundaryWeights(face, axis) == 1);

                            int adjacentIndex = directSolverIndices(adjacentCell);
                            ASSERT_TRUE(adjacentIndex >= 0);

                            sparseElements.emplace_back(index, adjacentIndex, -gridScale);
                        }
                    sparseElements.emplace_back(index, index, 4. * gridScale);
                }
                else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
                {
                    int index = directSolverIndices(cell);
                    ASSERT_TRUE(index >= 0);

                    double diagonal = 0;

                    for (int axis : {0, 1})
                        for (int direction : {0, 1})
                        {
                            Vec2i adjacentCell = cellToCell(cell, axis, direction);

                            if (domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
                            {
                                int adjacentIndex = directSolverIndices(adjacentCell);
                                ASSERT_TRUE(adjacentIndex >= 0);

                                Vec2i face = cellToFace(cell, axis, direction);
                                ASSERT_TRUE(boundaryWeights(face, axis) == 1);

                                sparseElements.emplace_back(index, adjacentIndex, -gridScale);
                                ++diagonal;
                            }
                            else if (domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
                            {
                                int adjacentIndex = directSolverIndices(adjacentCell);
                                ASSERT_TRUE(adjacentIndex >= 0);

                                Vec2i face = cellToFace(cell, axis, direction);
                                double weight = boundaryWeights(face, axis);

                                sparseElements.emplace_back(index, adjacentIndex, -gridScale * weight);
                                diagonal += weight;
                            }
                            else if (domainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
                            {
                                int adjacentIndex = directSolverIndices(adjacentCell);
                                ASSERT_TRUE(adjacentIndex == -1);

                                Vec2i face = cellToFace(cell, axis, direction);
                                double weight = boundaryWeights(face, axis);

                                diagonal += weight;
                            }
                            else
                            {
                                ASSERT_TRUE(domainCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL);
                                int adjacentIndex = directSolverIndices(adjacentCell);
                                ASSERT_TRUE(adjacentIndex == -1);

                                Vec2i face = cellToFace(cell, axis, direction);
                                ASSERT_TRUE(boundaryWeights(face, axis) == 0);
                            }
                        }

                    sparseElements.emplace_back(index, index, gridScale * diagonal);
                }
            });

            // Solve system
            sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
            sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());

            solver.compute(sparseMatrix);

            ASSERT_TRUE(solver.info() == Eigen::Success);
        }

        UniformGrid<double> solutionA(domainCellLabels.size(), 0);
        UniformGrid<double> solutionB(domainCellLabels.size(), 0);

        // Copy from grid to Eigen, solve, and copy back to grid
        {
            VectorXd rhsVectorA = VectorXd::Zero(interiorCellCount);
            VectorXd rhsVectorB = VectorXd::Zero(interiorCellCount);

            tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
                for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
                {
                    Vec2i cell = domainCellLabels.unflatten(cellIndex);

                    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
                    {
                        int index = directSolverIndices(cell);
                        ASSERT_TRUE(index >= 0);

                        rhsVectorA(index) = rhsA(cell);
                        rhsVectorB(index) = rhsB(cell);
                    }
                }
            });

            VectorXd directSolutionA = solver.solve(rhsVectorA);
            VectorXd directSolutionB = solver.solve(rhsVectorB);

            // Copy solution back
            tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
                for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
                {
                    Vec2i cell = domainCellLabels.unflatten(cellIndex);

                    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
                    {
                        int index = directSolverIndices(cell);
                        ASSERT_TRUE(index >= 0);

                        solutionA(cell) = directSolutionA(index);
                        solutionB(cell) = directSolutionB(index);
                    }
                }
            });
        }

        // Compute dot products
        double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
        double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

        EXPECT_TRUE(isNearlyEqual(dotA, dotB, 1e-10, true));
    }
    {
        // Test down and up sampling
        UniformGrid<CellLabels> coarseDomainLabels = buildCoarseCellLabels(domainCellLabels);

        ASSERT_TRUE(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(domainCellLabels, &boundaryWeights));
        ASSERT_TRUE(unitTestExteriorCells(coarseDomainLabels) && unitTestExteriorCells(domainCellLabels));
        ASSERT_TRUE(unitTestCoarsening(coarseDomainLabels, domainCellLabels));

        UniformGrid<double> coarseRhs(coarseDomainLabels.size(), 0);

        UniformGrid<double> solutionA(domainCellLabels.size(), 0);

        {
            downsample(coarseRhs, rhsA, coarseDomainLabels, domainCellLabels);
            upsampleAndAdd(solutionA, coarseRhs, domainCellLabels, coarseDomainLabels);
        }

        UniformGrid<double> solutionB(domainCellLabels.size(), 0);

        {
            downsample(coarseRhs, rhsB, coarseDomainLabels, domainCellLabels);
            upsampleAndAdd(solutionB, coarseRhs, domainCellLabels, coarseDomainLabels);
        }

        // Compute dot products
        double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
        double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

        EXPECT_TRUE(isNearlyEqual(dotA, dotB, 1e-10, true));
    }

    // Test single level correction
    {
        UniformGrid<CellLabels> coarseDomainLabels = buildCoarseCellLabels(domainCellLabels);

        ASSERT_TRUE(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(domainCellLabels, &boundaryWeights));
        ASSERT_TRUE(unitTestExteriorCells(coarseDomainLabels) && unitTestExteriorCells(domainCellLabels));
        ASSERT_TRUE(unitTestCoarsening(coarseDomainLabels, domainCellLabels));

        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> coarseSolver;
        Eigen::SparseMatrix<double> sparseMatrix;

        // Pre-build matrix at the coarsest level
        int interiorCellCount = 0;
        UniformGrid<int> directSolverIndices(coarseDomainLabels.size(), -1);
        {
            forEachVoxelRange(Vec2i::Zero(), coarseDomainLabels.size(), [&](const Vec2i& cell) {
                if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL || coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
                    directSolverIndices(cell) = interiorCellCount++;
            });

            // Build rows
            std::vector<Eigen::Triplet<double>> sparseElements;

            double gridScale = 1. / std::pow(2. * dx, 2);
            forEachVoxelRange(Vec2i::Zero(), coarseDomainLabels.size(), [&](const Vec2i& cell) {
                if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
                {
                    int index = directSolverIndices(cell);
                    ASSERT_TRUE(index >= 0);
                    for (int axis : {0, 1})
                        for (int direction : {0, 1})
                        {
                            Vec2i adjacentCell = cellToCell(cell, axis, direction);

                            auto adjacentLabels = coarseDomainLabels(adjacentCell);
                            ASSERT_TRUE(adjacentLabels == CellLabels::INTERIOR_CELL || adjacentLabels == CellLabels::BOUNDARY_CELL);

                            int adjacentIndex = directSolverIndices(adjacentCell);
                            ASSERT_TRUE(adjacentIndex >= 0);

                            sparseElements.emplace_back(index, adjacentIndex, -gridScale);
                        }

                    sparseElements.emplace_back(index, index, 4. * gridScale);
                }
                else if (coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
                {
                    double diagonal = 0;
                    int index = directSolverIndices(cell);
                    ASSERT_TRUE(index >= 0);
                    for (int axis : {0, 1})
                        for (int direction : {0, 1})
                        {
                            Vec2i adjacentCell = cellToCell(cell, axis, direction);

                            auto cellLabels = coarseDomainLabels(adjacentCell);
                            if (cellLabels == CellLabels::INTERIOR_CELL || cellLabels == CellLabels::BOUNDARY_CELL)
                            {
                                int adjacentIndex = directSolverIndices(adjacentCell);
                                ASSERT_TRUE(adjacentIndex >= 0);

                                sparseElements.emplace_back(index, adjacentIndex, -gridScale);
                                ++diagonal;
                            }
                            else if (cellLabels == CellLabels::DIRICHLET_CELL)
                                ++diagonal;
                        }

                    sparseElements.emplace_back(index, index, diagonal * gridScale);
                }
            });

            sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
            sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());

            coarseSolver.compute(sparseMatrix);

            ASSERT_TRUE(coarseSolver.info() == Eigen::Success);
        }

        // Transfer fine rhs to coarse rhs as if it was a residual with a zero initial guess
        UniformGrid<double> solutionA(domainCellLabels.size(), 0);
        UniformGrid<double> solutionB(domainCellLabels.size(), 0);

        {
            // Pre-smooth to get an initial guess
            VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

            // Test Jacobi symmetry
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
                boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);
            }

            interiorJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
            interiorJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

            for (int iteration = 0; iteration < 3; ++iteration)
            {
                boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
                boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);
            }

            // Compute new residual
            UniformGrid<double> residualA(domainCellLabels.size(), 0);
            UniformGrid<double> residualB(domainCellLabels.size(), 0);

            computePoissonResidual(residualA, solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
            computePoissonResidual(residualB, solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

            UniformGrid<double> coarseRhsA(coarseDomainLabels.size(), 0);
            UniformGrid<double> coarseRhsB(coarseDomainLabels.size(), 0);
            
            downsample(coarseRhsA, residualA, coarseDomainLabels, domainCellLabels);
            downsample(coarseRhsB, residualB, coarseDomainLabels, domainCellLabels);

            VectorXd coarseRHSVectorA = VectorXd::Zero(interiorCellCount);
            VectorXd coarseRHSVectorB = VectorXd::Zero(interiorCellCount);

            // Copy to Eigen and direct solve
            tbb::parallel_for(tbb::blocked_range<int>(0, coarseDomainLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
                for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
                {
                    Vec2i cell = coarseDomainLabels.unflatten(cellIndex);

                    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL || coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
                    {
                        int index = directSolverIndices(cell);
                        ASSERT_TRUE(index >= 0);

                        coarseRHSVectorA(index) = coarseRhsA(cell);
                        coarseRHSVectorB(index) = coarseRhsB(cell);
                    }
                }
            });

            UniformGrid<double> coarseSolutionA(coarseDomainLabels.size(), 0);
            UniformGrid<double> coarseSolutionB(coarseDomainLabels.size(), 0);

            VectorXd directSolutionA = coarseSolver.solve(coarseRHSVectorA);
            VectorXd directSolutionB = coarseSolver.solve(coarseRHSVectorB);

            // Copy solution back
            tbb::parallel_for(tbb::blocked_range<int>(0, coarseDomainLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
                for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
                {
                    Vec2i cell = coarseDomainLabels.unflatten(cellIndex);

                    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL || coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
                    {
                        int index = directSolverIndices(cell);
                        ASSERT_TRUE(index >= 0);

                        coarseSolutionA(cell) = directSolutionA(index);
                        coarseSolutionB(cell) = directSolutionB(index);
                    }
                }
            });

            upsampleAndAdd(solutionA, coarseSolutionA, domainCellLabels, coarseDomainLabels);
            upsampleAndAdd(solutionB, coarseSolutionB, domainCellLabels, coarseDomainLabels);

            for (int iteration = 0; iteration < 3; ++iteration)
            {
                boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
                boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);
            }

            interiorJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
            interiorJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);

            for (int iteration = 0; iteration < 3; ++iteration)
            {
                boundaryJacobiPoissonSmoother(solutionA, rhsA, domainCellLabels, boundaryCells, dx, &boundaryWeights);
                boundaryJacobiPoissonSmoother(solutionB, rhsB, domainCellLabels, boundaryCells, dx, &boundaryWeights);
            }
        }

        double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
        double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

        EXPECT_TRUE(isNearlyEqual(dotA, dotB, 1e-10, true));
    }

    // Test multigrid preconditioner symmetry
    {
        // Pre-build multigrid preconditioner
        GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx);

        UniformGrid<double> solutionA(domainCellLabels.size(), 0);
        mgSolver.applyMGVCycle(solutionA, rhsA);
        mgSolver.applyMGVCycle(solutionA, rhsA, true);
        mgSolver.applyMGVCycle(solutionA, rhsA, true);
        mgSolver.applyMGVCycle(solutionA, rhsA, true);

        UniformGrid<double> solutionB(domainCellLabels.size(), 0);
        mgSolver.applyMGVCycle(solutionB, rhsB);
        mgSolver.applyMGVCycle(solutionB, rhsB, true);
        mgSolver.applyMGVCycle(solutionB, rhsB, true);
        mgSolver.applyMGVCycle(solutionB, rhsB, true);

        double dotA = dotProduct(solutionA, rhsB, domainCellLabels);
        double dotB = dotProduct(solutionB, rhsA, domainCellLabels);

        EXPECT_TRUE(isNearlyEqual(dotA, dotB, 1e-10, true));
    }
}

TEST(GEOMETRIC_MULTIGRID_TESTS, SYMMETRY_TEST)
{
    std::vector<bool> complexDomainSettings = {false, true};
    std::vector<bool> solidSphereSettings = {false, true};

    const int startGrid = 16;
    const int endGrid = startGrid * int(pow(2, 5));

    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
        for (bool useComplexDomain : complexDomainSettings)
        {
            if (useComplexDomain)
            {
                for (bool useSolidSphere : solidSphereSettings)
                {
                    symmetryTest(useComplexDomain, useSolidSphere, gridSize);
                }
            }
            else
            {
                symmetryTest(useComplexDomain, false, gridSize);
            }
        }
}

static void smootherTest(bool useComplexDomain, bool useSolidSphere, int gridSize)
{
    UniformGrid<CellLabels> domainCellLabels;
    VectorGrid<double> boundaryWeights;
    int mgLevels;
    Vec2i exteriorOffset;
    {
        UniformGrid<CellLabels> baseDomainCellLabels;
        VectorGrid<double> baseBoundaryWeights;

        // Complex domain set up
        if (useComplexDomain)
            buildComplexDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, useSolidSphere);
        // Simple domain set up
        else
            buildSimpleDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, 1 /*dirichlet band*/);

        // Build expanded domain
        std::pair<Vec2i, int> mgSettings = buildExpandedDomain(domainCellLabels, boundaryWeights, baseDomainCellLabels, baseBoundaryWeights);

        exteriorOffset = mgSettings.first;
        mgLevels = mgSettings.second;
    }

    double dx = boundaryWeights.dx();

    UniformGrid<double> rhsGrid(domainCellLabels.size(), 0);
    UniformGrid<double> solutionGrid(domainCellLabels.size(), 0);
    UniformGrid<double> residualGrid(domainCellLabels.size(), 0);

    {
        VectorXd randVec = VectorXd::Random(domainCellLabels.voxelCount());

        tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec2i cell = domainCellLabels.unflatten(cellIndex);

                if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
                {
                    solutionGrid(cell) = randVec(cellIndex);
                }
            }
        });
    }

    // Set delta function
    double deltaPercent = .1;
    double deltaAmplitude = 1000;
    Vec2i deltaPoint = (deltaPercent * Vec2d(gridSize, gridSize)).cast<int>() + exteriorOffset;
    
    forEachVoxelRange(deltaPoint - Vec2i::Ones(), deltaPoint + Vec2i(2, 2), [&](const Vec2i& cell)
    {
        if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL || domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
            rhsGrid(cell) = deltaAmplitude;
    });

    VecVec2i boundaryCells = buildBoundaryCells(domainCellLabels, 3);

    int maxIterations = 10;
    
    computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

    double oldL2Residual = std::sqrt(squaredl2Norm(residualGrid, domainCellLabels));

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        boundaryJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, boundaryCells, dx, &boundaryWeights);

        interiorJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

        boundaryJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, boundaryCells, dx, &boundaryWeights);

        computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

        double l2Residual = std::sqrt(squaredl2Norm(residualGrid, domainCellLabels));

        EXPECT_LT(l2Residual, oldL2Residual);
        oldL2Residual = l2Residual;
    }
}

TEST(GEOMETRIC_MULTIGRID_TESTS, SMOOTHER_TEST)
{
    std::vector<bool> complexDomainSettings = {false, true};
    std::vector<bool> solidSphereSettings = {false, true};

    const int startGrid = 32;
    const int endGrid = startGrid * int(pow(2, 5));

    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
        for (bool useComplexDomain : complexDomainSettings)
        {
            if (useComplexDomain)
            {
                for (bool useSolidSphere : solidSphereSettings)
                {
                    smootherTest(useComplexDomain, useSolidSphere, gridSize);
                }
            }
            else
            {
                smootherTest(useComplexDomain, false, gridSize);
            }
        }
}
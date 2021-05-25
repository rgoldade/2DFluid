#include <Eigen/Sparse>

#include "gtest/gtest.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim2D;

class AnalyticalViscositySolver
{
    static constexpr int UNASSIGNED = -1;

public:
    AnalyticalViscositySolver(const Transform& xform, const Vec2i& size)
        : myXform(xform)
    {
        myVelocityIndex = VectorGrid<int>(xform, size, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
    }

    // Returns the infinity-norm error of the numerical solution
    template<typename Initial, typename Solution, typename Viscosity>
    double solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity, const double dt);

    Vec2d cellIndexToWorld(const Vec2d& index) const { return myXform.indexToWorld(index + Vec2d(.5, .5)); }
    Vec2d nodeIndexToWorld(const Vec2d& index) const { return myXform.indexToWorld(index); }

private:
    Transform myXform;
    int setVelocityIndices();
    VectorGrid<int> myVelocityIndex;
};

template<typename Initial, typename Solution, typename Viscosity>
double AnalyticalViscositySolver::solve(const Initial& initialFunction, const Solution& solutionFunction, const Viscosity& viscosityFunction, const double dt)
{
    int velocityDOFCount = setVelocityIndices();

    // Build reduced system.
    // (Note we don't need control volumes since the cells are the same size and there is no free surface).
    // (I - dt * mu * D^T K D) u(n+1) = u(n)

    std::vector<Eigen::Triplet<double>> sparseMatrixElements;

    VectorXd rhsVector = VectorXd::Zero(velocityDOFCount);

    double dx = myVelocityIndex.dx();
    double baseCoeff = dt / std::pow(dx, 2);

    for (int axis : {0, 1})
    {
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>> parallelSparseElements;

        Vec2i size = myVelocityIndex.size(axis);

        tbb::parallel_for(tbb::blocked_range<int>(0, myVelocityIndex.grid(axis).voxelCount()), [&](const tbb::blocked_range<int>& range) {

            auto& localSparseElements = parallelSparseElements.local();

            for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
            {
                Vec2i face = myVelocityIndex.grid(axis).unflatten(faceIndex);

                int velocityIndex = myVelocityIndex(face, axis);

                if (velocityIndex >= 0)
                {
                    Vec2d facePosition = myVelocityIndex.indexToWorld(face.cast<double>(), axis);

                    rhsVector(velocityIndex) += initialFunction(facePosition, axis);

                    localSparseElements.emplace_back(velocityIndex, velocityIndex, 1);

                    // Build cell-centered stresses.
                    for (int cellDirection : {0, 1})
                    {
                        Vec2i cell = faceToCell(face, axis, cellDirection);

                        Vec2d cellPosition = cellIndexToWorld(cell.cast<double>());
                        double cellCoeff = 2. * viscosityFunction(cellPosition) * baseCoeff;

                        double divSign = (cellDirection == 0) ? -1 : 1;

                        for (int faceDirection : {0, 1})
                        {
                            Vec2i adjacentFace = cellToFace(cell, axis, faceDirection);

                            double gradSign = (faceDirection == 0) ? -1 : 1;

                            int faceRow = myVelocityIndex(adjacentFace, axis);
                            if (faceRow >= 0)
                                localSparseElements.emplace_back(velocityIndex, faceRow, -divSign * gradSign * cellCoeff);
                            // No solid boundary to deal with since faces on the boundary
                            // are not included.
                        }
                    }

                    // Build node stresses.
                    for (int nodeDirection : {0, 1})
                    {
                        Vec2i node = faceToNode(face, axis, nodeDirection);

                        double divSign = (nodeDirection == 0) ? -1 : 1;

                        Vec2d nodePosition = nodeIndexToWorld(node.cast<double>());
                        double nodeCoeff = viscosityFunction(nodePosition) * baseCoeff;

                        for (int gradientAxis : {0, 1})
                            for (int faceDirection : {0, 1})
                            {
                                Vec2i adjacentFace = nodeToFace(node, gradientAxis, faceDirection);

                                int faceAxis = (gradientAxis + 1) % 2;

                                double gradSign = (faceDirection == 0) ? -1 : 1;

                                // Check for out of bounds
                                if (faceDirection == 0 && adjacentFace[gradientAxis] < 0 || faceDirection == 1 && adjacentFace[gradientAxis] >= size[gradientAxis])
                                {
                                    Vec2d adjacentFacePosition = myVelocityIndex.indexToWorld(adjacentFace.cast<double>(), faceAxis);
                                    rhsVector(velocityIndex) += divSign * gradSign * nodeCoeff * solutionFunction(adjacentFacePosition, faceAxis);
                                }
                                // Check for on the bounds
                                else if (nodeDirection == 0 && adjacentFace[faceAxis] == 0 || nodeDirection == 1 && adjacentFace[faceAxis] == myVelocityIndex.size(faceAxis)[faceAxis] - 1)
                                {
                                    Vec2d adjacentFacePosition = myVelocityIndex.indexToWorld(adjacentFace.cast<double>(), faceAxis);
                                    rhsVector(velocityIndex) += divSign * gradSign * nodeCoeff * solutionFunction(adjacentFacePosition, faceAxis);
                                }
                                else
                                {
                                    int adjacentRow = myVelocityIndex(adjacentFace, faceAxis);
                                    ASSERT_TRUE(adjacentRow >= 0);

                                    localSparseElements.emplace_back(velocityIndex, adjacentRow, -divSign * gradSign * nodeCoeff);
                                }
                            }
                    }
                }
            }            
        });

        mergeLocalThreadVectors(sparseMatrixElements, parallelSparseElements);
    }

    Eigen::SparseMatrix<double> sparseMatrix(velocityDOFCount, velocityDOFCount);
    sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> solver;
    solver.compute(sparseMatrix);

    if (solver.info() != Eigen::Success)
    {
        return -1;
    }

    VectorXd solutionVector = solver.solve(rhsVector);

    if (solver.info() != Eigen::Success)
    {
        return -1;
    }

    double error = 0;

    for (int axis : {0, 1})
    {
        double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, myVelocityIndex.grid(axis).voxelCount()), double(0),
            [&](const tbb::blocked_range<int>& range, double error) -> double {
                for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                {
                    Vec2i face = myVelocityIndex.grid(axis).unflatten(faceIndex);

                    int velocityIndex = myVelocityIndex(face, axis);

                    if (velocityIndex >= 0)
                    {
                        Vec2d facePosition = myVelocityIndex.indexToWorld(face.cast<double>(), axis);
                        double localError = fabs(solutionVector(velocityIndex) - solutionFunction(facePosition, axis));

                        error = std::max(error, localError);
                    }
                }

                return error;
            },
            [](double a, double b) -> double {
                return std::max(a, b);
            });

        error = std::max(error, localError);
    }

    return error;
}

int AnalyticalViscositySolver::setVelocityIndices()
{
    // Loop over each face. If it's not along the boundary
    // then include it into the system.

    int index = 0;

    for (int axis : {0, 1})
    {
        Vec2i size = myVelocityIndex.size(axis);

        forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face) {
            // Faces along the boundary are removed from the simulation
            if (!(face[axis] == 0 || face[axis] == size[axis] - 1))
                myVelocityIndex(face, axis) = index++;
        });
    }
    // Returning index gives the number of velocity positions required for a linear system
    return index;
}

TEST(ANALYTICAL_VISCOSITY_TEST, CONVERGENCE_TEST)
{
    double dt = 1.;

    auto initial = [&](const Vec2d& pos, int axis) {
        double x = pos[0];
        double y = pos[1];
        double val;
        if (axis == 0)
            val = sin(x) * sin(y) - dt * (2. / PI * cos(x) * sin(y) + (cos(x + y) - 2 * sin(x) * sin(y)) * (x / PI + .5));
        else
            val = sin(x) * sin(y) - dt * ((cos(x) * cos(y) - 3. * sin(x) * sin(y)) * (x / PI + .5) + 1. / PI * sin(x + y));

        return val;
    };

    auto solution = [](const Vec2d& pos, int) { return sin(pos[0]) * sin(pos[1]); };
    auto viscosity = [](const Vec2d& pos) { return pos[0] / PI + .5; };

    const int startGrid = 16;
    const int endGrid = startGrid * int(pow(2, 4));

    std::vector<double> errors;
    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
    {
        const double dx = PI / double(gridSize);
        Vec2d origin = Vec2d::Zero();
        Vec2i size(std::round(PI / dx), std::round(PI / dx));
        Transform xform(dx, origin);

        AnalyticalViscositySolver solver(xform, size);
        double error = solver.solve(initial, solution, viscosity, dt);

        errors.push_back(error);
        EXPECT_GT(error, 0.);
    }

    for (int errorIndex = 1; errorIndex < errors.size(); ++errorIndex)
    {
        double errorRatio = errors[errorIndex - 1] / errors[errorIndex];
        EXPECT_GT(errorRatio, 4.);
    }
}

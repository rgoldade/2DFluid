#ifndef MULTI_MATERIAL_PRESSURE_PROJECTION_H
#define MULTI_MATERIAL_PRESSURE_PROJECTION_H

#include <Eigen/Sparse>

#include "ComputeWeights.h"
#include "GeometricMultigridOperators.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// MultiMaterialPressureProjection.h/cpp
// Ryan Goldade 2017
//
////////////////////////////////////

namespace FluidSim2D
{

class MultiMaterialPressureProjection
{
	using MGCellLabels = GeometricMultigridOperators::CellLabels;
	using SolveReal = double;
	using Vector = Eigen::VectorXd;

	const int UNSOLVED_CELL = -1;

public:
	MultiMaterialPressureProjection(const std::vector<LevelSet>& surfaces,
									const std::vector<double>& densities,
									const LevelSet& solidSurface);

	const VectorGrid<VisitedCellLabels>& getValidFaces(int material)
	{
		assert(material < myMaterialCount);
		return myValidMaterialFaces[material];
	}

	void project(std::vector<VectorGrid<double>>& velocities);

	void setInitialGuess(const ScalarGrid<double>& initialGuessPressure)
	{
		assert(myFluidSurfaces[0].isGridMatched(initialGuessPressure));
		myUseInitialGuessPressure = true;
		myInitialGuessPressure = &initialGuessPressure;
	}

	void disableInitialGuess()
	{
		myUseInitialGuessPressure = false;
	}

	ScalarGrid<double> getPressureGrid()
	{
		return myPressure;
	}

	void drawPressure(Renderer& renderer) const;

	void printPressure(const std::string& filename) const
	{
		myPressure.printAsOBJ(filename + ".obj");
	}

private:

	//
	// Helper functions for multigrid preconditioner
	//

	void copyToPreconditionerGrids(std::vector<UniformGrid<SolveReal>>& mgSourceGrid,
		UniformGrid<SolveReal>& smootherSourceGrid,
		const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
		const UniformGrid<int>& materialCellLabels,
		const UniformGrid<int>& solvableCellIndices,
		const Vector& sourceVector,
		const VecVec2i& mgExpandedOffset) const;

	void applyBoundarySmoothing(std::vector<SolveReal>& tempDestinationVector,
		const VecVec2i& boundarySmootherCells,
		const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
		const UniformGrid<int>& materialCellLabels,
		const UniformGrid<SolveReal>& smootherDestinationGrid,
		const UniformGrid<SolveReal>& smootherSourceGrid,
		const VecVec2i& mgExpandedOffset) const;

	void updateDestinationGrid(UniformGrid<SolveReal>& smootherDestinationGrid,
		const UniformGrid<int>& materialCellLabels,
		const VecVec2i& boundarySmootherCells,
		const std::vector<SolveReal>& tempDestinationVector) const;

	void applyDirichletToMG(std::vector<UniformGrid<SolveReal>>& mgSourceGrid,
		std::vector<UniformGrid<SolveReal>>& mgDestinationGrid,
		const UniformGrid<SolveReal>& smootherDestinationGrid,
		const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
		const UniformGrid<int>& materialCellLabels,
		const UniformGrid<int>& solvableCellIndices,
		const Vector& sourceVector,
		const VecVec2i& boundarySmootherCells,
		const VecVec2i& mgExpandedOffset) const;

	void updateSmootherGrid(UniformGrid<SolveReal>& smootherDestinationGrid,
		const std::vector<UniformGrid<SolveReal>>& mgDestinationGrid,
		const UniformGrid<int>& materialCellLabels,
		const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
		const VecVec2i& mgExpandedOffset) const;

	void copyMGSolutionToVector(Vector& destinationVector,
		const UniformGrid<int>& materialCellLabels,
		const UniformGrid<int>& solvableCellIndices,
		const std::vector<UniformGrid<MGCellLabels>>& mgDomainCellLabels,
		const std::vector<UniformGrid<SolveReal>>& mgDestinationGrid,
		const VecVec2i& mgExpandedOffset) const;

	void copyBoundarySolutionToVector(Vector& destinationVector,
		const UniformGrid<int>& materialCellLabels,
		const UniformGrid<int>& solvableCellIndices,
		const UniformGrid<SolveReal>& smootherDestinationGrid,
		const VecVec2i& boundarySmootherCells) const;

	ScalarGrid<double> myPressure;

	const std::vector<LevelSet>& myFluidSurfaces;
	const std::vector<double>& myFluidDensities;

	VectorGrid<double> mySolidCutCellWeights;
	std::vector<VectorGrid<double>> myMaterialCutCellWeights;
	std::vector<VectorGrid<VisitedCellLabels>> myValidMaterialFaces;

	const LevelSet& mySolidSurface;
	const int myMaterialCount;

	const ScalarGrid<double>* myInitialGuessPressure;
	bool myUseInitialGuessPressure;
};

}
#endif
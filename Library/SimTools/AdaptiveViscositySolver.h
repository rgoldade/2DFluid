#ifndef FLUIDSIM2D_ADAPTIVE_VISCOSITY_SOLVER_H
#define FLUIDSIM2D_ADAPTIVE_VISCOSITY_SOLVER_H

#include "LevelSet.h"
#include "QuadtreeGrid.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

namespace FluidSim2D
{

class AdaptiveViscositySolver
{
	struct StressStencilFace
	{
		StressStencilFace(int index, double value) : myIndex(index), myValue(value) {}
		int myIndex;
		double myValue;
	};

	struct FaceInterpolationWeights
	{
		FaceInterpolationWeights() : face(-Vec2i::Ones()), weight(0), level(-1) {}
		FaceInterpolationWeights(const Vec2i& face, double weight, int level = 0)
			: face(face), weight(weight), level(level) {}

		Vec2i face;
		double weight;
		int level;
	};

public:

	AdaptiveViscositySolver(const LevelSet& surface,
		const LevelSet& solidSurface,
		const VectorGrid<double>& solidVelocity,
		const ScalarGrid<double>& viscosity,
		int levels);

	void solve(VectorGrid<double>& velocity, double dt);

private:

	int buildVelocityIndices();
	int buildNodeStressIndices();
	int buildCenterIndices();

	void buildNodeStencils(std::vector<std::vector<StressStencilFace>>& nodeStressStencils,
		std::vector<std::vector<double>>& nodeBoundaryStencils,
		std::vector<double>& nodeWeights) const;

	void getNodeStressFaces(std::vector<StressStencilFace>& faces,
		std::vector<double>& boundaryFaces,
		const Vec2i& node, const int level) const;

	double getNodeIntegrationVolumes(const Vec2i& node, int level) const;

	void buildCenterStencils(std::vector<std::vector<StressStencilFace>>& centerStressStencils,
		std::vector<std::vector<double>>& centerBoundaryStencils,
		std::vector<double>& centerWeights, 
		int centerDOFCount);


	void getCenterStressFaces(std::vector<StressStencilFace>& faces,
		std::vector<double>& boundaryFaces,
		const Vec2i& cell, int level,
		int axis) const;

	double getVelocityControlVolumes(const Vec2i& face, int axis, int level) const;

	void applyToMatrix(std::vector<Eigen::Triplet<double>>& elements, VectorXd& rhs, double& diagonal, double coeff, int faceIndex,
		const std::vector<StressStencilFace>& faces,
		const std::vector<double>& boundary_faces);

	std::vector<VectorGrid<int>> myVelocityIndices;
	std::vector<ScalarGrid<int>> myNodeStressIndices;
	std::vector<ScalarGrid<int>> myCenterStressIndices;

	ScalarGrid<double> myCenterAreas;
	ScalarGrid<double> myNodeAreas;
	VectorGrid<double> myFaceAreas;

	QuadtreeGrid myQuadtree;

	const LevelSet& mySurface;
	const LevelSet& mySolidSurface;

	const VectorGrid<double> mySolidVelocity;

	std::vector<VectorGrid<double>> myVelocitSolution;

	const ScalarGrid<double> myViscosity;
};

}

#endif
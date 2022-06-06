#ifndef FLUIDSIM2D_QUADTREE_VECTOR_INTERPOLATOR_H
#define FLUIDSIM2D_QUADTREE_VECTOR_INTERPOLATOR_H


#include "QuadtreeGrid.h"
#include "VectorGrid.h"

namespace FluidSim2D
{

class QuadtreeVectorInterpolator
{
	const int UNASSIGNED = -1;
	const int COLLISION = -2;
	const int OUTSIDE = -3;

	enum NodeActivity { INACTIVENODE, ACTIVENODE, DEPENDENTNODE };

public:

	QuadtreeVectorInterpolator(const QuadtreeGrid& referenceTree,
		const std::vector<VectorGrid<int>>& referenceIndexGrid,
		const std::vector<VectorGrid<double>>& referenceValueGrid);

	double biLerp(const Vec2d& worldPos, int axis) const;

private:

	void setActiveNodes(int level);
	void sampleActiveNodes(VectorGrid<double>& nodeWeights, ScalarGrid<int>& nodeFlags, int level);
	void bubbleActiveNodeValues(std::vector<VectorGrid<double>>& nodeWeights, std::vector<ScalarGrid<int>>& nodeFlags, int level);
	void finishIncompleteNodes(std::vector<VectorGrid<double>>& nodeWeights, std::vector<ScalarGrid<int>>& nodeFlags, int level);
	void normalizeActiveNodes(VectorGrid<double>& nodeWeights, ScalarGrid<int>& nodeFlags, int level);
	void distributeNodeValues(int level);

	const QuadtreeGrid& myQuadtree;
	const std::vector<VectorGrid<double>>& myValueGrid;
	const std::vector<VectorGrid<int>>& myIndexGrid;

	std::vector<ScalarGrid<NodeActivity>> myNodeLabels;
	std::vector<VectorGrid<double>> myNodeValues;

	std::vector<Transform> myXforms;
};

}
#endif

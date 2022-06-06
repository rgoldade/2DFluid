#include "gtest/gtest.h"

#include "LevelSet.h"
#include "QuadtreeGrid.h"
#include "Utilities.h"

using namespace FluidSim2D;

class AnalyticalQuadtreeViscositySolver
{
	struct StressStencilFace
	{
		StressStencilFace(int index, double coeff) : myIndex(index), myCoeff(coeff) {}

		int myIndex;
		double myCoeff;
	};

	static constexpr int UNASSIGNED = -1;
	static constexpr int BOUNDARY = -2;

public:

	template<typename Refinement>
	AnalyticalQuadtreeViscositySolver(const Transform& xform, const Vec2i& size, int levels, Refinement refiner)
		: myTree(xform, size, levels)
	{
		// Build quadtree according to the grid spacing, size and refinement criteria
		myTree.buildTree(refiner);

		levels = myTree.levels();

		myVelocityIndices.resize(levels);
		myNodeStressIndices.resize(levels);
		myCenterStressIndices.resize(levels);

		myXforms.resize(levels);

		for (int level = 0; level < levels; ++level)
		{
			Transform localXform = myTree.xform(level);

			myXforms[level] = localXform;

			Vec2i localSize = myTree.size(level);
			myVelocityIndices[level] = VectorGrid<int>(localXform, localSize, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
			myNodeStressIndices[level] = ScalarGrid<int>(localXform, localSize, UNASSIGNED, ScalarGridSettings::SampleType::NODE);
			myCenterStressIndices[level] = ScalarGrid<int>(localXform, localSize, UNASSIGNED, ScalarGridSettings::SampleType::CENTER);
		}
	}

	void refine();

	template<typename Initial, typename Solution, typename Viscosity>
	Vec2d solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity, double dt);

private:

	int setVelocityIndices();
	int setNodeStressIndices();
	int setCenterStressIndices();

	template<typename Solution, typename Viscosity>
	void buildNodeStencils(std::vector<std::vector<StressStencilFace>>& nodeStressStencils,
							std::vector<std::vector<double>>& nodeBoundaryStencils,
							std::vector<double>& nodeWeights,
							const Solution& solution,
							const Viscosity& viscosity);

	template<typename Solution>
	void getNodeStressFaces(std::vector<StressStencilFace>& faces,
							std::vector<double>& boundaryFaces,
							const Solution& solution,
							const Vec2i& node, const int level) const;

	double getNodeIntegrationVolumes(const Vec2i& node, int level) const;

	template<typename Solution, typename Viscosity>
	void buildCenterStencils(std::vector<std::vector<StressStencilFace>>& centerStressStencils,
								std::vector<std::vector<double>>& centerBoundaryStencils,
								std::vector<double>& centerWeights,
								const Solution& solution,
								const Viscosity& viscosity,
								int dofCount);

	template<typename Solution>
	void getCenterStressFaces(std::vector<StressStencilFace>& faces,
								std::vector<double>& boundaryFaces,
								const Solution& solution,
								const Vec2i& cell, const int level,
								const int axis) const;

	void applyToMatrix(std::vector<Eigen::Triplet<double>>& elements, VectorXd& rhsVector, double& diagonal, double coeff, int faceIndex,
		const std::vector<StressStencilFace>& faces,
		const std::vector<double>& boundaryFaces) const;

	double getVelocityIntegrationVolumes(const Vec2i& face, int axis, int level) const;

	QuadtreeGrid myTree;

	std::vector<Transform> myXforms;
	std::vector<VectorGrid<int>> myVelocityIndices;
	std::vector<ScalarGrid<int>> myNodeStressIndices;
	std::vector<ScalarGrid<int>> myCenterStressIndices;
};

void AnalyticalQuadtreeViscositySolver::refine()
{
	myTree.refineGrid();

	myVelocityIndices.clear();

	int levels = myTree.levels();

	myVelocityIndices.resize(levels);
	myNodeStressIndices.resize(levels);
	myCenterStressIndices.resize(levels);

	myXforms.resize(levels);

	for (int level = 0; level < levels; ++level)
	{
		Transform localXform = myTree.xform(level);

		myXforms[level] = localXform;

		Vec2i localSize = myTree.size(level);
		myVelocityIndices[level] = VectorGrid<int>(localXform, localSize, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
		myNodeStressIndices[level] = ScalarGrid<int>(localXform, localSize, UNASSIGNED, ScalarGridSettings::SampleType::NODE);
		myCenterStressIndices[level] = ScalarGrid<int>(localXform, localSize, UNASSIGNED, ScalarGridSettings::SampleType::CENTER);
	}
}

int AnalyticalQuadtreeViscositySolver::setVelocityIndices()
{
	// Loop over each level of the grid. Loop over each face.
	// If the face has one ACTIVE adjacent cell and one DOWN adjacent cell OR 
	// has two ACTIVE adjacent cells, then the face is considered active.

	int index = 0;
	int levels = myTree.levels();

	for (int level = 0; level < levels; ++level)
	{
		for (int axis : {0, 1})
		{
			Vec2i size = myVelocityIndices[level].size(axis);

			forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face)
			{
				// Grab adjacent cells
				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				// Boundary check
				if (face[axis] == 0)
				{
					if (myTree.isCellActive(forwardCell, level))
					{
						myVelocityIndices[level](face, axis) = BOUNDARY;
					}
				}
				else if (face[axis] == size[axis] - 1)
				{
					if (myTree.isCellActive(backwardCell, level))
					{
						myVelocityIndices[level](face, axis) = BOUNDARY;
					}
				}
				else
				{
					QuadtreeGrid::QuadtreeCellLabel backwardCellLabel = myTree.getCellLabel(backwardCell, level);
					QuadtreeGrid::QuadtreeCellLabel forwardCellLabel = myTree.getCellLabel(forwardCell, level);

					if ((backwardCellLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardCellLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardCellLabel == QuadtreeGrid::QuadtreeCellLabel::UP && forwardCellLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardCellLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardCellLabel == QuadtreeGrid::QuadtreeCellLabel::UP))
					{
						myVelocityIndices[level](face, axis) = index++;
					}
				}
			});
		}
	}

	return index;
}

int AnalyticalQuadtreeViscositySolver::setNodeStressIndices()
{
	int index = 0;

	// Loop over active cells
	int levels = myTree.levels();

	for (int level = 0; level < levels; ++level)
	{
		Vec2i size = myNodeStressIndices[level].size();

		forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& node)
		{
			// Loop over adjacent cells and check for activity.
			bool indexed = false;

			for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
			{
				Vec2i cell = nodeToCell(node, cellIndex);

				if (cell[0] < 0 || cell[1] < 0 ||
					cell[0] >= myTree.size(level)[0] ||
					cell[1] >= myTree.size(level)[1])
				{
					continue;
				}

				// If the grid points down then it is not in the system
				if (myTree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::DOWN)
				{
					indexed = false;
					break;
				}
				else if (myTree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
				{
					indexed = true;
				}
			}

			if (indexed)
			{
				myNodeStressIndices[level](node) = index++;
			}
		});
	}

	// Returning the index gives the number of stress positions required in a linear system
	return index;
}

int AnalyticalQuadtreeViscositySolver::setCenterStressIndices()
{
	int levels = myTree.levels();

	int index = 0;
	for (int level = 0; level < levels; ++level)
	{
		Vec2i size = myTree.size(level);
		forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
		{
			if (myTree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
			{
				myCenterStressIndices[level](cell) = index++;
			}
		});
	}

	return index;
}

template<typename Initial, typename Solution, typename Viscosity>
Vec2d AnalyticalQuadtreeViscositySolver::solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity, double dt)
{
	// Set active faces according to active cells
	int velocityDOFCount = setVelocityIndices();
	int nodeDOFCount = setNodeStressIndices();
	int centerDOFCount = setCenterStressIndices();

	// Build stress gradients
	std::vector<std::vector<StressStencilFace>> nodeStressStencils(nodeDOFCount);
	std::vector<std::vector<double>> nodeBoundaryStencils(nodeDOFCount);
	std::vector<double> nodeWeights(nodeDOFCount);

	buildNodeStencils(nodeStressStencils, nodeBoundaryStencils, nodeWeights,
		solution, viscosity);

	// We need 2 times the number of center stresses because there are separate stencils for du/dx
	// and dv/dy co-located at the center but each center is not given two indices.
	std::vector<std::vector<StressStencilFace>> centerStressStencils(2 * centerDOFCount);
	std::vector<std::vector<double>> centerBoundaryStencils(2 * centerDOFCount);
	std::vector<double> centerWeights(centerDOFCount);

	buildCenterStencils(centerStressStencils, centerBoundaryStencils, centerWeights,
		solution, viscosity, centerDOFCount);

	int levels = myTree.levels();

	std::vector<Eigen::Triplet<double>> elements(7 * velocityDOFCount);
	
	double baseCoeff = 2 * dt;

	VectorXd rhsVector = VectorXd::Zero(velocityDOFCount);
	VectorXd initialGuessVector = VectorXd::Zero(velocityDOFCount);

	for (int level = 0; level < levels; ++level)
	{
		for (int axis = 0; axis < 2; ++axis)
		{
			Vec2i size = myVelocityIndices[level].size(axis);

			int cellOffset = centerDOFCount * axis;

			forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face)
			{
				int velocityIndex = myVelocityIndices[level](face, axis);
				if (velocityIndex >= 0)
				{
					double diagonalElement = 0.;

					// Build cell centered-stress and dangling node stresses
					for (int cellDirection = 0; cellDirection < 2; ++cellDirection)
					{
						Vec2i cell = faceToCell(face, axis, cellDirection);

						Vec2i stressCell;
						int stressLevel;

						if (myTree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
						{
							stressCell = cell;
							stressLevel = level;
						}
						// Since we're face graded, we can safely assume that if the cell
						// isn't active then its parent must be.
						else
						{
							assert(myTree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::UP);
							Vec2i parent = myTree.getParentCell(cell);

							assert(myTree.isCellActive(parent, level + 1));

							stressCell = parent;
							stressLevel = level + 1;
						}

						{
							int cellIndex = myCenterStressIndices[stressLevel](stressCell);

							assert(cellIndex >= 0);

							const auto& faces = centerStressStencils[cellIndex + cellOffset];
							assert(faces.size() > 0);

							const auto& boundaryFaces = centerBoundaryStencils[cellIndex + cellOffset];

							double coeff = baseCoeff * centerWeights[cellIndex];

							applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);
						}

						// Search for dangling nodes in the orthogonal direction
						for (int faceDirection = 0; faceDirection < 2; ++faceDirection)
						{
							int otherAxis = (axis + 1) % 2;
							Vec2i adjacentFace = cellToFace(stressCell, otherAxis, faceDirection);

							if (myVelocityIndices[stressLevel](adjacentFace, otherAxis) == UNASSIGNED)
							{
								Vec2i node = myTree.getInsetNode(adjacentFace, otherAxis);

								int nodeIndex = myNodeStressIndices[stressLevel - 1](node);
								if (nodeIndex >= 0)
								{
									const auto& faces = nodeStressStencils[nodeIndex];
									assert(faces.size() > 0);

									const auto& boundaryFaces = nodeBoundaryStencils[nodeIndex];

									double coeff = baseCoeff * nodeWeights[nodeIndex];

									applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);

								}
							}
						}
					}

					// Build velocity mappings from adjacent nodes
					for (int nodeDirection = 0; nodeDirection < 2; ++nodeDirection)
					{
						Vec2i node = faceToNode(face, axis, nodeDirection);

						int nodeIndex = myNodeStressIndices[level](node);
						if (nodeIndex >= 0)
						{
							const auto& faces = nodeStressStencils[nodeIndex];
							assert(faces.size() > 0);

							const auto& boundaryFaces = nodeBoundaryStencils[nodeIndex];

							double coeff = baseCoeff * nodeWeights[nodeIndex];

							applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);
						}
						else if (nodeIndex == UNASSIGNED)
						{
							Vec2i childnode = myTree.getChildNode(node);
							int childNodeIndex = myNodeStressIndices[level - 1](childnode);

							if (childNodeIndex >= 0)
							{
								const auto& faces = nodeStressStencils[childNodeIndex];
								assert(faces.size() > 0);

								const auto& boundaryFaces = nodeBoundaryStencils[childNodeIndex];

								double coeff = baseCoeff * nodeWeights[childNodeIndex];

								applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);

							}
						}
					}

					double velocityVolume = getVelocityIntegrationVolumes(face, axis, level);
					elements.emplace_back(velocityIndex, velocityIndex, velocityVolume + diagonalElement);

					Vec2d facePos = myVelocityIndices[level].indexToWorld(face.cast<double>(), axis);
					rhsVector[velocityIndex] += velocityVolume * initial(facePos, axis);

					initialGuessVector[velocityIndex] = initial(facePos, axis);
				}
			});
		}
	}

	SparseMatrix sparseMatrix(velocityDOFCount, velocityDOFCount);
	sparseMatrix.setFromTriplets(elements.begin(), elements.end());

	Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper | Eigen::Lower> solver;
	solver.compute(sparseMatrix);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to build" << std::endl;
		return Vec2d::Constant(std::numeric_limits<double>::max());
	}

	solver.setTolerance(1E-10);

	VectorXd solutionVector = solver.solveWithGuess(rhsVector, initialGuessVector);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to converge" << std::endl;
		return Vec2d::Constant(std::numeric_limits<double>::max());
	}
	else
	{
		std::cout << "    Solver iterations:     " << solver.iterations() << std::endl;
		std::cout << "    Solver error: " << solver.error() << std::endl;
	}

	Vec2d error = Vec2d::Zero();
	for (int level = 0; level < levels; ++level)
	{
		for (int axis = 0; axis < 2; ++axis)
		{
			Vec2i size = myVelocityIndices[level].size(axis);

			forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face)
			{
				int velocityIndex = myVelocityIndices[level](face, axis);

				if (velocityIndex >= 0)
				{
					Vec2d pos = myVelocityIndices[level].indexToWorld(face.cast<double>(), axis);
					double localError = std::fabs(solutionVector(velocityIndex) - solution(pos, axis));

					if (error[axis] < localError)
					{
						error[axis] = localError;
					}
				}
			});
		}
	}

	return error;
}

template<typename Solution, typename Viscosity>
void AnalyticalQuadtreeViscositySolver::buildNodeStencils(std::vector<std::vector<StressStencilFace>>& nodeStressStencils,
															std::vector<std::vector<double>>& nodeBoundaryStencils,
															std::vector<double>& nodeWeights,
															const Solution& solution,
															const Viscosity& viscosity)
{
	int levels = myTree.levels();

	for (int level = 0; level < levels; ++level)
	{
		Vec2i size = myNodeStressIndices[level].size();

		forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& node)
		{
			int index = myNodeStressIndices[level](node);

			if (index >= 0)
			{
				auto& faces = nodeStressStencils[index];
				auto& boundaries = nodeBoundaryStencils[index];

				getNodeStressFaces(faces, boundaries, solution, node, level);

				double nodeWeight = getNodeIntegrationVolumes(node, level);

				// Incorporate velocity into the weights
				Vec2d pos = myNodeStressIndices[level].indexToWorld(node.cast<double>());
				nodeWeight *= viscosity(pos);

				// Account for diagonal K matrix
				nodeWeights[index] = 2. * nodeWeight;
			}
		});
	}
}

double AnalyticalQuadtreeViscositySolver::getNodeIntegrationVolumes(const Vec2i& node, int level) const
{
	int nodeIndex = myNodeStressIndices[level](node);
	assert(nodeIndex >= 0 || nodeIndex == BOUNDARY);

	// Build the dx per axis
	double dx = myVelocityIndices[level].dx();
	Vec2d ndx = Vec2d::Zero();

	for (int faceAxis : {0, 1})
	{
		int gradientAxis = (faceAxis + 1) % 2;
		for (int direction : {0, 1})
		{
			Vec2i face = nodeToFace(node, gradientAxis, direction);

			if (face[gradientAxis] < 0 || face[gradientAxis] >= myVelocityIndices[level].size(faceAxis)[gradientAxis])
			{
				ndx[gradientAxis] += .5 * dx;
			}
			else
			{
				// We don't need an out of bounds guard here
				int faceIndex = myVelocityIndices[level](face, faceAxis);

				if (faceIndex >= 0 || faceIndex == BOUNDARY)
				{
					ndx[gradientAxis] += .5 * dx;
				}
				else if (faceIndex == UNASSIGNED)
				{
					ndx[gradientAxis] += .5 * myVelocityIndices[level + 1].dx();
				}
				else
				{
					assert(false);
				}
			}
		}
	}

	return ndx[0] * ndx[1];
}

template<typename Solution, typename Viscosity>
void AnalyticalQuadtreeViscositySolver::buildCenterStencils(std::vector<std::vector<StressStencilFace>>& centerStressStencils,
	std::vector<std::vector<double>>& centerBoundaryStencils,
	std::vector<double>& centerWeights,
	const Solution& solution,
	const Viscosity& viscosity,
	int dofCount)
{
	int levels = myTree.levels();

	for (int level = 0; level < levels; ++level)
	{
		double dx = myCenterStressIndices[level].dx();

		double vol = std::pow(dx, 2);

		for (int axis = 0; axis < 2; ++axis)
		{
			const int offset = dofCount * axis;

			Vec2i size = myCenterStressIndices[level].size();

			forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
			{
				int index = myCenterStressIndices[level](cell);

				if (index >= 0)
				{
					auto& faces = centerStressStencils[index + offset];
					auto& boundaries = centerBoundaryStencils[index + offset];

					getCenterStressFaces(faces, boundaries, solution, cell, level, axis);

					// Incorporate velocity into the weights
					Vec2d pos = myCenterStressIndices[level].indexToWorld(cell.cast<double>());
					centerWeights[index] = vol * viscosity(pos);
				}
			});
		}
	}
}

void AnalyticalQuadtreeViscositySolver::applyToMatrix(std::vector<Eigen::Triplet<double>>& elements, VectorXd& rhsVector, double& diagonal, double coeff, int faceIndex,
	const std::vector<StressStencilFace>& faces,
	const std::vector<double>& boundary_faces) const
{
	bool indexFound = false;

	for (const auto& face : faces)
	{
		if (face.myIndex == faceIndex)
		{
			coeff *= face.myCoeff;
			indexFound = true;
			break;
		}
	}
	assert(indexFound);

	for (const auto& face : faces)
	{
		double element = coeff * face.myCoeff;

		if (face.myIndex == faceIndex)
		{
			diagonal += element;
		}
		else
		{
			elements.emplace_back(faceIndex, face.myIndex, element);
		}
	}
	for (const auto& face : boundary_faces)
	{
		rhsVector[faceIndex] += -coeff * face;
	}
}

template<typename Solution>
void AnalyticalQuadtreeViscositySolver::getNodeStressFaces(std::vector<StressStencilFace>& faces,
															std::vector<double>& boundaryFaces,
															const Solution& solution,
															const Vec2i& node, int level) const
{
	faces.clear();
	boundaryFaces.clear();

	// Build the dx per axis
	double dx = myVelocityIndices[level].dx();
	Vec2d ndx = Vec2d::Zero();

	for (int faceAxis : {0, 1})
	{
		int gradientAxis = (faceAxis + 1) % 2;
		for (int direction : {0, 1})
		{
			Vec2i face = nodeToFace(node, gradientAxis, direction);

			// Since nodes on the boundary are included, we need to account for ghost faces
			// out of bounds
			if (face[gradientAxis] < 0 || face[gradientAxis] >= myVelocityIndices[level].size(faceAxis)[gradientAxis])
			{
				ndx[gradientAxis] += .5 * dx;
			}
			else
			{
				int faceIndex = myVelocityIndices[level](face, faceAxis);

				if (faceIndex >= 0 || faceIndex == BOUNDARY)
				{
					ndx[gradientAxis] += .5 * dx;
				}
				// We know the node is at the finest level of adjacent grid
				// and the face grading makes it so that we can assume only
				// two levels are adjacent to the node.
				else if (faceIndex == UNASSIGNED)
				{
					ndx[gradientAxis] += .5 * myVelocityIndices[level + 1].dx();
				}
			}

		}
	}

	for (int faceAxis : {0, 1})
	{
		int gradientAxis = (faceAxis + 1) % 2;
		for (int direction : {0, 1})
		{
			Vec2i face = nodeToFace(node, gradientAxis, direction);

			double sign = (direction == 0) ? -1 : 1;
			
			if (face[gradientAxis] < 0 || face[gradientAxis] >= myVelocityIndices[level].size(faceAxis)[gradientAxis])
			{
				Vec2d pos = myVelocityIndices[level].indexToWorld(face.cast<double>(), faceAxis);
				double val = solution(pos, faceAxis);
				boundaryFaces.push_back(.5 * sign * val / ndx[gradientAxis]);
			}
			else
			{
				int faceIndex = myVelocityIndices[level](face, faceAxis);

				// If the face is set, life is easy. Add to gradient.
				if (faceIndex >= 0)
				{
					faces.push_back(StressStencilFace(faceIndex, .5 * sign / ndx[gradientAxis]));
				}
				else if (faceIndex == BOUNDARY)
				{
					Vec2d pos = myVelocityIndices[level].indexToWorld(face.cast<double>(), faceAxis);
					double val = solution(pos, faceAxis);
					boundaryFaces.push_back(.5 * sign * val / ndx[gradientAxis]);
				}
				// Face is inactive. We're possibly at a dangling node or an adjacent big face
				else if (faceIndex == UNASSIGNED)
				{
					// If the adjacent face is inactive, we need to handle the possibility of a dangling node.
					// Dangling nodes have "odd" index positions in the non-edge aligned axis.
					if (node[faceAxis] % 2 != 0)
					{
						// The dangling node case has a few cases that need to be handled. If the two axis-aligned
						// faces in the neighbouring cell are one level higher, we just average them together in the sparse matrix.
						// This is the same if both faces are at the same level (i.e. big grid but small faces). However if one is big
						// and the other is small, then the small faces must be averaged together to match the big face's position and
						// then added to the matrix. (TODO: I didn't do that yet).

						Vec2i offsetFace = face;
						--offsetFace[faceAxis];
						{
							Vec2i parentFace = myTree.getParentFace(offsetFace);
							int parentFaceIndex = myVelocityIndices[level + 1](parentFace, faceAxis);
							// We average the two faces together for the dangling edge
							if (parentFaceIndex >= 0)
							{
								// .25 because it is an average of two faces with a .5 coefficient
								faces.push_back(StressStencilFace(parentFaceIndex, .25 * sign / ndx[gradientAxis]));
							}
							else if (parentFaceIndex == BOUNDARY)
							{
								Vec2d pos = myVelocityIndices[level + 1].indexToWorld(parentFace.cast<double>(), faceAxis);
								double val = solution(pos, faceAxis);
								boundaryFaces.push_back(.25 * sign * val / ndx[gradientAxis]);
							}
							// If the small inset faces are active, we will want to accept all of them
							else if (parentFaceIndex == UNASSIGNED)
							{
								for (int childIndex = 0; childIndex < 2; ++childIndex)
								{
									// The way the face grids are set up is a little counter intuitive since the 
									// lower level faces must be inserts in the big face.
									Vec2i childFace = myTree.getChildFace(parentFace, faceAxis, childIndex);

									// Since we went up a level to get the parent and then got that child, we're back at
									// our starting level.
									int childFaceIndex = myVelocityIndices[level](childFace, faceAxis);

									if (childFaceIndex >= 0)
									{
										// We're averaging the two small faces together at the big face center, which is then
										// averaged again at the cell center.
										faces.push_back(StressStencilFace(childFaceIndex, .125 * sign / ndx[gradientAxis]));
									}
									else if (childFaceIndex == BOUNDARY)
									{
										Vec2d pos = myVelocityIndices[level].indexToWorld(childFace.cast<double>(), faceAxis);
										double val = solution(pos, faceAxis);
										boundaryFaces.push_back(.125 * sign * val / ndx[gradientAxis]);
									}
									else
									{
										assert(false);
									}
								}
							}
						}

						offsetFace[faceAxis] += 2;
						{
							Vec2i parentFace = myTree.getParentFace(offsetFace);
							int parentFaceIndex = myVelocityIndices[level + 1](parentFace, faceAxis);
							// We average the two faces together for the dangling edge
							if (parentFaceIndex >= 0)
							{
								faces.push_back(StressStencilFace(parentFaceIndex, .25 * sign / ndx[gradientAxis]));
							}
							// If the small inset faces are active, we will want to accept all of them
							else if (parentFaceIndex == BOUNDARY)
							{
								Vec2d pos = myVelocityIndices[level + 1].indexToWorld(parentFace.cast<double>(), faceAxis);
								double val = solution(pos, faceAxis);
								boundaryFaces.push_back(.25 * sign * val / ndx[gradientAxis]);
							}
							else if (parentFaceIndex == UNASSIGNED)
							{ 
								for (int childIndex = 0; childIndex < 2; ++childIndex)
								{
									// The way the face grids are set up is a little counter intuitive since the 
									// lower level faces must be inserts in the big face.
									Vec2i childFace = myTree.getChildFace(parentFace, faceAxis, childIndex);

									// Since we went up a level to get the parent and then got that child, we're back at
									// our starting level.
									int childFaceIndex = myVelocityIndices[level](childFace, faceAxis);

									if (childFaceIndex >= 0)
									{
										// We're averaging the two small faces together at the big face center, which is then
										// averaged again at the cell center.
										faces.push_back(StressStencilFace(childFaceIndex, .125 * sign / ndx[gradientAxis]));
									}
									else if (childFaceIndex == BOUNDARY)
									{
										Vec2d pos = myVelocityIndices[level].indexToWorld(childFace.cast<double>(), faceAxis);
										double val = solution(pos, faceAxis);
										boundaryFaces.push_back(.125 * sign * val / ndx[gradientAxis]);
									}
									else
									{
										assert(false);
									}
								}
							}
						}
					}
					// The adjacent face is at the parent level
					else
					{
						Vec2i parentFace = myTree.getParentFace(face);
						faceIndex = myVelocityIndices[level + 1](parentFace, faceAxis);

						if (faceIndex >= 0)
						{
							faces.push_back(StressStencilFace(faceIndex, .5 * sign / ndx[gradientAxis]));
						}
						else if (faceIndex == BOUNDARY)
						{
							Vec2d pos = myVelocityIndices[level + 1].indexToWorld(parentFace.cast<double>(), faceAxis);
							double val = solution(pos, faceAxis);
							boundaryFaces.push_back(.5 * sign * val / ndx[gradientAxis]);
						}
						else
						{
							assert(false);
						}
					}
				}
			}
		}
	}
}

template<typename Solution>
void AnalyticalQuadtreeViscositySolver::getCenterStressFaces(std::vector<StressStencilFace>& faces,
	std::vector<double>& boundaryFaces,
	const Solution& solution,
	const Vec2i& cell, const int level,
	const int axis) const
{
	faces.clear();
	boundaryFaces.clear();

	double dx = myVelocityIndices[level].dx();

	for (int direction : {0, 1})
	{
		Vec2i face = cellToFace(cell, axis, direction);

		double sign = (direction == 0) ? -1 : 1;

		int velocityIndex = myVelocityIndices[level](face, axis);

		if (velocityIndex >= 0)
		{
			faces.push_back(StressStencilFace(velocityIndex, sign / dx));
		}
		else if (velocityIndex == BOUNDARY)
		{
			Vec2d pos = myVelocityIndices[level].indexToWorld(face.cast<double>(), axis);
			double val = solution(pos, axis);
			boundaryFaces.push_back(sign * val / dx);
		}
		// If the adjacent face is not active, the child faces must be active.
		// We average them together to use as our velocity gradient.
		else if (velocityIndex == UNASSIGNED)
		{
			assert(level > 0);
			for (int childIndex = 0; childIndex < 2; ++childIndex)
			{
				Vec2i child = myTree.getChildFace(face, axis, childIndex);

				int faceIndex = myVelocityIndices[level - 1](child, axis);
				assert(faceIndex >= 0);

				// Note the coefficient of .5 because we're averaging the small faces
				faces.push_back(StressStencilFace(faceIndex, .5 * sign / dx));
			}
		}
	}
}

double AnalyticalQuadtreeViscositySolver::getVelocityIntegrationVolumes(const Vec2i& face, int axis, int level) const
{
	assert(face[0] < myVelocityIndices[level].size(axis)[0] &&
		face[1] < myVelocityIndices[level].size(axis)[1]);

	assert(myVelocityIndices[level](face, axis) >= 0);

	// Edge-aligned spacing
	double dx = myVelocityIndices[level].dx();

	double gdx = 0;
	// Find gradient dx. There are no safety checks here. It should be verified that these gradient samples are active
	for (int cellDirection : {0, 1})
	{
		Vec2i cell = faceToCell(face, axis, cellDirection);

		if (cell[axis] < 0 || cell[axis] >= myTree.size(level)[axis])
		{
			gdx += .5 * dx;
		}
		else
		{
			if (myTree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
			{
				gdx += .5 * dx;
			}
			else
			{
				Vec2i parent = myTree.getParentCell(cell);
				if (myTree.getCellLabel(parent, level + 1) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
				{
					gdx += .5 * myVelocityIndices[level + 1].dx();
				}
				else
				{
					assert(false);
				}
			}
		}
	}

	return dx * gdx;
}



TEST(QUADTREE_ANALYTICAL_VISCOSITY_TESTS, ANALYTICAL_VISCOSITY)
{
	double dt = 1.;
	double mu = 1.;

	auto initial = [&](const Vec2d& pos, int axis)
	{
		double x = pos[0];
		double y = pos[1];
		double val;
		if (axis == 0)
			val = sin(x) * sin(y) - dt * (2. / PI * cos(x) * sin(y) + (cos(x + y) - 2 * sin(x) * sin(y)) * (x / PI + .5));
		else
			val = sin(x) * sin(y) - dt * ((cos(x) * cos(y) - 3. * sin(x) * sin(y)) * (x / PI + .5) + 1. / PI * sin(x + y));

		return val;
	};

	auto solution = [](const Vec2d& pos, int axis) { return sin(pos[0]) * sin(pos[1]); };
	auto viscosity = [](const Vec2d& pos) -> double { return pos[0] / PI + .5; };

	int base = 32;
	double dx = PI / double(base);

	Vec2d origin = Vec2d::Zero();
	Vec2i size = Vec2i::Constant(int(std::round(PI / dx)));
	Transform xform(dx, origin);

	LevelSet surface(xform, size, 4);

	// Build an implicit surface to set fine cells for the quadtree
	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		Vec2d pos = surface.indexToWorld(cell.cast<double>());
		surface(cell) = std::min(pos.norm() - .5 * std::sqrt(2.) * PI, (pos - Vec2d::Constant(PI)).norm() - .5 * std::sqrt(2.) * PI);
	});

	auto surfaceRefiner = [&](const Vec2d& pos) { if (std::fabs(surface.biLerp(pos)) < dx) return 0; return -1; };

	int levels = 4;
    AnalyticalQuadtreeViscositySolver viscositySolver(xform, size, levels, surfaceRefiner);

	std::vector<Vec2d> error(levels);

	for (int level = 0; level < levels; ++level)
	{
		if (level > 0)
		{
			viscositySolver.refine();
		}

		error[level] = viscositySolver.solve(initial, solution, viscosity, dt);
	}

	for (int level = 1; level < levels; ++level)
    {
		EXPECT_GT(error[level - 1][0] / error[level][0], 1.5);
        EXPECT_GT(error[level - 1][1] / error[level][1], 1.5);
	}
}
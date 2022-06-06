#include "AdaptiveViscositySolver.h"

#include <iostream>
#include <queue>

#include <Eigen/Sparse>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "ComputeWeights.h"
#include "LevelSet.h"
#include "QuadtreeVectorInterpolator.h"

namespace FluidSim2D
{

static constexpr int FLUID = 0;
static constexpr int UNASSIGNED = -1;
static constexpr int COLLISION = -2;
static constexpr int OUTSIDE = -3;

AdaptiveViscositySolver::AdaptiveViscositySolver(const LevelSet& surface,
	const LevelSet& solidSurface,
	const VectorGrid<double>& solidVelocity,
	const ScalarGrid<double>& viscosity,
	int levels)
	: myQuadtree(surface.xform(), surface.size(), levels)
	, mySurface(surface)
	, mySolidSurface(solidSurface)
	, mySolidVelocity(solidVelocity)
	, myViscosity(viscosity)
{
	assert(mySurface.isGridMatched(mySolidSurface));
	assert(mySurface.isGridMatched(myViscosity));
	assert(mySolidVelocity.size(0)[0] - 1 == mySurface.size()[0] &&
		mySolidVelocity.size(0)[1] == mySurface.size()[1] &&
		mySolidVelocity.size(1)[0] == mySurface.size()[0] &&
		mySolidVelocity.size(1)[1] - 1 == mySurface.size()[1]);

	int samples = 3;

	myCenterAreas = computeSupersampledAreas(mySurface, ScalarGridSettings::SampleType::CENTER, samples);
	myNodeAreas = computeSupersampledAreas(mySurface, ScalarGridSettings::SampleType::NODE, samples);
	myFaceAreas = computeSupersampledFaceAreas(mySurface, samples);

	// Construct the quadtree
	auto refiner = [&](const Vec2d& pos) -> int
	{
		double sdf = mySurface.biLerp(pos);
		if (std::fabs(sdf) < 2. * mySurface.dx())
		{
			return 0;
		}
		else if (sdf < 0.)
		{
			if (mySolidSurface.biLerp(pos) < 2. * mySolidSurface.dx())
			{
				return 0;
			}
			else
			{
				return -1;
			}
		}
		else
		{
			return 1;
		}
	};

	myQuadtree.buildTree(refiner);

	levels = myQuadtree.levels();

	myVelocityIndices.resize(levels);
	myNodeStressIndices.resize(levels);
	myCenterStressIndices.resize(levels);
	myVelocitSolution.resize(levels);

	for (int level = 0; level < levels; ++level)
	{
		Transform localXform = myQuadtree.xform(level);
		Vec2i localGridSize = myQuadtree.size(level);

		myVelocityIndices[level] = VectorGrid<int>(localXform, localGridSize, UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);
		myNodeStressIndices[level] = ScalarGrid<int>(localXform, localGridSize, UNASSIGNED, ScalarGridSettings::SampleType::NODE);
		myCenterStressIndices[level] = ScalarGrid<int>(localXform, localGridSize, UNASSIGNED, ScalarGridSettings::SampleType::CENTER);

		myVelocitSolution[level] = VectorGrid<double>(localXform, localGridSize, 0, VectorGridSettings::SampleType::STAGGERED);
	}
}

int AdaptiveViscositySolver::buildVelocityIndices()
{
	// Loop over each level of the grid. Loop over each face.
	// If the face has one ACTIVE adjacent cell and one UP adjacent cell OR 
	// has two ACTIVE adjacent cells, then the face is considered active.

	int index = 0;
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		Vec2i gridSize;

		if (level == 0)
		{
			gridSize = mySurface.size();
		}
		else
		{
			gridSize = myQuadtree.size(level);
		}

		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i::Zero(), myVelocityIndices[level].size(axis), [&](const Vec2i& face)
			{
				// Grab adjacent cells
				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				if (backwardCell[axis] < 0 || forwardCell[axis] >= gridSize[axis])
				{
					if (level == 0)
					{
						myVelocityIndices[level](face, axis) = OUTSIDE;
					}
					return;
				}

				QuadtreeGrid::QuadtreeCellLabel backwardLabel = myQuadtree.getCellLabel(backwardCell, level);
				QuadtreeGrid::QuadtreeCellLabel forwardLabel = myQuadtree.getCellLabel(forwardCell, level);

				if (level == 0)
				{
					// Regular grid check
					if (backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
					{
						bool activeVelocity = false;

						if (myCenterAreas(backwardCell) > 0. ||
							myCenterAreas(forwardCell) > 0.)
						{
							activeVelocity = true;
						}

						if (!activeVelocity)
						{
							Vec2i backwardNode = faceToNode(face, axis, 0);
							Vec2i forwardNode = faceToNode(face, axis, 1);

							if (myNodeAreas(backwardNode) > 0. ||
								myNodeAreas(forwardNode) > 0.)
							{
								activeVelocity = true;
							}
						}

						if (activeVelocity)
						{
							if (mySolidSurface.biLerp(myVelocityIndices[level].indexToWorld(face.cast<double>(), axis)) <= 0.)
							{
								myVelocityIndices[level](face, axis) = COLLISION;
							}
							else
							{
								myVelocityIndices[level](face, axis) = index++;
							}
						}
						else
						{
							myVelocityIndices[level](face, axis) = OUTSIDE;
						}
					}
					else if (backwardLabel == QuadtreeGrid::QuadtreeCellLabel::INACTIVE || forwardLabel == QuadtreeGrid::QuadtreeCellLabel::INACTIVE)
					{
						myVelocityIndices[level](face, axis) = OUTSIDE;
					}
					else if ((backwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP))
					{
						myVelocityIndices[level](face, axis) = index++;
#if !defined(NDEBUG)
						Vec2d point = myVelocityIndices[level].indexToWorld(face.cast<double>(), axis);

						assert(mySolidSurface.biLerp(point) > 0);
						assert(mySurface.biLerp(point) < 0);
#endif
					}
					else
					{
						assert(!((backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN)
							|| (backwardLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)) &&
							!((backwardLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP)
								|| (backwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN)) &&
							!((backwardLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::INACTIVE)
								|| (backwardLabel == QuadtreeGrid::QuadtreeCellLabel::INACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::DOWN)));
					}
				}
				else
				{
					// The three possible instances of an active face.
					if ((backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE) ||
						(backwardLabel == QuadtreeGrid::QuadtreeCellLabel::ACTIVE && forwardLabel == QuadtreeGrid::QuadtreeCellLabel::UP))
					{
						Vec2d point = myVelocityIndices[level].indexToWorld(face.cast<double>(), axis);
						assert(mySolidSurface.biLerp(point) > 0);
						assert(mySurface.biLerp(point) < 0);

						myVelocityIndices[level](face, axis) = index++;
					}
				}
			});
		}
	}

	return index;
}

int AdaptiveViscositySolver::buildNodeStressIndices()
{
	int index = 0;
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		Vec2i gridSize;
		if (level == 0)
		{
			gridSize = mySurface.size();
		}
		else
		{
			gridSize = myQuadtree.size(level);
		}

		forEachVoxelRange(Vec2i::Zero(), myNodeStressIndices[level].size(), [&](const Vec2i& node)
			{
				bool stressActive = false;

				for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
				{
					Vec2i cell = nodeToCell(node, cellIndex);

					if (cell[0] < 0 || cell[1] < 0 ||
						cell[0] >= gridSize[0] || cell[1] >= gridSize[1])
					{
						myNodeStressIndices[level](node) = OUTSIDE;
						continue;
					}

					// If the grid points down then it is not in the system
					if (myQuadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::DOWN)
					{
						stressActive = false;
						break;
					}
					else if (myQuadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
					{
						stressActive = true;
					}
				}

				if (stressActive)
				{
					if (level == 0)
					{
						if (myNodeAreas(node) > 0.)
						{
							myNodeStressIndices[level](node) = index++;
						}
						else
						{
							myNodeStressIndices[level](node) = OUTSIDE;
						}
					}
					else
					{
						myNodeStressIndices[level](node) = index++;
#if !defined(NDEBUG)
						Vec2d pos = myNodeStressIndices[level].indexToWorld(node.cast<double>());
						assert(mySurface.biLerp(pos) <= 0);
#endif
					}
				}
			});
	}

	return index;
}

int AdaptiveViscositySolver::buildCenterIndices()
{
	int index = 0;
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		forEachVoxelRange(Vec2i::Zero(), myQuadtree.size(level), [&](const Vec2i& cell)
		{
			if (myQuadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
			{
				if (level != 0 || myCenterAreas(cell) > 0)
				{
					myCenterStressIndices[level](cell) = index++;
				}
			}
		});
	}

	return index;
}

void AdaptiveViscositySolver::getNodeStressFaces(std::vector<StressStencilFace>& faces,
	std::vector<double>& boundary_faces,
	const Vec2i& node, const int level) const
{
	faces.clear();
	boundary_faces.clear();

	// Build the dx per axis
	double dx = myVelocityIndices[level].dx();
	Vec2d ndx(0);

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
				int fidx = myVelocityIndices[level](face, faceAxis);

				if (fidx >= 0 || fidx == OUTSIDE || fidx == COLLISION)
				{
					ndx[gradientAxis] += .5 * dx;
				}
				// We know the node is at the finest level of adjacent grid
				// and the face grading makes it so that we can assume only
				// two levels are adjacent to the node.
				else if (fidx == UNASSIGNED)
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

	for (int faceAxis : {0, 1})
	{
		int gradientAxis = (faceAxis + 1) % 2;
		for (int direction : {0, 1})
		{
			Vec2i face = nodeToFace(node, gradientAxis, direction);

			double sign = direction == 0 ? -1 : 1;

			if (face[gradientAxis] < 0 || face[gradientAxis] >= myVelocityIndices[level].size(faceAxis)[gradientAxis])
			{
				continue;
			}

			int faceIndex = myVelocityIndices[level](face, faceAxis);

			// If the face is set, life is easy. Add to gradient.
			if (faceIndex >= 0)
			{
				faces.push_back(StressStencilFace(faceIndex, .5 * sign / ndx[gradientAxis]));
			}
			else if (faceIndex == COLLISION)
			{
				Vec2d pos = myVelocityIndices[level].indexToWorld(face.cast<double>(), faceAxis);
				double solidVelocity = mySolidVelocity.biLerp(pos, faceAxis);
				boundary_faces.push_back(.5 * sign * solidVelocity / ndx[gradientAxis]);
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
						Vec2i parentFace = myQuadtree.getParentFace(offsetFace);
						int parentFaceIndex = myVelocityIndices[level + 1](parentFace, faceAxis);
						// We average the two faces together for the dangling edge
						if (parentFaceIndex >= 0)
						{
							// .25 because it is an average of two faces with a .5 coefficient
							faces.push_back(StressStencilFace(parentFaceIndex, .25 * sign / ndx[gradientAxis]));
						}
						else if (parentFaceIndex == COLLISION)
						{
							Vec2d pos = myVelocityIndices[level + 1].indexToWorld(parentFace.cast<double>(), faceAxis);
							double solidVelocity = mySolidVelocity.biLerp(pos, faceAxis);
							boundary_faces.push_back(.25 * sign * solidVelocity / ndx[gradientAxis]);
						}
						// If the small inset faces are active, we will want to accept all of them
						else if (parentFaceIndex == UNASSIGNED)
						{
							for (int childIndex = 0; childIndex < 2; ++childIndex)
							{
								// The way the face grids are set up is a little counter intuitive since the 
								// lower level faces must be inserts in the big face.
								Vec2i childFace = myQuadtree.getChildFace(parentFace, faceAxis, childIndex);

								// Since we went up a level to get the parent and then got that child, we're back at
								// our starting level.
								int childFaceIndex = myVelocityIndices[level](childFace, faceAxis);

								if (childFaceIndex >= 0)
								{
									// We're averaging the two small faces together at the big face center, which is then
									// averaged again at the cell center.
									faces.push_back(StressStencilFace(childFaceIndex, .125 * sign / ndx[gradientAxis]));
								}
								else if (childFaceIndex == COLLISION)
								{
									Vec2d pos = myVelocityIndices[level].indexToWorld(childFace.cast<double>(), faceAxis);
									double solidVelocity = mySolidVelocity.biLerp(pos, faceAxis);
									boundary_faces.push_back(.125 * sign * solidVelocity / ndx[gradientAxis]);
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
						Vec2i parentFace = myQuadtree.getParentFace(offsetFace);
						int parentFaceIndex = myVelocityIndices[level + 1](parentFace, faceAxis);
						// We average the two faces together for the dangling edge
						if (parentFaceIndex >= 0)
						{
							faces.push_back(StressStencilFace(parentFaceIndex, .25 * sign / ndx[gradientAxis]));
						}
						else if (parentFaceIndex == COLLISION)
						{
							Vec2d pos = myVelocityIndices[level + 1].indexToWorld(parentFace.cast<double>(), faceAxis);
							double solidVelocity = mySolidVelocity.biLerp(pos, faceAxis);
							boundary_faces.push_back(.25 * sign * solidVelocity / ndx[gradientAxis]);
						}
						// If the small inset faces are active, we will want to accept all of them
						else if (parentFaceIndex == UNASSIGNED)
						{
							for (int childIndex = 0; childIndex < 2; ++childIndex)
							{
								// The way the face grids are set up is a little counter intuitive since the 
								// lower level faces must be inserts in the big face.
								Vec2i childFace = myQuadtree.getChildFace(parentFace, faceAxis, childIndex);

								// Since we went up a level to get the parent and then got that child, we're back at
								// our starting level.
								int childFaceIndex = myVelocityIndices[level](childFace, faceAxis);

								if (childFaceIndex >= 0)
								{
									// We're averaging the two small faces together at the big face center, which is then
									// averaged again at the cell center.
									faces.push_back(StressStencilFace(childFaceIndex, .125 * sign / ndx[gradientAxis]));
								}
								else if (childFaceIndex == COLLISION)
								{
									Vec2d pos = myVelocityIndices[level].indexToWorld(childFace.cast<double>(), faceAxis);
									double solidVelocity = mySolidVelocity.biLerp(pos, faceAxis);
									boundary_faces.push_back(.125 * sign * solidVelocity / ndx[gradientAxis]);
								}
								else assert(false);
							}
						}
					}

				}
				// The adjacent face is at the parent level
				else
				{
					Vec2i parentFace = myQuadtree.getParentFace(face);
					int parentFaceIndex = myVelocityIndices[level + 1](parentFace, faceAxis);
					if (parentFaceIndex >= 0)
					{
						faces.push_back(StressStencilFace(parentFaceIndex, .5 * sign / ndx[gradientAxis]));
					}
					else if (parentFaceIndex == COLLISION)
					{
						Vec2d pos = myVelocityIndices[level + 1].indexToWorld(parentFace.cast<double>(), faceAxis);
						double solidVelocity = mySolidVelocity.biLerp(pos, faceAxis);
						boundary_faces.push_back(.5 * sign * solidVelocity / ndx[gradientAxis]);
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

double AdaptiveViscositySolver::getNodeIntegrationVolumes(const Vec2i& node, int level) const
{
	int noteIndex = myNodeStressIndices[level](node);
	assert(noteIndex >= 0 || noteIndex == COLLISION);

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
				int faceIndex = myVelocityIndices[level](face, faceAxis);

				// Since nodes on the boundary are included, we need to account for ghost faces
				// out of bounds
				if (faceIndex >= 0 || faceIndex == OUTSIDE || faceIndex == COLLISION)
				{
					ndx[gradientAxis] += .5 * dx;
				}
				// We know the node is at the finest level of adjacent grid
				// and the face grading makes it so that we can assume only
				// two levels of faces are adjacent to the node.
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

void AdaptiveViscositySolver::buildNodeStencils(std::vector<std::vector<StressStencilFace>>& nodeStressStencils,
	std::vector<std::vector<double>>& nodeBoundaryStencils,
	std::vector<double>& nodeWeights) const
{
	double dx = myNodeStressIndices[0].dx();
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		forEachVoxelRange(Vec2i::Zero(), myNodeStressIndices[level].size(), [&](const Vec2i& node)
		{
			int nodeIndex = myNodeStressIndices[level](node);

			if (nodeIndex >= 0)
			{
				auto& faces = nodeStressStencils[nodeIndex];
				auto& boundaries = nodeBoundaryStencils[nodeIndex];

				getNodeStressFaces(faces, boundaries, node, level);

				double nodeWeight;

				if (level == 0)
				{
					nodeWeight = myNodeAreas(node);

					// If the control volume is full, it is possible that
					// the node is at a transition. In this case, 1. is not
					// accurate as the volume will be stretched into the
					// coarse cells.
					if (nodeWeight == 1.)
					{
						nodeWeight = getNodeIntegrationVolumes(node, level);
					}
					else
					{
						nodeWeight *= dx * dx;
					}
				}
				else
				{
					nodeWeight = getNodeIntegrationVolumes(node, level);
				}

				// Incorporate velocity into the weights
				Vec2d pos = myNodeStressIndices[level].indexToWorld(node.cast<double>());
				nodeWeight *= myViscosity.biLerp(pos);

				// Account for diagonal K matrix
				nodeWeights[nodeIndex] = 2. * nodeWeight;
			}
		});
	}
}

void AdaptiveViscositySolver::buildCenterStencils(std::vector<std::vector<StressStencilFace>>& centerStressStencils,
	std::vector<std::vector<double>>& centerBoundaryStencils,
	std::vector<double>& centerWeights,
	int centerDOFCount)
{
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		double dx = myCenterStressIndices[level].dx();

		double vol = dx * dx;

		for (int axis = 0; axis < 2; ++axis)
		{
			const int offset = centerDOFCount * axis;

			
			forEachVoxelRange(Vec2i::Zero(), myCenterStressIndices[level].size(), [&](const Vec2i& cell)
			{
				int centerIndex = myCenterStressIndices[level](cell);

				if (centerIndex >= 0)
				{
					auto& faces = centerStressStencils[centerIndex + offset];
					auto& boundaries = centerBoundaryStencils[centerIndex + offset];

					getCenterStressFaces(faces, boundaries, cell, level, axis);

					if (axis == 0)
					{
						double weight = vol;

						if (level == 0)
						{
							weight *= myCenterAreas(cell);
						}

						Vec2d pos = myCenterStressIndices[level].indexToWorld(cell.cast<double>());
						weight *= myViscosity.biLerp(pos);

						centerWeights[centerIndex] = weight;
					}
				}
			});
		}
	}
}

void AdaptiveViscositySolver::getCenterStressFaces(std::vector<StressStencilFace>& faces,
	std::vector<double>& boundaryFaces,
	const Vec2i& cell, int level,
	int axis) const
{
	faces.clear();
	boundaryFaces.clear();

	double dx = myQuadtree.xform(level).dx();

	for (int direction : {0, 1})
	{
		Vec2i face = cellToFace(cell, axis, direction);
			
		double sign = direction == 0 ? -1 : 1;

		int faceIndex = myVelocityIndices[level](face, axis);

		if (faceIndex >= 0)
		{
			faces.push_back(StressStencilFace(faceIndex, sign / dx));
		}
		else if (faceIndex == COLLISION)
		{
			Vec2d pos = myVelocityIndices[level].indexToWorld(face.cast<double>(), axis);
			double solidVelocity = mySolidVelocity.biLerp(pos, axis);
			boundaryFaces.push_back(sign * solidVelocity / dx);

			assert(level == 0);
		}
		// If the adjacent face is not active, the child faces must be active.
		// We average them together to use as our velocity gradient.
		else if (faceIndex == UNASSIGNED)
		{
			for (int childIndex = 0; childIndex < 2; ++childIndex)
			{
				Vec2i childFace = myQuadtree.getChildFace(face, axis, childIndex);

				int childFaceIndex = myVelocityIndices[level - 1](childFace, axis);
				assert(childFaceIndex >= 0);

				// Note the coefficient of .5 because we're averaging the small faces
				faces.push_back(StressStencilFace(childFaceIndex, .5 * sign / dx));
			}
		}
	}
}

void AdaptiveViscositySolver::applyToMatrix(std::vector<Eigen::Triplet<double>>& elements, VectorXd& rhs, double& diagonal, double coeff, int faceIndex,
	const std::vector<StressStencilFace>& faces,
	const std::vector<double>& boundary_faces)
{
	bool indexFound = false;

	for (const auto& face : faces)
	{
		if (face.myIndex == faceIndex)
		{
			coeff *= face.myValue;
			indexFound = true;
			break;
		}
	}
	assert(indexFound);

	for (const auto& face : faces)
	{
		double element = coeff * face.myValue;

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
		rhs(faceIndex) += -coeff * face;
	}
}

double AdaptiveViscositySolver::getVelocityControlVolumes(const Vec2i& face, int axis, int level) const
{
	assert(face[0] < myVelocityIndices[level].size(axis)[0] &&
			face[1] < myVelocityIndices[level].size(axis)[1]);

	assert(myVelocityIndices[level](face, axis) >= 0);

	// Edge-aligned spacing
	double dx = myVelocityIndices[level].dx();

	double gdx = 0;
	// Find gradient dx. There are no safety checks here. It should be verified that these gradient samples are active
	for (int direction : {0, 1})
	{
		Vec2i cell = faceToCell(face, axis, direction);

		if (cell[axis] < 0 || cell[axis] >= myQuadtree.size(level)[axis])
		{
			gdx += .5 * dx;
		}
		else
		{
			if (myQuadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
			{
				gdx += .5 * dx;
			}
			else
			{
				Vec2i parentCell = myQuadtree.getParentCell(cell);
				if (myQuadtree.getCellLabel(parentCell, level + 1) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
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

void AdaptiveViscositySolver::solve(VectorGrid<double>& velocity, double dt)
{
	//
	// Build indices for active quadtree velocity and stress DOFs
	//

	int velocityDOFCount = buildVelocityIndices();
	int nodeDOFCount = buildNodeStressIndices();
	int centerDOFCount = buildCenterIndices();

	//
	// Precompute stress gradients
	//

	std::vector<std::vector<StressStencilFace>> nodeStressStencils(nodeDOFCount);
	std::vector<std::vector<double>> nodeBoundaryStencils(nodeDOFCount);
	std::vector<double> nodeWeights(nodeDOFCount);

	buildNodeStencils(nodeStressStencils, nodeBoundaryStencils, nodeWeights);

	// We need 2 times the number of center stresses because there are separate stencils for du/dx
	// and dv/dy co-located at the center but each center is not given two indices.
	std::vector<std::vector<StressStencilFace>> centerStressStencils(2 * centerDOFCount);
	std::vector<std::vector<double>> centerBoundaryStencils(2 * centerDOFCount);
	std::vector<double> centerWeights(centerDOFCount);

	buildCenterStencils(centerStressStencils, centerBoundaryStencils, centerWeights, centerDOFCount);

	//
	// Build mapping from regular velocity grid to quadtree
	//

	VectorGrid<int> regularSolvableFaces(mySurface.xform(), mySurface.size(), UNASSIGNED, VectorGridSettings::SampleType::STAGGERED);

	int regcount = 0;
	for (int axis = 0; axis < 2; ++axis)
	{
		Vec2i cellSize = myCenterAreas.size();
		forEachVoxelRange(Vec2i::Zero(), regularSolvableFaces.size(axis), [&](const Vec2i& face)
		{
			// Grab adjacent cells
			Vec2i backwardCell = faceToCell(face, axis, 0);
			Vec2i forwardCell = faceToCell(face, axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] >= cellSize[axis])
			{
				return;
			}

			bool inSolve = false;

			if (myCenterAreas(backwardCell) > 0. ||
				myCenterAreas(forwardCell) > 0.)
			{
				inSolve = true;
			}

			if (!inSolve)
			{
				Vec2i backwardNode = faceToNode(face, axis, 0);
				Vec2i forwardNode = faceToNode(face, axis, 1);

				if (myNodeAreas(backwardNode) > 0. ||
					myNodeAreas(forwardNode) > 0.)
				{
					inSolve = true;
				}
			}

			if (inSolve)
			{
				if (mySolidSurface.biLerp(regularSolvableFaces.indexToWorld(face.cast<double>(), axis)) <= 0.)
				{
					regularSolvableFaces(face, axis) = COLLISION;
				}
				else
				{
					regularSolvableFaces(face, axis) = FLUID;
				}
			}
		});
	}

	std::vector<double> quadtreeVelocity(velocityDOFCount, 0);
	{
		constexpr double inAxisWeights[3] = { 1. / 8., 1. / 4., 1. / 8. };

		std::queue<FaceInterpolationWeights> faceQueue;
		for (int level = 0; level < myQuadtree.levels(); ++level)
		{
			for (int axis = 0; axis < 2; ++axis)
			{
				Vec2i size = myVelocityIndices[level].size(axis);

				forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face)
				{
					int velocityIndex = myVelocityIndices[level](face, axis);

					if (velocityIndex >= 0)
					{
						double localVelocity = 0;
						assert(faceQueue.empty());
						faceQueue.push(FaceInterpolationWeights(face, 1., level));

						while (!faceQueue.empty())
						{
							// If the current face's level is zero,
							// add it to the vector. If not, add the 
							// children to the queue.

							Vec2i face = faceQueue.front().face;
							double faceWeight = faceQueue.front().weight;
							int facelevel = faceQueue.front().level;

							faceQueue.pop();

							if (facelevel == 0)
							{
								int regularFaceIndex = regularSolvableFaces(face, axis);

								if (regularFaceIndex < 0)
								{
									std::cout << "Regular index not in the system" << std::endl;
									assert(false);
								}
								else
								{
									localVelocity += faceWeight * velocity(face, axis);
								}
							}
							else
							{
								for (int childIndex = 0; childIndex < 2; ++childIndex)
								{
									Vec2i childFace = myQuadtree.getChildFace(face, axis, childIndex);

									for (int ca = -1; ca < 2; ++ca)
									{
										Vec2i adjacentFace = childFace;
										adjacentFace[axis] += ca;

										double localFaceWeight = inAxisWeights[ca + 1];

										faceQueue.push(FaceInterpolationWeights(adjacentFace, localFaceWeight * faceWeight, facelevel - 1));
									}
								}
							}
						}

						quadtreeVelocity[velocityIndex] = localVelocity;
					}
				});
			}
		}
	}

	// Build variational system
	// (Wu + dt D^T K Wtau M D) u(n+1) = Wu u(n)

	//
	// Build velocity system directly
	//

	double baseCoeff = 2. * dt;

	std::vector<Eigen::Triplet<double>> elements;

	VectorXd rhsVector = VectorXd::Zero(velocityDOFCount);

	VectorXd initialGuessVector = VectorXd::Zero(velocityDOFCount);

	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		for (int axis = 0; axis < 2; ++axis)
		{
			Vec2i size = myVelocityIndices[level].size(axis);

			int cellOffset = centerDOFCount * axis;

			forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& face)
			{
				const int velocityIndex = myVelocityIndices[level](face, axis);
				if (velocityIndex >= 0)
				{
					double diagonalElement = 0;

					// Build cell centered-stress and dangling node stresses

					for (int cellDirection : {0, 1}) 
					{
						Vec2i cell = faceToCell(face, axis, cellDirection);

						if (cell[axis] < 0 || cell[axis] >= myQuadtree.size(level)[axis])
						{
							continue;
						}

						Vec2i stressCell;
						int stressLevel;

						if (myQuadtree.getCellLabel(cell, level) == QuadtreeGrid::QuadtreeCellLabel::ACTIVE)
						{
							stressCell = cell;
							stressLevel = level;
						}
						// Since we're face graded, we can safely assume that if the cell
						// isn't active then its parent must be.
						else
						{
							stressCell = myQuadtree.getParentCell(cell);
							stressLevel = level + 1;
						}

						// Apply cell stress contributions to matrix A and rhs
						{
							int cellIndex = myCenterStressIndices[stressLevel](stressCell);

							if (cellIndex == UNASSIGNED)
							{
								assert(stressLevel == 0);
								assert(myCenterAreas(stressCell) == 0);
							}
							else
							{
								assert(cellIndex >= 0);

								const auto& faces = centerStressStencils[cellIndex + cellOffset];
								assert(faces.size() > 0);

								const auto& boundaryFaces = centerBoundaryStencils[cellIndex + cellOffset];

								double coeff = baseCoeff * centerWeights[cellIndex];

								applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);
							}
						}

						// Search for dangling nodes in the orthogonal direction
						for (int faceDirection : {0, 1})
						{
							int otherAxis = (axis + 1) % 2;

							Vec2i adjacentFace = cellToFace(cell, otherAxis, faceDirection);

							if (myVelocityIndices[stressLevel](adjacentFace, otherAxis) == UNASSIGNED)
							{
								Vec2i node = myQuadtree.getInsetNode(adjacentFace, otherAxis);

								int nodeIndex = myNodeStressIndices[stressLevel - 1](node);
								if (nodeIndex >= 0)
								{
									const auto& faces = nodeStressStencils[nodeIndex];
									assert(faces.size() > 0);

									const auto& boundaryFaces = nodeBoundaryStencils[nodeIndex];

									double coeff = baseCoeff * nodeWeights[nodeIndex];

									applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);

								}
								else
								{
									assert(nodeIndex == OUTSIDE);
								}
							}
						}
					}

					// Build velocity mappings from adjacent nodes
					for (int nodeDirection : {0, 1})
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
							Vec2i childNode = myQuadtree.getChildNode(node);
							int childNodeIndex = myNodeStressIndices[level - 1](childNode);

							if (childNodeIndex >= 0)
							{
								const auto& faces = nodeStressStencils[childNodeIndex];
								assert(faces.size() > 0);

								const auto& boundaryFaces = nodeBoundaryStencils[childNodeIndex];

								double coeff = baseCoeff * nodeWeights[childNodeIndex];

								applyToMatrix(elements, rhsVector, diagonalElement, coeff, velocityIndex, faces, boundaryFaces);

							}
							else
							{
								assert(childNodeIndex == OUTSIDE);
							}
						}
						else assert(nodeIndex == OUTSIDE);
					}
					{
						double velocityVolume;
						if (level == 0)
						{
							double sampleDx = myVelocityIndices[level].dx();

							velocityVolume = myFaceAreas(face, axis);

							if (velocityVolume == 1.)
							{
								velocityVolume = getVelocityControlVolumes(face, axis, level);
							}
							else
							{
								velocityVolume *= sampleDx * sampleDx;
							}
						}
						else
						{
							velocityVolume = getVelocityControlVolumes(face, axis, level);
						}

						elements.emplace_back(velocityIndex, velocityIndex, velocityVolume);

						rhsVector(velocityIndex) = velocityVolume * quadtreeVelocity[velocityIndex];
						initialGuessVector(velocityIndex) = quadtreeVelocity[velocityIndex];

					}
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
		return;
	}

	solver.setTolerance(1E-10);

	VectorXd solutionVector = solver.solveWithGuess(rhsVector, initialGuessVector);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to converge" << std::endl;
		return;
	}
	else
	{
		std::cout << "    Solver iterations:     " << solver.iterations() << std::endl;
		std::cout << "    Solver error: " << solver.error() << std::endl;
	}

	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		for (int axis : {0, 1})
		{
			forEachVoxelRange(Vec2i::Zero(), myVelocityIndices[level].size(axis), [&](const Vec2i& face)
			{
				int velocityIndex = myVelocityIndices[level](face, axis);
				if (velocityIndex >= 0)
				{
					myVelocitSolution[level](face, axis) = solutionVector[velocityIndex];
				}
			});
		}
	}

	QuadtreeVectorInterpolator interpolator(myQuadtree, myVelocityIndices, myVelocitSolution);

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), regularSolvableFaces.size(axis), [&](const Vec2i& face)
		{
			int regularVelocityIndex = regularSolvableFaces(face, axis);
			if (regularVelocityIndex >= 0)
			{
				int velocityIndex = myVelocityIndices[0](face, axis);
				if (velocityIndex >= 0)
				{
					velocity(face, axis) = solutionVector[velocityIndex];
				}
				else
				{
					Vec2d pos = regularSolvableFaces.indexToWorld(face.cast<double>(), axis);
					velocity(face, axis) = interpolator.biLerp(pos, axis);
				}
			}
			else if (regularVelocityIndex == COLLISION)
			{
				Vec2d pos = regularSolvableFaces.indexToWorld(face.cast<double>(), axis);
				velocity(face, axis) = mySolidVelocity.biLerp(pos, axis);
			}
		});
	}
}

}
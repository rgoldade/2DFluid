#include "QuadtreeVectorInterpolator.h"

namespace FluidSim2D
{

QuadtreeVectorInterpolator::QuadtreeVectorInterpolator(const QuadtreeGrid& referenceTree,
	const std::vector<VectorGrid<int>>& referenceIndexGrid,
	const std::vector<VectorGrid<double>>& referenceValueGrid)
	: myQuadtree(referenceTree)
	, myIndexGrid(referenceIndexGrid)
	, myValueGrid(referenceValueGrid)
{
	int levels = myQuadtree.levels();

#if !defined(NDEBUG)
	for (int level = 0; level < levels; ++level)
	{
		assert(myIndexGrid[level].sampleType() == VectorGridSettings::SampleType::STAGGERED);
		assert(myValueGrid[level].sampleType() == VectorGridSettings::SampleType::STAGGERED);
	}
#endif

	myNodeLabels.resize(levels);
	myNodeValues.resize(levels);

	std::vector<ScalarGrid<int>> nodeFlags(levels);
	std::vector<VectorGrid<double>> nodeWeights(levels);

	myXforms.resize(levels);

	for (int level = 0; level < levels; ++level)
	{
		Transform localXform = myQuadtree.xform(level);
		Vec2i localSize = myQuadtree.size(level);

		myXforms[level] = localXform;

		myNodeLabels[level] = ScalarGrid<NodeActivity>(localXform, localSize, NodeActivity::INACTIVENODE, ScalarGridSettings::SampleType::NODE);
		myNodeValues[level] = VectorGrid<double>(localXform, localSize, 0, VectorGridSettings::SampleType::NODE);

		nodeFlags[level] = ScalarGrid<int>(localXform, localSize, 0, ScalarGridSettings::SampleType::NODE);
		nodeWeights[level] = VectorGrid<double>(localXform, localSize, 0, VectorGridSettings::SampleType::NODE);
	}

	for (int level = 0; level < levels; ++level)
	{
		setActiveNodes(level);
	}

	// Sample values for active nodes
	for (int level = 0; level < levels; ++level)
	{
		sampleActiveNodes(nodeWeights[level], nodeFlags[level], level);
	}

	// Bubble values from active nodes up
	for (int level = 0; level < levels - 1; ++level)
	{
		bubbleActiveNodeValues(nodeWeights, nodeFlags, level);
	}

	// Finish incomplete nodes.
	for (int level = 0; level < levels - 1; ++level)
	{
		finishIncompleteNodes(nodeWeights, nodeFlags, level);
	}

	// Normalize node values based on weight
	for (int level = 0; level < levels; ++level)
	{
		normalizeActiveNodes(nodeWeights[level], nodeFlags[level], level);
	}

	// Distribute node values back down to child nodes
	for (int level = levels - 2; level >= 0; --level)
	{
		distributeNodeValues(level);
	}
}

double QuadtreeVectorInterpolator::biLerp(const Vec2d& worldPos, int axis) const
{
	Vec2d pos = myNodeLabels[0].worldToIndex(worldPos);

	Vec2i cell = floor(pos).cast<int>();

	// Find what grid cell in the tree we fall into
	for (int level = 0; level < myQuadtree.levels(); ++level)
	{
		if (myQuadtree.isCellActive(cell, level))
		{
			Vec2d nodePos = myNodeLabels[level].worldToIndex(worldPos);

			assert(nodePos[0] >= 0 && nodePos[1] >= 0);

			Vec2i bottomCorner = cellToNode(cell, 0);
			Vec2i topCorner = cellToNode(cell, 3);

			Vec2i size = myQuadtree.size(level);

			assert(topCorner[0] <= size[0] && topCorner[1] <= size[1]);

			double nodeValue[4];

			for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
			{
				Vec2i node = cellToNode(cell, nodeIndex);

				if (!(myNodeLabels[level](node) != NodeActivity::INACTIVENODE))
				{
					assert(myNodeLabels[level](node) != NodeActivity::INACTIVENODE);
				}

				nodeValue[nodeIndex] = myNodeValues[level](node, axis);
			}

			Vec2d s = nodePos - floor(nodePos);

			s = clamp(s, Vec2d::Zero().eval(), Vec2d::Ones().eval());

			double lerpValue = lerp(lerp(nodeValue[0], nodeValue[1], s[0]), lerp(nodeValue[2], nodeValue[3], s[0]), s[1]);

			return lerpValue;
		}

		cell = myQuadtree.getParentCell(cell);
	}
	assert(false);
	return 0;
}

void QuadtreeVectorInterpolator::setActiveNodes(int level)
{
	forEachVoxelRange(Vec2i::Zero(), myNodeLabels[level].size(), [&](const Vec2i& node)
	{
		bool nodeActive = false;
		bool nodeInactive = false;

		for (int faceAxis :  {0, 1})
		{
			int offsetAxis = (faceAxis + 1) % 2;

			for (int direction : {0, 1})
			{
				Vec2i face = nodeToFace(node, offsetAxis, direction);

				if (face[offsetAxis] < 0 || face[offsetAxis] >= myIndexGrid[level].size(faceAxis)[offsetAxis])
				{
					nodeInactive = true;
					break;
				}

				int velocityIndex = myIndexGrid[level](face, faceAxis);

				if (velocityIndex >= 0)
				{
					nodeActive = true;
				}

				else if (velocityIndex == COLLISION || velocityIndex == OUTSIDE)
				{
					nodeInactive = true;
					break;
				}
			}
		}

		if (nodeActive && !nodeInactive)
		{
			myNodeLabels[level](node) = NodeActivity::ACTIVENODE;
		}
	});
}

void QuadtreeVectorInterpolator::sampleActiveNodes(VectorGrid<double>& nodeWeights, ScalarGrid<int>& nodeFlags, int level)
{

	double weight = double(1 << (myQuadtree.levels() - level - 1));

	forEachVoxelRange(Vec2i::Zero(), nodeFlags.size(), [&](const Vec2i& node)
	{
		if (myNodeLabels[level](node) == NodeActivity::ACTIVENODE)
		{
			int flag = 0;

			for (int faceAxis : {0, 1})
			{
				int offsetAxis = (faceAxis + 1) % 2;

				double accumulatedValue = 0;
				double accumulatedWeight = 0;

				for (int direction : {0, 1})
				{
					Vec2i face = nodeToFace(node, offsetAxis, direction);

					// We have to clean up weights and flags later.
					if (face[offsetAxis] < 0 ||
						face[offsetAxis] >= myIndexGrid[level].size(faceAxis)[offsetAxis])
					{
						flag += 1 << (faceAxis * 2 + direction);
						accumulatedWeight += weight;
						continue;
					}

					int velocityIndex = myIndexGrid[level](face, faceAxis);

					if (velocityIndex >= 0)
					{
						accumulatedValue += weight * myValueGrid[level](face, faceAxis);
						accumulatedWeight += weight;

						flag += 1 << (faceAxis * 2 + direction);
					}
					else if (velocityIndex != UNASSIGNED)
					{
						assert(level == 0);
						accumulatedWeight += weight;
						flag += 1 << (faceAxis * 2 + direction);
					}
				}

				myNodeValues[level](node, faceAxis) = accumulatedValue;
				nodeWeights(node, faceAxis) = accumulatedWeight;
			}

			nodeFlags(node) = flag;
		}
	});
}

void QuadtreeVectorInterpolator::bubbleActiveNodeValues(std::vector<VectorGrid<double>>& nodeWeights, std::vector<ScalarGrid<int>>& nodeFlags, int level)
{
	forEachVoxelRange(Vec2i::Zero(), myNodeLabels[level].size(), [&](const Vec2i& node)
	{
		if (myNodeLabels[level](node) == NodeActivity::ACTIVENODE)
		{
			int oddCount = (node[0] % 2) + (node[1] % 2);

			// Co-located parent nodes can only exist at even indices.
			if (oddCount == 0)
			{
				Vec2i parentNode = myQuadtree.getParentNode(node);

				if (myNodeLabels[level + 1](parentNode) == NodeActivity::ACTIVENODE)
				{
					int flag = nodeFlags[level](node);
					int parentFlag = nodeFlags[level + 1](parentNode);

					nodeFlags[level + 1](parentNode) = flag + parentFlag;

					assert(!(flag & parentFlag));

					for (int axis : {0, 1})
					{
						nodeWeights[level + 1](parentNode, axis) += nodeWeights[level](node, axis);
						myNodeValues[level + 1](parentNode, axis) += myNodeValues[level](node, axis);
					}

					myNodeLabels[level](node) = NodeActivity::DEPENDENTNODE;
				}
			}
		}
	});
}

void QuadtreeVectorInterpolator::finishIncompleteNodes(std::vector<VectorGrid<double>>& nodeWeights, std::vector<ScalarGrid<int>>& nodeFlags, int level)
{
	double weight = double(1 << (myQuadtree.levels() - level - 1));

	forEachVoxelRange(Vec2i::Zero(), myNodeLabels[level].size(), [&](const Vec2i& node)
	{
		if (myNodeLabels[level](node) == NodeActivity::ACTIVENODE)
		{
			int flag = nodeFlags[level](node);

			// An incomplete flag means there is a face that could not be
			// accessed while bubbling information up. This must be 
			// a t-junction node.
			if (flag != 0xF)
			{
				int oddCount = (node[0] % 2) + (node[1] % 2);

				assert(oddCount == 1);

				int shiftCount = 0;

				int tempFlag = flag;

				while (flag != 0xF && shiftCount < 4)
				{
					if (!(tempFlag & 0x1))
					{
						int faceAxis = shiftCount / 2;
						int offsetAxis = (faceAxis + 1) % 2;
						int direction = shiftCount % 2;

						bool faceFound = false;

						{
							Vec2i face = nodeToFace(node, offsetAxis, direction);

							Vec2i parentFace = myQuadtree.getParentFace(face);
							int faceIndex = myIndexGrid[level + 1](parentFace, faceAxis);

							if (faceIndex >= 0)
							{
								double ghostValue = myValueGrid[level + 1](parentFace, faceAxis);

								myNodeValues[level](node, faceAxis) += weight * ghostValue;

								nodeWeights[level](node, faceAxis) += weight;

								flag += (1 << shiftCount);

								faceFound = true;
							}
						}

						if (!faceFound)
						{
							assert(node[faceAxis] % 2 != 0);

							Vec2i face = nodeToFace(node, offsetAxis, direction);

							double ghostValue = 0;
							Vec2i cell = faceToCell(face, faceAxis, 1);

							int searchLevel = level;

							Vec2i searchCell(cell);

							while (!myQuadtree.isCellActive(searchCell, searchLevel))
							{
								searchCell = myQuadtree.getParentCell(searchCell);
								++searchLevel;
								assert(searchLevel < myQuadtree.levels());
							}

							Vec2d facePos = myValueGrid[level].indexToWorld(face.cast<double>(), faceAxis);
							Vec2d nodePos = myNodeLabels[searchLevel].worldToIndex(facePos);

							double s = nodePos[faceAxis] - std::floor(nodePos[faceAxis]);

							// Average velocities from adjacent big faces
							Vec2i offsetFace = cellToFace(searchCell, faceAxis, 0);

							{
								int offsetFaceIndex = myIndexGrid[searchLevel](offsetFace, faceAxis);

								if (offsetFaceIndex >= 0)
								{
									ghostValue += (1. - s) * myValueGrid[searchLevel](offsetFace, faceAxis);
								}
								else if (offsetFaceIndex == UNASSIGNED)
								{
									for (int childIndex = 0; childIndex < 2; ++childIndex)
									{
										Vec2i childFace = myQuadtree.getChildFace(offsetFace, faceAxis, childIndex);

										int offsetFaceIndex = myIndexGrid[searchLevel - 1](childFace, faceAxis);

										if (offsetFaceIndex >= 0)
										{
											ghostValue += .5 * (1. - s) * myValueGrid[searchLevel - 1](childFace, faceAxis);
										}
										else
										{
											assert(false);
										}
									}
								}
								else
								{
									assert(offsetFaceIndex != COLLISION);
								}
							}

							offsetFace = cellToFace(searchCell, faceAxis, 1);
							{
								int offsetFaceIndex = myIndexGrid[searchLevel](offsetFace, faceAxis);

								if (offsetFaceIndex >= 0)
								{
									ghostValue += s * myValueGrid[searchLevel](offsetFace, faceAxis);
								}
								else if (offsetFaceIndex == UNASSIGNED)
								{
									for (unsigned childIndex = 0; childIndex < 2; ++childIndex)
									{
										Vec2i childFace = myQuadtree.getChildFace(offsetFace, faceAxis, childIndex);

										int offsetFaceIndex = myIndexGrid[searchLevel - 1](childFace, faceAxis);

										if (offsetFaceIndex >= 0)
										{
											ghostValue += .5 * s * myValueGrid[searchLevel - 1](childFace, faceAxis);
										}
										else
										{
											assert(false);
										}
									}
								}
								else
								{
									assert(offsetFaceIndex != COLLISION);
								}
							}

							myNodeValues[level](node, faceAxis) += weight * ghostValue;
							nodeWeights[level](node, faceAxis) += weight;

							flag += 1 << shiftCount;
						}
					}

					++shiftCount;
					tempFlag = tempFlag >> 1;
				}

				assert(flag == 0xF);
				nodeFlags[level](node) = flag;
			}
		}
	});
}

void QuadtreeVectorInterpolator::normalizeActiveNodes(VectorGrid<double> &nodeWeights, ScalarGrid<int>& nodeFlags, int level)
{
	forEachVoxelRange(Vec2i::Zero(), myNodeLabels[level].size(), [&](const Vec2i& node)
	{
		if (myNodeLabels[level](node) == NodeActivity::ACTIVENODE)
		{
			assert(nodeFlags(node) == 0xF);

			for (int axis : {0, 1})
			{
				myNodeValues[level](node, axis) /= nodeWeights(node, axis);
			}
		}
	});
}

void QuadtreeVectorInterpolator::distributeNodeValues(int level)
{
	forEachVoxelRange(Vec2i::Zero(), myNodeLabels[level].size(), [&](const Vec2i& node)
	{
		if (myNodeLabels[level](node) == NodeActivity::DEPENDENTNODE)
		{
			Vec2i parentNode = myQuadtree.getParentNode(node);

			if (myNodeLabels[level + 1](parentNode) == NodeActivity::ACTIVENODE)
			{
				for (int axis : {0, 1})
				{
					myNodeValues[level](node, axis) = myNodeValues[level + 1](parentNode, axis);
				}
			}
			else
			{
				assert(false);
			}

			myNodeLabels[level](node) = NodeActivity::ACTIVENODE;
		}
	});
}

}
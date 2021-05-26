#include "LevelSet.h"

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "GridUtilities.h"

namespace FluidSim2D
{

void LevelSet::drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const
{
	if (doOnlyNarrowBand)
	{
		forEachVoxelRange(Vec2i::Zero(), size(), [&](const Vec2i& cell)
		{
			if (std::fabs(myPhiGrid(cell)) < myNarrowBand)
				myPhiGrid.drawGridCell(renderer, cell);
		});
	}
	else myPhiGrid.drawGrid(renderer);
}

void LevelSet::drawMeshGrid(Renderer& renderer) const
{
	Transform xform(myPhiGrid.dx(), myPhiGrid.offset() + Vec2d(.5, .5) * myPhiGrid.dx());
	ScalarGrid<int> tempGrid(xform, myPhiGrid.size());
	tempGrid.drawGrid(renderer);
}

void LevelSet::drawSupersampledValues(Renderer& renderer, double radius, int samples, double sampleSize) const
{
	myPhiGrid.drawSupersampledValues(renderer, radius, samples, sampleSize);
}
void LevelSet::drawNormals(Renderer& renderer, const Vec3d& colour, double length) const
{
	myPhiGrid.drawSampleGradients(renderer, colour, length);
}

void LevelSet::drawSurface(Renderer& renderer, const Vec3d& colour, double lineWidth) const
{
	EdgeMesh surface = buildMSMesh();
	surface.drawMesh(renderer, colour, lineWidth);
}

void LevelSet::drawDCSurface(Renderer& renderer, const Vec3d& colour, double lineWidth) const
{
	EdgeMesh surface = buildDCMesh();
	surface.drawMesh(renderer, colour, lineWidth);
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.

Vec2d LevelSet::findSurface(const Vec2d& worldPoint, int iterationLimit, double tolerance) const
{
	assert(iterationLimit >= 0);

	double phi = myPhiGrid.biCubicInterp(worldPoint);

	double epsilon = tolerance * dx();
	Vec2d tempPoint = worldPoint;

	int iterationCount = 0;
	if (std::fabs(phi) < myNarrowBand)
	{
		while (std::fabs(phi) > epsilon && iterationCount < iterationLimit)
		{
			tempPoint -= phi * .8 * normal(tempPoint, false);
			phi = myPhiGrid.biCubicInterp(tempPoint);
			++iterationCount;
		}
	}

	return tempPoint;
}

Vec2d LevelSet::findSurfaceIndex(const Vec2d& indexPoint, int iterationLimit, double tolerance) const
{
	Vec2d worldPoint = indexToWorld(indexPoint);
	worldPoint = findSurface(worldPoint, iterationLimit, tolerance);
	return worldToIndex(worldPoint);
}

void LevelSet::initFromMesh(const EdgeMesh& initialMesh, bool doResizeGrid)
{
	// The internal code can't handle a mesh that falls outside of the bounds.
	// If the mesh does, we need to create a new copy and clamp it into the bounds of the grid.

	if (!doResizeGrid)
	{
		bool outOfBounds = false;
		for (const Vec2d& vertex : initialMesh.vertices())
		{
			Vec2d indexPoint = worldToIndex(vertex);
			if (indexPoint[0] <= 0 || indexPoint[1] <= 0 || indexPoint[0] >= size()[0] - 1 || indexPoint[1] >= size()[1] - 1)
				outOfBounds = true;
		}

		if (outOfBounds)
		{
			EdgeMesh clampedMesh = initialMesh;

			// Clamp to be inside grid
			for (int vertIndex = 0; vertIndex < clampedMesh.vertexCount(); ++vertIndex)
			{
				Vec2d vertex = clampedMesh.vertex(vertIndex);
				Vec2d indexPoint = worldToIndex(vertex);

				double offset = 1e-5 * dx();

				indexPoint[0] = std::clamp(indexPoint[0], offset, size()[0] - 1. - offset);
				indexPoint[1] = std::clamp(indexPoint[1], offset, size()[1] - 1. - offset);

				clampedMesh.setVertex(vertIndex, indexToWorld(indexPoint));
			}

			initFromMeshImpl(clampedMesh, doResizeGrid);
			return;
		}
	}

	initFromMeshImpl(initialMesh, doResizeGrid);
}

void LevelSet::initFromMeshImpl(const EdgeMesh& initialMesh, bool doResizeGrid)
{
	if (doResizeGrid)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		AlignedBox2d bbox = initialMesh.boundingBox();;

		// Expand grid beyond the narrow band of the mesh
		double maxPadding = 50. * dx();
		maxPadding = std::min(2. * myNarrowBand, maxPadding);

		bbox.extend(bbox.min() - Vec2d(maxPadding, maxPadding));
		bbox.extend(bbox.max() + Vec2d(maxPadding, maxPadding));

		Vec2d origin = indexToWorld(floor(worldToIndex(bbox.min())).eval());
		Transform xform(dx(), origin);
		Vec2d topRight = indexToWorld(ceil(worldToIndex(bbox.max())).eval());

		// TODO: add the ability to reset grid so we don't have ot re-allocate memory
		myPhiGrid = ScalarGrid<double>(xform, ((topRight - origin) / dx()).cast<int>(), myIsBackgroundNegative ? -myNarrowBand : myNarrowBand);
	}

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED.
	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);
	UniformGrid<int> meshCellParities(size(), 0);

	for (const Vec2i& edge : initialMesh.edges())
	{
		const Vec2d& startPoint = worldToIndex(initialMesh.vertex(edge[0]));
		const Vec2d& endPoint = worldToIndex(initialMesh.vertex(edge[1]));

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		AlignedBox2d edgeBbox;
		edgeBbox.extend(startPoint);
		edgeBbox.extend(endPoint);

		Vec2i edgeCeilMin = ceil(edgeBbox.min()).cast<int>();
		Vec2i edgeFloorMin = floor(edgeBbox.min()).cast<int>() - Vec2i::Ones();
		Vec2i edgeFloorMax = floor(edgeBbox.max()).cast<int>();

		for (int j = edgeCeilMin[1]; j <= edgeFloorMax[1]; ++j)
			for (int i = edgeFloorMax[0]; i >= edgeFloorMin[0]; --i)
			{
				Vec2d gridNode(i, j);
				IntersectionLabels intersectionResult = exactEdgeIntersect(startPoint, endPoint, gridNode, Axis::XAXIS);

				assert(gridNode[0] >= 0 && gridNode[1] >= 0 && gridNode[0] < myPhiGrid.size()[0] - 1 &&
					gridNode[1] < myPhiGrid.size()[1] - 1);

				if (intersectionResult == IntersectionLabels::NO) continue;

				// Increment the parity since the grid_node is
				// "left" of the mesh-edge crossing the grid-edge.
				// This indicates a negative normal in the x-direction
				// and means we're entering into the material.
				int parityChange = -1;
				if (startPoint[1] < endPoint[1])
					parityChange = 1;

				if (intersectionResult == IntersectionLabels::YES)
					meshCellParities(i + 1, j) += parityChange;
				else
				{
					assert(intersectionResult == IntersectionLabels::ON);

					if (parityChange == 1)
						meshCellParities(i, j) += parityChange;
					else
					{
						assert(parityChange == -1);
						meshCellParities(i + 1, j) += parityChange;
					}
				}

				break;
			}
	}

	// Now that all the x-axis edge crossings have been found, we can compile the parity changes
	// and label grid nodes that are at the interface
	tbb::parallel_for(tbb::blocked_range<int>(0, size()[1], tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int j = range.begin(); j != range.end(); ++j)
		{
			int parity = myIsBackgroundNegative ? 1 : 0;

			// We loop x-major because that's how we've set up our edge intersection.
			for (int i = 0; i < size()[0]; ++i) // TODO: double check that I've resized right
			{
				// Update parity before changing sign since the parity values above used the convention
				// of putting the change on the "far" node (i.e. after the mesh-grid intersection).
				Vec2i cell(i, j);
				parity += meshCellParities(cell);
				meshCellParities(cell) = parity;

				if (parity > 0)
					myPhiGrid(cell) = -myNarrowBand;
				else
					myPhiGrid(cell) = myNarrowBand;
			}

			assert(myIsBackgroundNegative ? parity == 1 : parity == 0);
		}			
	});


	// With the parity assigned, loop over the grid once more and label nodes that have an implied sign change
	// with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
    tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
        {
			Vec2i cell = unflatten(flatIndex);

			if (cell[0] == 0 || cell[0] == size()[0] - 1 || cell[1] == 0 || cell[1] == size()[1] - 1)
                continue;

            bool isCellInside = meshCellParities(cell) > 0;

            for (int axis : {0, 1})
                for (int direction : {0, 1})
                {
                    Vec2i adjacentCell = cellToCell(cell, axis, direction);

                    bool isAdjacentCellInside = meshCellParities(adjacentCell) > 0;

                    if (isCellInside != isAdjacentCellInside)
                        reinitializedCells(cell) = VisitedCellLabels::FINISHED_CELL;
                }
            
		}
    });

	// Loop over all the edges in the mesh. Level set grid cells labelled as VISITED will be
	// updated with the distance to the surface if it happens to be shorter than the current
	// distance to the surface.
	for (const Vec2i& edge : initialMesh.edges())
	{
		// Using the vertices of the edge, we can update distance values for cells
		// within the bounding box of the mesh. It's easier to work in our index space
		// and just scale the distance later.
		const Vec2d& startPoint = worldToIndex(initialMesh.vertex(edge[0]));
		const Vec2d& endPoint = worldToIndex(initialMesh.vertex(edge[1]));

		// Build bounding box
		AlignedBox2i edgeBbox;
		edgeBbox.extend(floor(startPoint).cast<int>());
		edgeBbox.extend(ceil(endPoint).cast<int>());

		// Expand outward by 2-voxels in each direction
		edgeBbox.extend(edgeBbox.min() - Vec2i(2, 2));
		edgeBbox.extend(edgeBbox.max() + Vec2i(2, 2));

		AlignedBox2i clampBBox;
		clampBBox.extend(Vec2i::Zero());
		clampBBox.extend(size() - Vec2i::Ones());

		edgeBbox.clamp(clampBBox);

		// Update distances to the mesh at grid cells within the bounding box
		assert(edgeBbox.min()[0] >= 0 && edgeBbox.min()[1] >= 0 && edgeBbox.max()[0] < size()[0] && edgeBbox.max()[1] < size()[1]);

		forEachVoxelRange(edgeBbox.min(), edgeBbox.max() + Vec2i::Ones(), [&](const Vec2i& cell)
		{
			if (reinitializedCells(cell) != VisitedCellLabels::UNVISITED_CELL)
			{
				Vec2d cellPoint = cell.cast<double>();
				Vec2d vec0 = endPoint - startPoint;
				Vec2d vec1 = cellPoint - startPoint;

				double s = vec0.dot(vec1) / vec0.dot(vec0); // Find projection along edge.
				s = std::clamp(s, double(0), double(1));

				Vec2d projPoint = startPoint + s * vec0;
				double dist = (projPoint - cellPoint).norm() * dx();

				// Update if the distance to this edge is shorter than previous values.
				if (dist < std::fabs(myPhiGrid(cell)))
				{
					// If the parity says the node is inside, set it to be negative
					myPhiGrid(cell) = (meshCellParities(cell) > 0) ? -dist : dist;
				}
			}
		});
	}

	reinitFastMarching(reinitializedCells);
}

void LevelSet::reinitFastMarching(UniformGrid<VisitedCellLabels>& reinitializedCells)
{
	assert(reinitializedCells.size() == size());

	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward
	auto solveEikonal = [&](const Vec2i& cell) -> double
	{
		double max = std::numeric_limits<double>::max();

		Vec2d Uaxis(max, max);
		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i adjacentCell = cellToCell(cell, axis, direction);

				if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis])
				{
					assert(myPhiGrid(cell) > 0 && !myIsBackgroundNegative || myPhiGrid(cell) < 0 && myIsBackgroundNegative);
					Uaxis[axis] = max;
				}
				else
				{
					Uaxis[axis] = std::min(std::fabs(myPhiGrid(adjacentCell)), Uaxis[axis]);
				}
			}

		double U;

		double Uh = Uaxis[0];
		double Uv = Uaxis[1];
		
		if (std::fabs(Uh - Uv) >= dx())
			U = std::min(Uh, Uv) + dx();
		else
		{
			// Quadratic equation from the Eikonal
			double rootEntry = std::pow(Uh + Uv, 2) - 2. * (std::pow(Uh, 2) + std::pow(Uv, 2) - std::pow(dx(), 2));
			assert(rootEntry >= 0);
			U = .5 * (Uh + Uv) + .5 * std::sqrt(rootEntry);
		}

		return U;
	};

	// Load up the BFS queue with the unvisited cells next to the finished ones
	using Node = std::pair<Vec2i, double>;
	auto cmp = [](const Node& a, const Node& b) -> bool { return std::fabs(a.second) > std::fabs(b.second); };
	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> marchingQ(cmp);

	forEachVoxelRange(Vec2i::Zero(), size(), [&](const Vec2i& cell)
	{
		if (reinitializedCells(cell) == VisitedCellLabels::FINISHED_CELL)
		{
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

					if (reinitializedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
					{
						double dist = solveEikonal(adjacentCell);
						assert(dist >= 0);

						myPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) < 0 ? -dist : dist;

						Node node(adjacentCell, dist);

						marchingQ.push(node);
						reinitializedCells(adjacentCell) = VisitedCellLabels::VISITED_CELL;
					}
				}
		}
	});

	while (!marchingQ.empty())
	{
		Node localNode = marchingQ.top();
		Vec2i localCell = localNode.first;
		marchingQ.pop();

		// Since you can't just update parts of the priority queue,
		// it's possible that a cell has been solidified at a smaller distance
		// and an older insert if floating around.
		if (reinitializedCells(localCell) == VisitedCellLabels::FINISHED_CELL)
		{
			// Make sure that the distance assigned to the cell is smaller than
			// what is floating around
			assert(std::fabs(myPhiGrid(localCell)) <= std::fabs(localNode.second));
			continue;
		}
		assert(reinitializedCells(localCell) == VisitedCellLabels::VISITED_CELL);

		if (std::fabs(myPhiGrid(localCell)) < myNarrowBand)
		{
			// Debug check that there is indeed a FINISHED cell next to it
			bool foundFinishedCell = false;

			// Loop over the neighbouring cells and load the unvisited cells
			// and update the visited cells
			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(localCell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= reinitializedCells.size()[axis])
						continue;

					if (reinitializedCells(adjacentCell) == VisitedCellLabels::FINISHED_CELL)
						foundFinishedCell = true;
					else // If visited, then we'll update it
					{
						double dist = solveEikonal(adjacentCell);
						assert(dist >= 0);
						
						if (dist > myNarrowBand) dist = myNarrowBand;

						if (reinitializedCells(adjacentCell) == VisitedCellLabels::VISITED_CELL && dist > std::fabs(myPhiGrid(adjacentCell)))
							continue;

						myPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) < 0 ? -dist : dist;

						Node node(adjacentCell, dist);

						marchingQ.push(node);
						reinitializedCells(adjacentCell) = VisitedCellLabels::VISITED_CELL;
					}
				}

			//Check that a marked cell was indeed visited
			assert(foundFinishedCell);
		}
		// Clamp to narrow band
		else myPhiGrid(localCell) = myPhiGrid(localCell) < 0 ? -myNarrowBand : myNarrowBand;

		// Solidify cell now that we've handled all it's neighbours
		reinitializedCells(localCell) = VisitedCellLabels::FINISHED_CELL;
	}
}

EdgeMesh LevelSet::buildMSMesh() const
{
	VecVec2d verts;

	VectorGrid<int> vertexIndexGrid(xform(), size() - Vec2i::Ones(), -1, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i::Zero(), vertexIndexGrid.size(axis), [&](const Vec2i& face)
		{
			Vec2i startNode = faceToNode(face, axis, 0);
			Vec2i endNode = faceToNode(face, axis, 1);

			if (myPhiGrid(startNode) <= 0 && myPhiGrid(endNode) > 0 ||
				myPhiGrid(startNode) > 0 && myPhiGrid(endNode) <= 0)
			{
				Vec2d indexPoint = interpolateInterface(startNode, endNode);
				vertexIndexGrid(face, axis) = int(verts.size());
				verts.push_back(indexToWorld(indexPoint));
			}	
		});
	}

	VecVec2i edges;

	// Run marching squares loop
	forEachVoxelRange(Vec2i::Zero(), size() - Vec2i::Ones(), [&](const Vec2i& cell)
	{
		int mcKey = 0;

		for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
		{
			Vec2i node = cellToNodeCCW(cell, nodeIndex);
			if (myPhiGrid(node) <= 0.) mcKey += (1 << nodeIndex);
		}

		// Connect edges using the marching squares template
		for (int edgeIndex = 0; edgeIndex < 4 && marchingSquaresTemplate[mcKey][edgeIndex] >= 0; edgeIndex += 2)
		{
			// Find first vertex
			int edge = marchingSquaresTemplate[mcKey][edgeIndex];

			Vec3i faceMap = cellToFaceCCW(cell, edge);

			Vec2i face(faceMap[0], faceMap[1]);
			int axis = faceMap[2];

			int firstVertxIndex = vertexIndexGrid(face, axis);
			assert(firstVertxIndex >= 0);

			// Find second vertex
			edge = marchingSquaresTemplate[mcKey][edgeIndex + 1];
			faceMap = cellToFaceCCW(cell, edge);

			face = Vec2i(faceMap[0], faceMap[1]);
			axis = faceMap[2];

			int secondVertxIndex = vertexIndexGrid(face, axis);
			assert(secondVertxIndex >= 0);

			edges.emplace_back(firstVertxIndex, secondVertxIndex);
		}
	});

	return EdgeMesh(edges, verts);
}

// Extract a mesh representation of the interface using dual contouring
EdgeMesh LevelSet::buildDCMesh() const
{
	VecVec2d verts;
	VecVec2i edges;

	// Create grid to store index to dual contouring point. Note that phi is
	// center sampled so the DC grid must be node sampled and one cell shorter
	// in each dimension
	UniformGrid<int> dcPointIndex(size() - Vec2i::Ones(), -1);

	// Run dual contouring loop
	forEachVoxelRange(Vec2i::Zero(), dcPointIndex.size(), [&](const Vec2i& cell)
	{
		Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(2, 2);

		Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2);

		Eigen::VectorXd pointCOM = Eigen::VectorXd::Zero(2);

		double pointCount = 0;

		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i face = cellToFace(cell, axis, direction);

				Vec2i backwardNode = faceToNode(face, axis, 0);
				Vec2i forwardNode = faceToNode(face, axis, 1);

				if ((myPhiGrid(backwardNode) <= 0 && myPhiGrid(forwardNode) > 0) ||
					(myPhiGrid(backwardNode) > 0 && myPhiGrid(forwardNode) <= 0))
				{
					// Find interface point
					Vec2d point = interpolateInterface(backwardNode, forwardNode);

					assert(point[0] >= backwardNode[0] && point[0] <= forwardNode[0] &&
						point[1] >= backwardNode[1] && point[1] <= forwardNode[1]);

					// Find associated surface normal
					Vec2d localNormal = normal(indexToWorld(point), false);

					AtA(0, 0) += localNormal[0] * localNormal[0];
					AtA(0, 1) += localNormal[1] * localNormal[0];
					AtA(1, 0) += localNormal[0] * localNormal[1];
					AtA(1, 1) += localNormal[1] * localNormal[1];

					double norm_dot = localNormal.dot(point);
					rhs(0) += localNormal[0] * norm_dot;
					rhs(1) += localNormal[1] * norm_dot;

					pointCOM[0] += point[0];
					pointCOM[1] += point[1];

					++pointCount;
				}
			}

		if (pointCount > 0)
		{
			pointCOM.array() /= pointCount;

			// Add zero-length spring to COM
			double comWeight = .00001;
			AtA(0, 0) += comWeight;
			AtA(1, 1) += comWeight;

			rhs += comWeight * pointCOM;

			Eigen::VectorXd newQEFPoint = AtA.fullPivLu().solve(rhs);

			if (newQEFPoint[0] < cell[0] || newQEFPoint[1] < cell[1] || newQEFPoint[0] >= cell[0] + 1 || newQEFPoint[1] >= cell[1] + 1)
			{
				newQEFPoint = pointCOM;
			}


			verts.push_back(indexToWorld(newQEFPoint.block(0, 0, 2, 1)));
			dcPointIndex(cell) = int(verts.size()) - 1;
		}
	});

	for (int axis : {0, 1})
	{
		Vec2i start(0,0); ++start[axis];
		Vec2i end(dcPointIndex.size());

		forEachVoxelRange(start, end, [&](const Vec2i& face)
		{
			Vec2i backwardNode = faceToNode(face, axis, 0);
			Vec2i forwardNode = faceToNode(face, axis, 1);

			if ((myPhiGrid(backwardNode) <= 0 && myPhiGrid(forwardNode) > 0) ||
				(myPhiGrid(backwardNode) > 0 && myPhiGrid(forwardNode) <= 0))
			{
				Vec2i backwardCell = faceToCell(Vec2i(face), axis, 0);
				Vec2i forwardCell = faceToCell(Vec2i(face), axis, 1);

				assert(dcPointIndex(backwardCell) >= 0 && dcPointIndex(forwardCell) >= 0);

				Vec2i edge;
				if (myPhiGrid(backwardNode) <= 0.)
				{
					if (axis == 0)
						edge = Vec2i(dcPointIndex(backwardCell), dcPointIndex(forwardCell));
					else
						edge = Vec2i(dcPointIndex(forwardCell), dcPointIndex(backwardCell));
				}
				else
				{
					if (axis == 0)
						edge = Vec2i(dcPointIndex(forwardCell), dcPointIndex(backwardCell));
					else
						edge = Vec2i(dcPointIndex(backwardCell), dcPointIndex(forwardCell));
				}

				edges.push_back(edge);
			}
		});
	}

	return EdgeMesh(edges, verts);
}

Vec2d LevelSet::interpolateInterface(const Vec2i& startPoint, const Vec2i& endPoint) const
{
	assert((myPhiGrid(startPoint) <= 0 && myPhiGrid(endPoint) > 0) ||
			(myPhiGrid(startPoint) > 0 && myPhiGrid(endPoint) <= 0));

	double theta = lengthFraction(myPhiGrid(startPoint), myPhiGrid(endPoint));
	
	assert(theta >= 0 && theta <= 1);

	if (myPhiGrid(startPoint) > 0)
		theta = 1. - theta;

	return startPoint.cast<double>() + theta * (endPoint - startPoint).cast<double>();
}

void LevelSet::unionSurface(const LevelSet& unionPhi)
{
	assert(isGridMatched(unionPhi));


	tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
            Vec2i cell = unflatten(flatIndex);
            myPhiGrid(cell) = std::min(myPhiGrid(cell), unionPhi(cell));
        }
	});

	reinit();
}

}
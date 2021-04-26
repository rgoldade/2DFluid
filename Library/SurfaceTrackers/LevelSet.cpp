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

Vec2d LevelSet::findSurface(const Vec2d& worldPoint, int iterationLimit) const
{
	assert(iterationLimit >= 0);

	double phi = myPhiGrid.biLerp(worldPoint);

	double epsilon = 1E-2 * dx();
	Vec2d tempPoint = worldPoint;

	int iterationCount = 0;
	if (std::fabs(phi) < myNarrowBand)
	{
		while (std::fabs(phi) > epsilon && iterationCount < iterationLimit)
		{
			tempPoint -= phi * normal(tempPoint);
			phi = myPhiGrid.biCubicInterp(tempPoint);
			++iterationCount;
		}
	}

	return tempPoint;
}

Vec2d LevelSet::findSurfaceIndex(const Vec2d& indexPoint, int iterationLimit) const
{
	Vec2d worldPoint = indexToWorld(indexPoint);
	worldPoint = findSurface(worldPoint, iterationLimit);
	return worldToIndex(worldPoint);
}

void LevelSet::reinit()
{
	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);

	// Find the zero crossings, update their distances and flag as source cells
	ScalarGrid<double> tempPhiGrid = myPhiGrid;

	tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount()), [&](const tbb::blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = reinitializedCells.unflatten(cellIndex);

			// Check for a zero crossing
			bool isAtZeroCrossing = false;

			Vec2d distToZero = Vec2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

			for (int axis : {0, 1})
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis])
					{
						if (myPhiGrid(cell) > 0 && myIsBackgroundNegative)
						{
							distToZero[axis] = 0;
							isAtZeroCrossing = true;
						}
						else if (myPhiGrid(cell) <= 0 && !myIsBackgroundNegative)
						{
							distToZero[axis] = 0;
							isAtZeroCrossing = true;
						}
					}
					else if ((myPhiGrid(cell) <= 0 && myPhiGrid(adjacentCell) > 0) ||
								(myPhiGrid(cell) > 0 && myPhiGrid(adjacentCell) <= 0))
					{
						isAtZeroCrossing = true;

						// Compute length fraction to zero crossing
						double theta = lengthFraction(myPhiGrid(cell), myPhiGrid(adjacentCell));

						if (myPhiGrid(cell) > 0)
							theta = 1. - theta;

						assert(theta >= 0 && theta <= 1);

						distToZero[axis] = min(theta * dx(), distToZero[axis]);
					}
				}

			if (isAtZeroCrossing)
			{
				if (distToZero[0] == 0 || distToZero[1] == 0)
				{
					tempPhiGrid(cell) = 0;
				}
				else
				{
					assert(distToZero[0] > 0 && distToZero[1] > 0);
					if (distToZero[0] == std::numeric_limits<double>::max())
					{
						assert(distToZero[1] < std::numeric_limits<double>::max());
						tempPhiGrid(cell) = distToZero[1];
					}
					else if (distToZero[1] == std::numeric_limits<double>::max())
					{
						assert(distToZero[0] < std::numeric_limits<double>::max());
						tempPhiGrid(cell) = distToZero[0];
					}
					else
					{
						assert(distToZero[0] < std::numeric_limits<double>::max() &&
								distToZero[1] < std::numeric_limits<double>::max());

						tempPhiGrid(cell) = std::sqrt((distToZero[0] * distToZero[1]) / (std::pow(distToZero[0], 2) + std::pow(distToZero[1], 2)));
					}
				}

				reinitializedCells(cell) = VisitedCellLabels::FINISHED_CELL;
			}
			// Set unvisited grid cells to background value, using old grid for inside/outside sign
			else
			{
				assert(reinitializedCells(cell) == VisitedCellLabels::UNVISITED_CELL);
				tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -myNarrowBand : myNarrowBand;
			}
		}
	});

	std::swap(myPhiGrid, tempPhiGrid);
	reinitFastMarching(reinitializedCells);
}

void LevelSet::initFromMesh(const EdgeMesh& initialMesh, bool doResizeGrid)
{
	if (doResizeGrid)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		AlignedBox2d bbox;

		for (int vertexIndex = 0; vertexIndex < initialMesh.vertexCount(); ++vertexIndex)
			bbox.extend(initialMesh.vertex(vertexIndex));

		// Just for nice whole numbers, let's clamp the bounding box to be an integer
		// offset in index space then bring it back to world space
		double maxNarrowBand = 10.;
		maxNarrowBand = std::min(myNarrowBand / dx(), maxNarrowBand);

		AlignedBox2d scaledBBox;
		scaledBBox.extend(dx() * (floor((bbox.max() / dx()).eval()) - 2. * Vec2d(maxNarrowBand, maxNarrowBand)));
		scaledBBox.extend(dx() * (ceil((bbox.max() / dx()).eval()) + 2. * Vec2d(maxNarrowBand, maxNarrowBand)));

		clear();
		Transform xform(dx(), scaledBBox.min());
		// Since we know how big the mesh is, we know how big our grid needs to be (wrt to grid spacing)
		myPhiGrid = ScalarGrid<double>(xform, ((scaledBBox.max() - scaledBBox.max()) / dx()).cast<int>(), myNarrowBand);
	}
	else
		myPhiGrid.resize(size(), myNarrowBand);

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED.
	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);
	UniformGrid<int> meshCellParities(size(), 0);

	for (const Vec2i& edge : initialMesh.edges())
	{
		// It's easier to work in our index space and just scale the distance later.
		const Vec2d& startPoint = worldToIndex(initialMesh.vertex(edge[0]));
		const Vec2d& endPoint = worldToIndex(initialMesh.vertex(edge[1]));

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		AlignedBox2d vertBBox;
		vertBBox.extend(startPoint);
		vertBBox.extend(endPoint);

		Vec2i edgeCeilMin = ceil(vertBBox.min()).cast<int>();
		Vec2i edgeFloorMin = floor(vertBBox.min()).cast<int>() - Vec2i::Ones();
		Vec2i edgeFloorMax = floor(vertBBox.max()).cast<int>();

		for (int j = edgeCeilMin[1]; j <= edgeFloorMax[1]; ++j)
			for (int i = edgeFloorMax[0]; i >= edgeFloorMin[0]; --i)
			{
				Vec2d gridNode(i, j);
				IntersectionLabels intersectionResult = exactEdgeIntersect(startPoint, endPoint, gridNode, Axis::XAXIS);

				// TODO: remove once test complete
				if (gridNode[0] < 0 || gridNode[1] < 0 || gridNode[0] >= myPhiGrid.size()[0] ||
					gridNode[1] >= myPhiGrid.size()[1])
				{
					std::cout << "Caught out of bounds. Node: " << gridNode[0] << " " << gridNode[1] << std::endl;
				}

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
				// If the grid node is explicitly on the mesh-edge, set distance to zero
				// since it might not be exactly zero due to floating point error above.
				else
				{
					assert(intersectionResult == IntersectionLabels::ON);
					// Technically speaking, the zero isocountour means we're inside
					// the surface. So we should change the parity at the node that
					// is intersected even though it's zero and the sign is meaningless.
					reinitializedCells(i, j) = VisitedCellLabels::FINISHED_CELL;
					myPhiGrid(i, j) = 0.;
					meshCellParities(i, j) += parityChange;
				}

				break;
			}
	}

	// Now that all the x-axis edge crossings have been found, we can compile the parity changes
	// and label grid nodes that are at the interface
	for (int j = 0; j < size()[1]; ++j)
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

			// Set inside cells to negative
			if (parity > 0) myPhiGrid(cell) = -std::fabs(myPhiGrid(cell));
		}

		assert(myIsBackgroundNegative ? parity == 1 : parity == 0);
	}

	// With the parity assigned, loop over the grid once more and label nodes that have an implied sign change
	// with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
	forEachVoxelRange(Vec2i::Ones(), size() - Vec2i::Ones(), [&](const Vec2i& cell)
	{
		bool isCellInside = meshCellParities(cell) > 0;

		for (int axis : {0, 1})
			for (int direction : {0, 1})
			{
				Vec2i adjacentCell = cellToCell(cell, axis, direction);

				bool isAdjacentCellInside = meshCellParities(adjacentCell) > 0;

				if (isCellInside != isAdjacentCellInside)
					reinitializedCells(cell) = VisitedCellLabels::FINISHED_CELL;
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
		edgeBbox.extend(floor(endPoint).cast<int>());

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
				Vec2d vec0 = cellPoint - startPoint;
				Vec2d vec1 = endPoint - startPoint;

				double s = vec0.dot(vec1) / vec1.dot(vec1); // Find projection along edge.
				s = std::clamp(s, double(0), double(1));

				// Remove on-edge projection to get vector from closest point on edge to cell point.
				double surfaceDistance = (vec0 - s * vec1).norm() * dx();

				// Update if the distance to this edge is shorter than previous values.
				if (surfaceDistance < std::fabs(myPhiGrid(cell)))
				{
					// If the parity says the node is inside, set it to be negative
					myPhiGrid(cell) = (meshCellParities(cell) > 0) ? -surfaceDistance : surfaceDistance;
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
					assert(myPhiGrid(cell) > 0 && !myIsBackgroundNegative || myPhiGrid(cell) <= 0 && myIsBackgroundNegative);
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

		return double(U);
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

						myPhiGrid(adjacentCell) = (myPhiGrid(adjacentCell) < 0.) ? -dist : dist;

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

// Extract a mesh representation of the interface. Useful for rendering but not much
// else since there will be duplicate vertices per grid edge in the current implementation.

EdgeMesh LevelSet::buildMSMesh() const
{
	VecVec2d verts;
	VecVec2i edges;

	// Run marching squares loop
	forEachVoxelRange(Vec2i::Zero(), size() - Vec2i::Ones(), [&](const Vec2i& cell)
	{
		int mcKey = 0;

		for (int direction = 0; direction < 4; ++direction)
		{
			Vec2i node = cellToNodeCCW(cell, direction);
			if (myPhiGrid(node) <= 0.) mcKey += (1 << direction);
		}

		// Connect edges using the marching squares template
		for (int edgeIndex = 0; edgeIndex < 4 && marchingSquaresTemplate[mcKey][edgeIndex] >= 0; edgeIndex += 2)
		{
			// Find first vertex
			int edge = marchingSquaresTemplate[mcKey][edgeIndex];

			Vec3i faceMap = cellToFaceCCW(cell, edge);

			Vec2i face(faceMap[0], faceMap[1]);
			int axis = faceMap[2];
			Vec2i startNode = faceToNode(face, axis, 0);
			Vec2i endNode = faceToNode(face, axis, 1);

			Vec2d startPoint = interpolateInterface(startNode, endNode);

			// Find second vertex
			edge = marchingSquaresTemplate[mcKey][edgeIndex + 1];
			faceMap = cellToFaceCCW(cell, edge);

			face = Vec2i(faceMap[0], faceMap[1]);
			axis = faceMap[2];

			startNode = faceToNode(face, axis, 0);
			endNode = faceToNode(face, axis, 1);

			Vec2d endPoint = interpolateInterface(startNode, endNode);

			// Store vertices
			Vec2d worldStartPoint = indexToWorld(startPoint);
			Vec2d worldEndPoint = indexToWorld(endPoint);

			verts.push_back(worldStartPoint);
			verts.push_back(worldEndPoint);

			edges.emplace_back(verts.size() - 2, verts.size() - 1);
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
		VecVec2d qefPoints;
		VecVec2d qefNormals;

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
					Vec2d interfacePoint = interpolateInterface(backwardNode, forwardNode);
					qefPoints.push_back(interfacePoint);

					// Find associated surface normal
					Vec2d surfaceNormal = normal(indexToWorld(interfacePoint));
					qefNormals.push_back(surfaceNormal);
				}
			}

		if (qefPoints.size() > 0)
		{
			Eigen::MatrixXd A(qefPoints.size(), 2);
			Eigen::VectorXd b(qefPoints.size());
			Eigen::VectorXd pointCOM = Eigen::VectorXd::Zero(2);

			assert(qefPoints.size() > 1);

			for (int pointIndex = 0; pointIndex < qefPoints.size(); ++pointIndex)
			{
				A(pointIndex, 0) = qefNormals[pointIndex][0];
				A(pointIndex, 1) = qefNormals[pointIndex][1];

				b(pointIndex) = qefNormals[pointIndex].dot(qefPoints[pointIndex]);

				pointCOM[0] += qefPoints[pointIndex][0];
				pointCOM[1] += qefPoints[pointIndex][1];
			}

			pointCOM /= double(qefPoints.size());

			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			svd.setThreshold(1E-2);

			Eigen::VectorXd dcPoint = pointCOM + svd.solve(b - A * pointCOM);

			Vec2d vecCOM(pointCOM[0], pointCOM[1]);

			Vec2d boundingBoxMin = floor(vecCOM);
			Vec2d boundingBoxMax = ceil(vecCOM);

			if (dcPoint[0] < boundingBoxMin[0] ||
				dcPoint[1] < boundingBoxMin[1] ||
				dcPoint[0] > boundingBoxMax[0] ||
				dcPoint[1] > boundingBoxMax[1])
				dcPoint = pointCOM;

			verts.push_back(indexToWorld(Vec2d(dcPoint[0], dcPoint[1])));
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
	assert((myPhiGrid(startPoint[0], startPoint[1]) <= 0 && myPhiGrid(endPoint[0], endPoint[1]) > 0) ||
			(myPhiGrid(startPoint[0], startPoint[1]) > 0 && myPhiGrid(endPoint[0], endPoint[1]) <= 0));

	double theta = lengthFraction(myPhiGrid(startPoint), myPhiGrid(endPoint));
	
	assert(theta >= 0 && theta <= 1);

	return startPoint.cast<double>() + theta * (endPoint - startPoint).cast<double>();
}

void LevelSet::unionSurface(const LevelSet& unionPhi)
{
	assert(isGridMatched(unionPhi));

	forEachVoxelRange(Vec2i::Zero(), size(), [&](const Vec2i& cell)
	{
		if (std::fabs(unionPhi(cell)) < myNarrowBand)
			myPhiGrid(cell) = std::min(myPhiGrid(cell), unionPhi(cell));
	});

	reinit();
}

}
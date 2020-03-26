#include "LevelSet.h"

#include <iostream>

#include "tbb/tbb.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace FluidSim2D::SurfaceTrackers
{

void LevelSet::drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const
{
	if (doOnlyNarrowBand)
	{
		forEachVoxelRange(Vec2i(0), size(), [&](const Vec2i& cell)
		{
			if (std::fabs(myPhiGrid(cell)) < myNarrowBand)
				myPhiGrid.drawGridCell(renderer, cell);
		});
	}
	else myPhiGrid.drawGrid(renderer);
}

void LevelSet::drawMeshGrid(Renderer& renderer) const
{
	Transform xform(myPhiGrid.dx(), myPhiGrid.offset() + Vec2f(0.5) * myPhiGrid.dx());
	ScalarGrid<int> tempGrid(xform, myPhiGrid.size());
	tempGrid.drawGrid(renderer);
}

void LevelSet::drawSupersampledValues(Renderer& renderer, float radius, int samples, float sampleSize) const
{
	myPhiGrid.drawSupersampledValues(renderer, radius, samples, sampleSize);
}
void LevelSet::drawNormals(Renderer& renderer, const Vec3f& colour, float length) const
{
	myPhiGrid.drawSampleGradients(renderer, colour, length);
}

void LevelSet::drawSurface(Renderer& renderer, const Vec3f& colour, float lineWidth) const
{
	EdgeMesh surface = buildMSMesh();
	surface.drawMesh(renderer, colour, lineWidth);
}

void LevelSet::drawDCSurface(Renderer& renderer, const Vec3f& colour, float lineWidth) const
{
	EdgeMesh surface = buildDCMesh();
	surface.drawMesh(renderer, colour, lineWidth);
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.

Vec2f LevelSet::findSurface(const Vec2f& worldPoint, int iterationLimit) const
{
	assert(iterationLimit >= 0);

	float phi = myPhiGrid.biLerp(worldPoint);

	float epsilon = 1E-2 * dx();
	Vec2f tempPoint = worldPoint;

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

Vec2f LevelSet::findSurfaceIndex(const Vec2f& indexPoint, int iterationLimit) const
{
	Vec2f worldPoint = indexToWorld(indexPoint);
	worldPoint = findSurface(worldPoint, iterationLimit);
	return worldToIndex(worldPoint);
}

void LevelSet::reinit(bool rebuildWithFIM)
{
	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);

	// Find the zero crossings, update their distances and flag as source cells
	ScalarGrid<float> tempPhiGrid = myPhiGrid;

	tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = reinitializedCells.unflatten(cellIndex);

			// Check for a zero crossing
			bool isAtZeroCrossing = false;
			for (int axis = 0; axis < 2 && !isAtZeroCrossing; ++axis)
				for (int direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(cell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

					if ((myPhiGrid(cell) <= 0 && myPhiGrid(adjacentCell) > 0) ||
						(myPhiGrid(cell) > 0 && myPhiGrid(adjacentCell) <= 0))
					{
						isAtZeroCrossing = true;

						Vec2f worldPoint = indexToWorld(Vec2f(cell));
						Vec2f interfacePoint = findSurface(worldPoint, 5);

						float distance = dist(worldPoint, interfacePoint);

						tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -distance : distance;
						reinitializedCells(cell) = VisitedCellLabels::FINISHED_CELL;

						break;
					}
				}

			// Set unvisited grid cells to background value, using old grid for inside/outside sign
			if (!isAtZeroCrossing)
			{
				assert(reinitializedCells(cell) == VisitedCellLabels::UNVISITED_CELL);
				tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -myNarrowBand : myNarrowBand;
			}
		}
	});

	//std::swap(myPhiGrid, tempPhiGrid);
	myPhiGrid = tempPhiGrid;

	if (rebuildWithFIM)
		reinitFastIterative(reinitializedCells);
	else
		reinitFastMarching(reinitializedCells);
}

void LevelSet::reinitFastIterative(UniformGrid<VisitedCellLabels>& reinitializedCells)
{
	assert(reinitializedCells.size() == size());

	//
	// Before starting the iterations, we want to construct the active list of voxels
	// to reinitialize.
	//

	tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelActiveCellList;

	tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		std::vector<Vec2i>& localActiveCellList = parallelActiveCellList.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = reinitializedCells.unflatten(cellIndex);

			if (reinitializedCells(cell) == VisitedCellLabels::FINISHED_CELL)
			{
				// Add neighbours to the list
				for (int axis : {0, 1})
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

						if (reinitializedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
						{
							localActiveCellList.push_back(adjacentCell);
							reinitializedCells(adjacentCell) = VisitedCellLabels::VISITED_CELL;
						}
					}
			}
		}
	});

	std::vector<Vec2i> activeCellList;
	mergeLocalThreadVectors(activeCellList, parallelActiveCellList);

	parallelActiveCellList.clear();

	auto vecCompare = [](const Vec2i& a, const Vec2i& b) -> bool
	{
		if (a[0] < b[0]) return true;
		else if (a[0] == b[0] && a[1] < b[1]) return true;
		return false;
	};

	tbb::parallel_sort(activeCellList.begin(), activeCellList.end(), vecCompare);

	float dx = myPhiGrid.dx();

	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward.

	auto solveEikonal = [&](const Vec2i& idx) -> float
	{
		float Ul = (idx[0] == 0) ? std::numeric_limits<float>::max() : myPhiGrid(idx[0] - 1, idx[1]);
		float Ur = (idx[0] == myPhiGrid.size()[0] - 1) ? std::numeric_limits<float>::max() : myPhiGrid(idx[0] + 1, idx[1]);

		float Ub = (idx[1] == 0) ? std::numeric_limits<float>::max() : myPhiGrid(idx[0], idx[1] - 1);
		float Ut = (idx[1] == myPhiGrid.size()[1] - 1) ? std::numeric_limits<float>::max() : myPhiGrid(idx[0], idx[1] + 1);

		float u = std::fabs(myPhiGrid(idx[0], idx[1]));

		int count = 0;

		float a = std::min(std::fabs(Ul), std::fabs(Ur));
		if (u - a <= 0.) a = std::numeric_limits<float>::max();
		else ++count;

		float b = std::min(std::fabs(Ub), std::fabs(Ut));

		if (u - b <= 0.) b = std::numeric_limits<float>::max();
		else ++count;

		if (a > b) std::swap(a, b);

		if (count == 1) u = a + dx;
		else if (count == 2)
		{
			float temp = -sqr(a) - sqr(b) + 2. * a * b + 2. * sqr(dx);
			if (temp < 0.) u = a + dx;
			else u = .5 * (a + b + sqrt(temp));
			assert(std::isfinite(u));
		}
		// There shouldn't be a case where count is 0 but it seems to be happenning..

		return u;

	};

	ScalarGrid<float> tempPhiGrid = myPhiGrid;

	float tolerance = dx * 1E-5;
	bool stillActiveCells = true;

	int activeCellCount = activeCellList.size();

	int iteration = 0;
	int maxIterations = 5 * myNarrowBand / dx;

	while (activeCellCount > 0 && iteration < maxIterations)
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, activeCellCount, tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			auto& localActiveCellList = parallelActiveCellList.local();

			Vec2i oldCell(-1);

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i newCell = activeCellList[cellIndex];

				if (oldCell == newCell)
					continue;

				oldCell = newCell;

				assert(reinitializedCells(newCell) == VisitedCellLabels::VISITED_CELL);

				float newPhi = solveEikonal(newCell);

				// If we hit the narrow band, we don't need to make any changes
				if (newPhi > myNarrowBand) continue;

				tempPhiGrid(newCell) = myPhiGrid(newCell) < 0 ? -newPhi : newPhi;

				// Check if new phi is converged
				float oldPhi = myPhiGrid(newCell);

				// If the cell is converged, load up the neighbours that aren't currently being VISITED
				if (std::fabs(newPhi - std::fabs(oldPhi)) < tolerance)
				{
					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(newCell, axis, direction);

							if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

							if (reinitializedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
							{
								float adjacentNewPhi = solveEikonal(adjacentCell);

								if (adjacentNewPhi > myNarrowBand) continue;

								// Check if new phi is less than the current value
								float adjacentOldPhi = std::fabs(myPhiGrid(adjacentCell));

								if ((adjacentNewPhi < adjacentOldPhi) && (std::fabs(adjacentNewPhi - adjacentOldPhi) > tolerance))
								{
									tempPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) < 0 ? -adjacentNewPhi : adjacentNewPhi;

									localActiveCellList.push_back(adjacentCell);
								}
							}

						}
				}
				// If the cell hasn't converged, toss it back into the list
				else
					localActiveCellList.push_back(newCell);
			}
		});

		// Turn off VISITED labels for current list
		tbb::parallel_for(tbb::blocked_range<int>(0, activeCellCount, tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = activeCellList[cellIndex];
				assert(reinitializedCells(cell) != VisitedCellLabels::FINISHED_CELL);
				reinitializedCells(cell) = VisitedCellLabels::UNVISITED_CELL;
			}
		});

		activeCellList.clear();

		mergeLocalThreadVectors(activeCellList, parallelActiveCellList);

		parallelActiveCellList.clear();

		tbb::parallel_sort(activeCellList.begin(), activeCellList.end(), vecCompare);

		activeCellCount = activeCellList.size();

		// Turn on VISITED labels for new list
		tbb::parallel_for(tbb::blocked_range<int>(0, activeCellCount, tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = activeCellList[cellIndex];
				assert(reinitializedCells(cell) != VisitedCellLabels::FINISHED_CELL);
				reinitializedCells(cell) = VisitedCellLabels::VISITED_CELL;
			}
		});

		//std::swap(tempPhiGrid, myPhiGrid);
		myPhiGrid = tempPhiGrid;

		++iteration;
	}
}

void LevelSet::initFromMesh(const EdgeMesh& initialMesh, bool doResizeGrid)
{
	if (doResizeGrid)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		Vec2f minBoundingBox(std::numeric_limits<float>::max());
		Vec2f maxBoundingBox(std::numeric_limits<float>::lowest());

		for (int vertexIndex = 0; vertexIndex < initialMesh.vertexCount(); ++vertexIndex)
			updateMinAndMax(minBoundingBox, maxBoundingBox, initialMesh.vertex(vertexIndex).point());

		// Just for nice whole numbers, let's clamp the bounding box to be an integer
		// offset in index space then bring it back to world space
		float maxNarrowBand = 10.;
		maxNarrowBand = std::min(myNarrowBand / dx(), maxNarrowBand);

		minBoundingBox = dx() * (Vec2f(floor(minBoundingBox / dx())) - 2 * Vec2f(maxNarrowBand));
		maxBoundingBox = dx() * (Vec2f(ceil(maxBoundingBox / dx())) + 2 * Vec2f(maxNarrowBand));

		clear();
		Transform xform(dx(), minBoundingBox);
		// Since we know how big the mesh is, we know how big our grid needs to be (wrt to grid spacing)
		myPhiGrid = ScalarGrid<float>(xform, Vec2i((maxBoundingBox - minBoundingBox) / dx()), myNarrowBand);
	}
	else
		myPhiGrid.resize(size(), myNarrowBand);

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED.
	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);
	UniformGrid<int> meshCellParities(size(), 0);

	for (const auto& edge : initialMesh.edges())
	{
		// It's easier to work in our index space and just scale the distance later.
		const Vec2f& startPoint = worldToIndex(initialMesh.vertex(edge.vertex(0)).point());
		const Vec2f& endPoint = worldToIndex(initialMesh.vertex(edge.vertex(1)).point());

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		Vec2f vmin, vmax;
		minAndMax(vmin, vmax, startPoint, endPoint);

		Vec2i edgeCeilMin = Vec2i(ceil(vmin));
		Vec2i edgeFloorMin = Vec2i(floor(vmin)) - Vec2i(1);
		Vec2i edgeFloorMax = Vec2i(floor(vmax));

		for (int j = edgeCeilMin[1]; j <= edgeFloorMax[1]; ++j)
			for (int i = edgeFloorMax[0]; i >= edgeFloorMin[0]; --i)
			{
				Vec2f gridNode(i, j);
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
	forEachVoxelRange(Vec2i(1), size() - Vec2i(1), [&](const Vec2i& cell)
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
	for (const auto& edge : initialMesh.edges())
	{
		// Using the vertices of the edge, we can update distance values for cells
		// within the bounding box of the mesh. It's easier to work in our index space
		// and just scale the distance later.
		const Vec2f& startPoint = worldToIndex(initialMesh.vertex(edge.vertex(0)).point());
		const Vec2f& endPoint = worldToIndex(initialMesh.vertex(edge.vertex(1)).point());

		// Build bounding box

		Vec2i minBoundingBox = Vec2i(floor(minUnion(startPoint, endPoint))) - Vec2i(2);
		minBoundingBox = maxUnion(minBoundingBox, Vec2i(0));

		Vec2i maxBoundingBox = Vec2i(ceil(maxUnion(startPoint, endPoint))) + Vec2i(2);
		Vec2i top = size() - Vec2i(1);
		maxBoundingBox = minUnion(maxBoundingBox, top);

		// Update distances to the mesh at grid cells within the bounding box
		assert(minBoundingBox[0] >= 0 && minBoundingBox[1] >= 0 && maxBoundingBox[0] < size()[0] && maxBoundingBox[1] < size()[1]);

		forEachVoxelRange(minBoundingBox, maxBoundingBox + Vec2i(1), [&](const Vec2i& cell)
		{
			if (reinitializedCells(cell) != VisitedCellLabels::UNVISITED_CELL)
			{
				Vec2f cellPoint(cell);
				Vec2f vec0 = cellPoint - startPoint;
				Vec2f vec1 = endPoint - startPoint;

				float s = dot(vec0, vec1) / dot(vec1, vec1); // Find projection along edge.
				s = clamp(s, float(0), float(1));

				// Remove on-edge projection to get vector from closest point on edge to cell point.
				float surfaceDistance = mag(vec0 - s * vec1) * dx();

				// Update if the distance to this edge is shorter than previous values.
				if (surfaceDistance < std::fabs(myPhiGrid(cell)))
				{
					// If the parity says the node is inside, set it to be negative
					myPhiGrid(cell) = (meshCellParities(cell) > 0) ? -surfaceDistance : surfaceDistance;
				}
			}
		});
	}

	reinitFastIterative(reinitializedCells);
	//reinitFastMarching(reinitializedCells);
}

void LevelSet::reinitFastMarching(UniformGrid<VisitedCellLabels>& reinitializedCells)
{
	assert(reinitializedCells.size() == size());

	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward
	auto solveEikonal = [&](const Vec2i& cell) -> float
	{
		float max = std::numeric_limits<float>::max();

		float U_bx = (cell[0] > 0) ? std::fabs(myPhiGrid(cell[0] - 1, cell[1])) : max;
		float U_fx = (cell[0] < size()[0] - 1) ? std::fabs(myPhiGrid(cell[0] + 1, cell[1])) : max;

		float U_by = (cell[1] > 0) ? std::fabs(myPhiGrid(cell[0], cell[1] - 1)) : max;
		float U_fy = (cell[1] < size()[1] - 1) ? std::fabs(myPhiGrid(cell[0], cell[1] + 1)) : max;

		float Uh = std::min(U_bx, U_fx);
		float Uv = std::min(U_by, U_fy);
		float U;
		
		if (std::fabs(Uh - Uv) >= dx())
			U = std::min(Uh, Uv) + dx();
		else
			// Quadratic equation from the Eikonal
			U = (Uh + Uv) / 2. + .5 * std::sqrt(pow(Uh + Uv, 2.) - 2. * (sqr(Uh) + sqr(Uv) - sqr(dx())));

		return U;
	};

	// Load up the BFS queue with the unvisited cells next to the finished ones
	using Node = std::pair<Vec2i, float>;
	auto cmp = [](const Node& a, const Node& b) -> bool { return std::fabs(a.second) > std::fabs(b.second); };
	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> marchingQ(cmp);

	forEachVoxelRange(Vec2i(0), size(), [&](const Vec2i& cell)
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
						float dist = solveEikonal(adjacentCell);
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
						float dist = solveEikonal(adjacentCell);
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
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	// Run marching squares loop
	forEachVoxelRange(Vec2i(0), size() - Vec2i(1), [&](const Vec2i& cell)
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

			Vec2f startPoint = interpolateInterface(startNode, endNode);

			// Find second vertex
			edge = marchingSquaresTemplate[mcKey][edgeIndex + 1];
			faceMap = cellToFaceCCW(cell, edge);

			face = Vec2i(faceMap[0], faceMap[1]);
			axis = faceMap[2];

			startNode = faceToNode(face, axis, 0);
			endNode = faceToNode(face, axis, 1);

			Vec2f endPoint = interpolateInterface(startNode, endNode);

			// Store vertices
			Vec2f worldStartPoint = indexToWorld(startPoint);
			Vec2f worldEndPoint = indexToWorld(endPoint);

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
	std::vector<Vec2f> verts;
	std::vector<Vec2i> edges;

	// Create grid to store index to dual contouring point. Note that phi is
	// center sampled so the DC grid must be node sampled and one cell shorter
	// in each dimension
	UniformGrid<int> dcPointIndex(size() - Vec2i(1), -1);

	// Run dual contouring loop
	forEachVoxelRange(Vec2i(0), dcPointIndex.size(), [&](const Vec2i& cell)
	{
		std::vector<Vec2f> qefPoints;
		std::vector<Vec2f> qefNormals;

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
					Vec2f interfacePoint = interpolateInterface(backwardNode, forwardNode);
					qefPoints.push_back(interfacePoint);

					// Find associated surface normal
					Vec2f surfaceNormal = normal(indexToWorld(interfacePoint));
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

				b(pointIndex) = dot(qefNormals[pointIndex], qefPoints[pointIndex]);

				pointCOM[0] += qefPoints[pointIndex][0];
				pointCOM[1] += qefPoints[pointIndex][1];
			}

			pointCOM /= float(qefPoints.size());

			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			svd.setThreshold(1E-2);

			Eigen::VectorXd dcPoint = pointCOM + svd.solve(b - A * pointCOM);

			Vec2f vecCOM(pointCOM[0], pointCOM[1]);

			Vec2f boundingBoxMin = floor(vecCOM);
			Vec2f boundingBoxMax = ceil(vecCOM);

			if (dcPoint[0] < boundingBoxMin[0] ||
				dcPoint[1] < boundingBoxMin[1] ||
				dcPoint[0] > boundingBoxMax[0] ||
				dcPoint[1] > boundingBoxMax[1])
				dcPoint = pointCOM;

			verts.push_back(indexToWorld(Vec2f(dcPoint[0], dcPoint[1])));
			dcPointIndex(cell) = verts.size() - 1;
		}
	});

	for (int axis : {0, 1})
	{
		Vec2i start(0); ++start[axis];
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

Vec2f LevelSet::interpolateInterface(const Vec2i& startPoint, const Vec2i& endPoint) const
{
	assert((myPhiGrid(startPoint[0], startPoint[1]) <= 0 && myPhiGrid(endPoint[0], endPoint[1]) > 0) ||
			(myPhiGrid(startPoint[0], startPoint[1]) > 0 && myPhiGrid(endPoint[0], endPoint[1]) <= 0));

	//Find weight to zero isosurface
	float s = myPhiGrid(startPoint) / (myPhiGrid(startPoint) - myPhiGrid(endPoint));
	s = clamp(s, float(0), float(1));

	Vec2f dx = Vec2f(endPoint) - Vec2f(startPoint);
	return Vec2f(startPoint) + s * dx;
}

void LevelSet::unionSurface(const LevelSet& unionPhi)
{
	assert(isGridMatched(unionPhi));

	tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = myPhiGrid.unflatten(cellIndex);
			if (unionPhi(cell) < 2 * unionPhi.dx())
				myPhiGrid(cell) = std::min(myPhiGrid(cell), unionPhi(cell));
		}
	});

	reinitMesh();
}

}
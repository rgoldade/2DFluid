#include "LevelSet2D.h"

#include <limits>
#include <utility>

#include "tbb/tbb.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "VectorGrid.h"

void LevelSet2D::drawGrid(Renderer& renderer) const
{
	myPhiGrid.drawGrid(renderer);
}

void LevelSet2D::drawMeshGrid(Renderer& renderer) const
{
	Transform xform(myPhiGrid.dx(), myPhiGrid.offset() + Vec2R(0.5) * myPhiGrid.dx());
	ScalarGrid<int> tempGrid(xform, myPhiGrid.size());
	tempGrid.drawGrid(renderer);
}

void LevelSet2D::drawSupersampledValues(Renderer& renderer, Real radius, unsigned samples, unsigned size) const
{
	myPhiGrid.drawSupersampledValues(renderer, radius, samples, size);
}
void LevelSet2D::drawNormals(Renderer& renderer, const Vec3f& colour, Real length) const
{
	myPhiGrid.drawSampleGradients(renderer, colour, length);
}

void LevelSet2D::drawSurface(Renderer& renderer, const Vec3f& colour)
{
	Mesh2D surface = buildMSMesh();
	surface.drawMesh(renderer, colour);
}

void LevelSet2D::drawDCSurface(Renderer& renderer, const Vec3f& colour)
{
	Mesh2D surface = buildDCMesh();
	surface.drawMesh(renderer, colour, true, true, Vec3f(1.,0,0));
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.

Vec2R LevelSet2D::findSurface(const Vec2R& worldPoint, unsigned iterationLimit) const
{
	Real phi = myPhiGrid.interp(worldPoint);
	unsigned iterationCount = 0;
	Real epsilon = 1E-2 * dx();
	Vec2R tempPoint = worldPoint;

	if (fabs(phi) < myNarrowBand)
	{
		while (fabs(phi) > epsilon && iterationCount < iterationLimit)
		{
			tempPoint -= phi * normal(tempPoint);
			phi = myPhiGrid.cubicInterp(tempPoint);
			++iterationCount;
		}
	}

	return tempPoint;
}

Vec2R LevelSet2D::findSurfaceIndex(const Vec2R& indexPoint, unsigned iterationLimit) const
{
	Vec2R worldPoint = indexToWorld(indexPoint);
	worldPoint = findSurface(worldPoint, iterationLimit);
	return worldToIndex(worldPoint);
}

auto vecCompare = [](const Vec2ui &a, const Vec2ui &b) -> bool
{
	if (a[0] < b[0]) return true;
	else if (a[0] == b[0] && a[1] < b[1]) return true;
	return false;
};

void LevelSet2D::reinitFIM()
{
	UniformGrid<MarkedCells> reinitializedCells(size(), MarkedCells::UNVISITED);
	
	// Find the zero crossings, update their distances and flag as source cells
	ScalarGrid<Real> tempPhiGrid = myPhiGrid;

	unsigned totalVoxels = size()[0] * size()[1];

	tbb::parallel_for(tbb::blocked_range<unsigned>(0, totalVoxels), [&](const tbb::blocked_range<unsigned> &range)
	{
		for (unsigned r = range.begin(); r != range.end(); ++r)
		{
			Vec2ui cell = reinitializedCells.unflatten(r);

			// Check for a zero crossing
			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

					if (myPhiGrid(cell) * myPhiGrid(Vec2ui(adjacentCell)) <= 0.)
					{
						// Update distance
						Vec2R worldPoint = indexToWorld(Vec2R(cell));
						Vec2R interfacePoint = findSurface(worldPoint, 5);
					
						Real udf = dist(interfacePoint, worldPoint);

						// If the cell has not be updated yet OR the update is lower than a previous
						// update, assign the SDF to the cell
						if (reinitializedCells(cell) != MarkedCells::FINISHED)
						{
							reinitializedCells(cell) = MarkedCells::FINISHED;
							tempPhiGrid(cell) = udf;
						}
						else if (tempPhiGrid(cell) > udf)
							tempPhiGrid(cell) = udf;
					}
				}

			Real val = myNarrowBand;
			if (reinitializedCells(cell) == MarkedCells::FINISHED)
				val = tempPhiGrid(cell);

			tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -val : val;
		}
	});

	std::swap(myPhiGrid, tempPhiGrid);

	reinitFastIterative(reinitializedCells);
}

void LevelSet2D::reinitFastIterative(UniformGrid<MarkedCells> &reinitializedCells)
{
	assert(reinitializedCells.size() == size());

	//
	// Before starting the iterations, we want to construct the active list of voxels
	// to reinitialize.
	//

	unsigned totalVoxels = size()[0] * size()[1];

	tbb::enumerable_thread_specific<std::vector<Vec2ui>> parallelActiveCellList;

	tbb::parallel_for(tbb::blocked_range<unsigned>(0, totalVoxels), [&](const tbb::blocked_range<unsigned> &range)
	{
		std::vector<Vec2ui> &localActiveCellList = parallelActiveCellList.local();

		for (unsigned r = range.begin(); r != range.end(); ++r)
		{
			Vec2ui cell = reinitializedCells.unflatten(r);

			if (reinitializedCells(cell) == MarkedCells::FINISHED)
			{
				// Add neighbours to the list
				for (unsigned axis : {0, 1})
					for (unsigned direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

						if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

						if (reinitializedCells(Vec2ui(adjacentCell)) == MarkedCells::UNVISITED)
						{
							localActiveCellList.push_back(Vec2ui(adjacentCell));
							reinitializedCells(Vec2ui(adjacentCell)) = MarkedCells::VISITED;
						}
					}
			}
		}
	});

	std::vector<Vec2ui> activeCellList;

	for (const auto& localList : parallelActiveCellList)
	{
		activeCellList.insert(activeCellList.end(), localList.begin(), localList.end());
	}

	parallelActiveCellList.clear();

	tbb::parallel_sort(activeCellList.begin(), activeCellList.end(), vecCompare);

	Real dx = myPhiGrid.dx();

	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward.

	auto solveEikonal = [&](const Vec2i& idx) -> Real
	{
		Real Ul = (idx[0] == 0) ? std::numeric_limits<Real>::max() : myPhiGrid(idx[0] - 1, idx[1]);
		Real Ur = (idx[0] == myPhiGrid.size()[0] - 1) ? std::numeric_limits<Real>::max() : myPhiGrid(idx[0] + 1, idx[1]);

		Real Ub = (idx[1] == 0) ? std::numeric_limits<Real>::max() : myPhiGrid(idx[0], idx[1] - 1);
		Real Ut = (idx[1] == myPhiGrid.size()[1] - 1) ? std::numeric_limits<Real>::max() :myPhiGrid(idx[0], idx[1] + 1);

		Real u = fabs(myPhiGrid(idx[0], idx[1]));

		int count = 0;

		Real a = std::min(fabs(Ul), fabs(Ur));
		if (u - a <= 0.) a = std::numeric_limits<Real>::max();
		else ++count;

		Real b = std::min(fabs(Ub), fabs(Ut));

		if (u - b <= 0.) b = std::numeric_limits<Real>::max();
		else ++count;

		if (a > b) std::swap(a, b);

		if (count == 1) u = a + dx;
		else if (count == 2)
		{
			Real temp = -Util::sqr(a) - Util::sqr(b) + 2. * a * b + 2. * Util::sqr(dx);
			if (temp < 0.) u = a + dx;
			else u = .5 * (a + b + sqrt(temp));
			assert(std::isfinite(u));
		}
		// There shouldn't be a case where count is 0 but it seems to be happenning..

		return u;

	};

	ScalarGrid<Real> tempPhiGrid = myPhiGrid;

	Real tolerance = dx * 1E-5;
	bool stillActiveCells = true;

	unsigned activeCellCount = activeCellList.size();

	unsigned iter = 0;
	unsigned maxIters = 5 * myNarrowBand / dx;

	while (activeCellCount > 0 && iter < maxIters)
	{
		tbb::parallel_for(tbb::blocked_range<unsigned>(0, activeCellCount), [&](const tbb::blocked_range<unsigned> &range)
		{
			std::vector<Vec2ui> &localActiveCellList = parallelActiveCellList.local();
			
			Vec2ui oldCell(myPhiGrid.size());

			for (unsigned r = range.begin(); r != range.end(); ++r)
			{
				Vec2ui newCell = activeCellList[r];

				if (oldCell != newCell)
				{
					assert(reinitializedCells(newCell) == MarkedCells::VISITED);

					Real newPhi = solveEikonal(Vec2i(newCell));

					// If we hit the narrow band, we don't need to make any changes
					if (newPhi > myNarrowBand) continue;

					tempPhiGrid(newCell) = myPhiGrid(newCell) < 0 ? -newPhi : newPhi;

					// Check if new phi is converged
					Real oldPhi = myPhiGrid(newCell);

					// If the cell is converged, load up the neighbours that aren't currently being VISITED
					if (fabs(newPhi - fabs(oldPhi)) < tolerance)
					{
						for (unsigned axis : {0, 1})
							for (unsigned direction : {0, 1})
							{
								Vec2i adjacentCell = cellToCell(Vec2i(newCell), axis, direction);

								if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

								if (reinitializedCells(Vec2ui(adjacentCell)) == MarkedCells::UNVISITED)
								{
									Real adjacentNewPhi = solveEikonal(adjacentCell);

									if (adjacentNewPhi > myNarrowBand) continue;

									// Check if new phi is less than the current value
									Real adjacentOldPhi = fabs(myPhiGrid(Vec2ui(adjacentCell)));

									if ((adjacentNewPhi < adjacentOldPhi) && (fabs(adjacentNewPhi - adjacentOldPhi) > tolerance))
									{
										tempPhiGrid(Vec2ui(adjacentCell)) = myPhiGrid(Vec2ui(adjacentCell)) < 0 ? -adjacentNewPhi : adjacentNewPhi;

										localActiveCellList.push_back(Vec2ui(adjacentCell));
									}
								}

							}
					}
					// If the cell hasn't converged, toss it back into the list
					else
						localActiveCellList.push_back(newCell);
				}

				oldCell = newCell;
			}
		});

		// Turn off VISITED labels for current list
		tbb::parallel_for(tbb::blocked_range<unsigned>(0, activeCellCount), [&](const tbb::blocked_range<unsigned> &range)
		{
			for (unsigned r = range.begin(); r != range.end(); ++r)
			{
				Vec2ui cell = activeCellList[r];
				assert(reinitializedCells(cell) != MarkedCells::FINISHED);
				reinitializedCells(cell) = MarkedCells::UNVISITED;
			}
		});

		activeCellList.clear();

		for (auto i = parallelActiveCellList.begin(); i != parallelActiveCellList.end(); ++i)
		{
			activeCellList.insert(activeCellList.end(), i->begin(), i->end());
		}

		parallelActiveCellList.clear();

		tbb::parallel_sort(activeCellList.begin(), activeCellList.end(), vecCompare);
		
		activeCellCount = activeCellList.size();

		// Turn on VISITED labels for new list
		tbb::parallel_for(tbb::blocked_range<unsigned>(0, activeCellCount), [&](const tbb::blocked_range<unsigned> &range)
		{
			for (unsigned r = range.begin(); r != range.end(); ++r)
			{
				Vec2ui idx = activeCellList[r];
				assert(reinitializedCells(idx) != MarkedCells::FINISHED);
				reinitializedCells(idx) = MarkedCells::VISITED;
			}
		}); 
		
		std::swap(tempPhiGrid, myPhiGrid);

		++iter;
	}
}

void LevelSet2D::reinit()
{	
	ScalarGrid<Real> tempPhiGrid = myPhiGrid;

	UniformGrid<MarkedCells> reinitializedCells(size(), MarkedCells::UNVISITED);

	// Find zero crossing
	forEachVoxelRange(Vec2ui(0), myPhiGrid.size(), [&](const Vec2ui& cell)
	{
		for (unsigned axis : {0, 1})
			for (unsigned direction : {0, 1})
			{
				Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

				if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

				if (myPhiGrid(cell) * myPhiGrid(Vec2ui(adjacentCell)) <= 0.)
				{
					Vec2R worldPoint = indexToWorld(Vec2R(cell));
					Vec2R interfacePoint = findSurface(worldPoint, 5);

					Real udf = dist(worldPoint, interfacePoint);

					// If the cell has not be updated yet OR the update is lower than a previous
					// update, assign the SDF to the cell
					if (reinitializedCells(cell) != MarkedCells::FINISHED || fabs(tempPhiGrid(cell)) > udf)
					{
						tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -udf : udf;
						reinitializedCells(cell) = MarkedCells::FINISHED;
					}
				}
			}
		// Set remaining grids with background value.
		if (reinitializedCells(cell) != MarkedCells::FINISHED)
		{
			//Real max = std::numeric_limits<Real>::max();
			tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -myNarrowBand : myNarrowBand;
		}
	});

	myPhiGrid = tempPhiGrid;

	reinitFastMarching(reinitializedCells);
}

void LevelSet2D::init(const Mesh2D& initMesh, bool resize)
{
	if (resize)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		Vec2R minBoundingBox(std::numeric_limits<Real>::max());
		Vec2R maxBoundingBox(std::numeric_limits<Real>::lowest());

		for (unsigned vertexIndex = 0; vertexIndex < initMesh.vertexListSize(); ++vertexIndex)
			updateMinAndMax(minBoundingBox, maxBoundingBox, initMesh.vertex(vertexIndex).point());

		// Just for nice whole numbers, let's clamp the bounding box to be an integer
		// offset in index space then bring it back to world space
		Real maxNarrowBand = 10.;
		maxNarrowBand = std::min(myNarrowBand/dx(), maxNarrowBand);

		minBoundingBox = (Vec2R(floor(minBoundingBox / dx())) - Vec2R(maxNarrowBand)) * dx();
		maxBoundingBox = (Vec2R(ceil(maxBoundingBox / dx())) + Vec2R(maxNarrowBand)) * dx();

		clear();
		Transform xform(dx(), minBoundingBox);
		// Since we know how big the mesh is, we know how big our grid needs to be (wrt to grid spacing)
		myPhiGrid = ScalarGrid<Real>(xform, Vec2ui((maxBoundingBox - minBoundingBox) / dx()), myNarrowBand);
	}
	else myPhiGrid.resize(size(), myNarrowBand);

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED. The following pass will do a 
	// BFS to assign the remaining UNVISITED cells with the appropriate distances.
	UniformGrid<MarkedCells> reinitializedCells(size(), MarkedCells::UNVISITED);
	UniformGrid<int> meshParityCells(size(), 0);

	for (auto edge : initMesh.edges())
	{
		// It's easier to work in our index space and just scale the distance later.
		const Vec2R startPoint = worldToIndex(initMesh.vertex(edge.vertex(0)).point());
		const Vec2R endPoint = worldToIndex(initMesh.vertex(edge.vertex(1)).point());

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		Vec2R vmin, vmax;
		minAndMax(vmin, vmax, startPoint, endPoint);

		Vec2R edgeCeilMin = ceil(vmin);
		Vec2R edgeFloorMin = floor(vmin) - Vec2R(1);
		Vec2R edgeFloorMax = floor(vmax);

		for (int j = edgeCeilMin[1]; j <= edgeFloorMax[1]; ++j)
			for (int i = edgeFloorMax[0]; i >= edgeFloorMin[0]; --i)
			{
				Vec2R gridNode(i, j);
				Intersection result = exactEdgeIntersect(startPoint, endPoint, gridNode);

				// TODO: remove once test complete
				if (gridNode[0] < 0 || gridNode[1] < 0 || gridNode[0] >= myPhiGrid.size()[0] ||
					gridNode[1] >= myPhiGrid.size()[1])
				{
					std::cout << "Caught out of bounds. Node: " << gridNode[0] << " " << gridNode[1] << std::endl;
				}

				if (result == Intersection::NO) continue;
				else if (result == Intersection::YES)
				{
					if (startPoint[1] < endPoint[1])
					{
						// Decrement the parity and increment since the grid_node is
						// "left" of the mesh-edge crossing the grid-edge
						++meshParityCells(i + 1, j);
					}
					else
					{
						--meshParityCells(i + 1, j);
					}
				}
				// If the grid node is explicitly on the mesh-edge, set distance to zero
				// since it might not be exactly zero due to floating point error above.
				else if (result == Intersection::ON)
				{
					// Technically speaking, the zero isocountour means we're inside
					// the surface. So we should change the parity at the node that
					// is intersected even though it's zero and therefore the sign
					// is meaningless.
					if (startPoint[1] < endPoint[1]) // grid_node is to the "left" of the edge
					{
						// Decrement the parity and increment since the grid_node is
						// "left" of the mesh-edge crossing the grid-edge
						++meshParityCells(i, j);
					}
					else
					{
						--meshParityCells(i, j);
					}

					reinitializedCells(i, j) = MarkedCells::FINISHED;
					myPhiGrid(i, j) = 0.;
				}

				break;
			}
	}

	// Now that all the x-axis edge crossings have been found, we can compile the parity changes
	// and label grid nodes that are at the interface
	for (unsigned j = 0; j < size()[1]; ++j)
	{
		int parity = myIsInverted ? 1 : 0;

		// We loop x-major because that's how we've set up our edge intersection.
		// This is definitely cache incoherent based on the UniformGrid structure..
		// It's probably not a deal breaker in 2-D but it could suck in 3-D
		for (unsigned i = 0; i < size()[0]; ++i) // TODO: double check that I've resized right
		{
			// Update parity before changing sign since the parity values above used the convention
			// of putting the change on the "far" node (i.e. after the mesh-grid intersection).

			Vec2ui cell(i, j);
			parity += meshParityCells(cell);
			meshParityCells(cell) = parity;
			if (parity > 0) myPhiGrid(cell) = -myPhiGrid(cell);
		}
	}

	// With the parity assigned, loop over the grid once more and label nodes that have a sign change
	// with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
	for (unsigned i = 1; i < (size()[0] - 1); ++i)
		for (unsigned j = 1; j < (size()[1] - 1); ++j)
		{
			const Vec2ui cell(i, j);

			bool isInside = meshParityCells(cell) > 0;

			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

					bool isAdjacentInside = meshParityCells(Vec2ui(adjacentCell)) > 0;

					// Flag to indicate that this cell needs an exact distance based on the mesh
					if (isInside != isAdjacentInside) reinitializedCells(cell) = MarkedCells::FINISHED;
				}
		}

	// Loop over all the edges in the mesh. Level set grid cells labelled as VISITED will be
	// updated with the distance to the surface if it happens to be shorter than the current
	// distance to the surface.
	for (auto edge : initMesh.edges())
	{
		// Using the vertices of the edge, we can update distance values for cells
		// within the bounding box of the mesh. It's easier to work in our index space
		// and just scale the distance later.
		const Vec2R startPoint = worldToIndex(initMesh.vertex(edge.vertex(0)).point());
		const Vec2R endPoint = worldToIndex(initMesh.vertex(edge.vertex(1)).point());

		// Build bounding box

		Vec2R minBoundingBox = floor(minUnion(startPoint, endPoint)) - Vec2R(2);
		minBoundingBox = maxUnion(minBoundingBox, Vec2R(0));
		Vec2R maxBoundingBox = ceil(maxUnion(startPoint, endPoint)) + Vec2R(2);
		Vec2R top(size()[0] - 1, size()[1] - 1);
		maxBoundingBox = minUnion(maxBoundingBox, top);

		// Update distances to the mesh at grid cells within the bounding box
		assert(minBoundingBox[0] >= 0 && minBoundingBox[1] >= 0 && maxBoundingBox[0] < size()[0] && maxBoundingBox[1] < size()[1]);

		for (unsigned i = minBoundingBox[0]; i <= maxBoundingBox[0]; ++i)
			for (unsigned j = minBoundingBox[1]; j <= maxBoundingBox[1]; ++j)
			{
				Vec2ui cell(i, j);
				if (reinitializedCells(cell) != MarkedCells::UNVISITED)
				{
					Vec2R cellPoint(cell);
					Vec2R vec0 = cellPoint - startPoint;
					Vec2R vec1 = endPoint - startPoint;

					Real s = (dot(vec0, vec1) / dot(vec1, vec1)); // Find projection along edge.
					s = ((s < 0.) ? 0. : ((s > 1.) ? 1. : s)); // Truncate scale to be within v0-v1 segment
					
					// Remove on-edge projection to get vector from closest point on edge to cell point.
					Real dist = mag(vec0 - s * vec1) * dx(); 

					// Update if the distance to this edge is shorter than previous values.
					if (fabs(myPhiGrid(cell)) > dist)
					{
						// If the parity says the node is inside, set it to be negative
						myPhiGrid(cell) = (meshParityCells(cell) > 0) ? -dist : dist;
					}
				}
			}
	}

	reinitFastMarching(reinitializedCells);
}

void LevelSet2D::reinitFastMarching(UniformGrid<MarkedCells>& reinitializedCells)
{
	assert(reinitializedCells.size() == size());

	// Now that the correct distances and signs have been recorded at the interface,
	// it's important to flood fill that the signed distances outwards into the entire grid.
	// We use the Eikonal equation here to build this outward
	auto solveEikonal = [&](const Vec2i& idx) -> Real
	{
		Real max = std::numeric_limits<Real>::max();
		Real U_bx = (idx[0] > 0) ? fabs(myPhiGrid(idx[0] - 1, idx[1])) : max;
		Real U_fx = (idx[0] < size()[0] - 1) ? fabs(myPhiGrid(idx[0] + 1, idx[1])) : max;

		Real U_by = (idx[1] > 0) ? fabs(myPhiGrid(idx[0], idx[1] - 1)) : max;
		Real U_fy = (idx[1] < size()[1] - 1) ? fabs(myPhiGrid(idx[0], idx[1] + 1)) : max;

		Real Uh = Util::min(U_bx, U_fx);
		Real Uv = Util::min(U_by, U_fy);
		Real U;
		if (fabs(Uh - Uv) >= dx())
			U = Util::min(Uh, Uv) + dx();
		else
			// Quadratic equation from the Eikonal
			U = (Uh + Uv) / 2. + .5 * sqrt(pow(Uh + Uv, 2.) - 2. * (Util::sqr(Uh) + Util::sqr(Uv) - Util::sqr(dx())));

		return U;
	};

	// Load up the BFS queue with the unvisited cells next to the finished ones
	using Node = std::pair<Vec2ui, Real>;
	auto cmp = [](const Node& a, const Node& b) -> bool { return fabs(a.second) > fabs(b.second); };
	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> phiCellQ(cmp);

	forEachVoxelRange(Vec2ui(0), size(), [&](const Vec2ui& cell)
	{
		if (reinitializedCells(cell) == MarkedCells::FINISHED)
		{
			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(Vec2i(cell), axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

					if (reinitializedCells(Vec2ui(adjacentCell)) == MarkedCells::UNVISITED)
					{
						Real udf = solveEikonal(adjacentCell);
						myPhiGrid(Vec2ui(adjacentCell)) = (myPhiGrid(Vec2ui(adjacentCell)) <= 0.) ? -udf : udf;

						assert(udf >= 0);
						Node node(Vec2ui(adjacentCell), udf);

						phiCellQ.push(node);
						reinitializedCells(Vec2ui(adjacentCell)) = MarkedCells::VISITED;
					}
				}
		}
	});

	while (!phiCellQ.empty())
	{
		Node localNode = phiCellQ.top();
		Vec2ui localCell = localNode.first;
		phiCellQ.pop();

		// Since you can't just update parts of the priority queue,
		// it's possible that a cell has been solidified at a smaller distance
		// and an older insert if floating around.
		if (reinitializedCells(localCell) == MarkedCells::FINISHED)
		{
			// Make sure that the distance assigned to the cell is smaller than
			// what is floating around
			assert(fabs(myPhiGrid(localCell)) <= fabs(localNode.second));
			continue;
		}
		assert(reinitializedCells(localCell) == MarkedCells::VISITED);

		if (fabs(myPhiGrid(localCell)) < myNarrowBand)
		{
			// Debug check that there is indeed a FINISHED cell next to it.
			bool hasFinishedNeighbour = false;

			// Loop over the neighbouring cells and load the unvisited cells
			// and update the visited cells
			for (unsigned axis : {0, 1})
				for (unsigned direction : {0, 1})
				{
					Vec2i adjacentCell = cellToCell(Vec2i(localCell), axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

					if (reinitializedCells(Vec2ui(adjacentCell)) == MarkedCells::FINISHED)
						hasFinishedNeighbour = true;
					else // If visited, then we'll update it
					{
						Real udf = solveEikonal(adjacentCell);
						assert(udf >= 0);
						if (udf > myNarrowBand) udf = myNarrowBand;

						// If the computed distance is greater than the existing distance, we can skip it
						if (reinitializedCells(Vec2ui(adjacentCell)) == MarkedCells::VISITED && udf > fabs(myPhiGrid(Vec2ui(adjacentCell))))
							continue;

						myPhiGrid(Vec2ui(adjacentCell)) = myPhiGrid(Vec2ui(adjacentCell)) < 0. ? -udf : udf;

						Node node(Vec2ui(adjacentCell), udf);

						phiCellQ.push(node);
						reinitializedCells(Vec2ui(adjacentCell)) = MarkedCells::VISITED;
					}
				}

			//Check that a marked cell was indeed visited
			assert(hasFinishedNeighbour);
		}
		// Clamp to narrow band
		else
		{
			myPhiGrid(localCell) = (myPhiGrid(localCell) < 0.) ? -myNarrowBand : myNarrowBand;
		}

		// Solidify cell now that we've handled all it's neighbours
		reinitializedCells(localCell) = MarkedCells::FINISHED;
	}
}

// Extract a mesh representation of the interface. Useful for rendering but not much
// else since there will be duplicate vertices per grid edge in the current implementation.

Mesh2D LevelSet2D::buildMSMesh() const
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;
		
	// Run marching squares loop
	forEachVoxelRange(Vec2ui(0), size() - Vec2ui(1), [&](const Vec2ui& cell)
	{
		unsigned mcKey = 0;

		for (unsigned direction = 0; direction < 4; ++direction)
		{
			Vec2ui node = cellToNode(cell, direction);

			if (myPhiGrid(node) <= 0.) mcKey += (1 << direction);
		}

		// Connect edges using the marching squares template
		for (unsigned edgeIndex = 0; edgeIndex < 4 && marchingSquaresTemplate[mcKey][edgeIndex] >= 0; edgeIndex += 2)
		{
			// Find first vertex
			unsigned edge = marchingSquaresTemplate[mcKey][edgeIndex];

			Vec3ui faceMap = cellToFace(cell, edge);
			
			Vec2ui face(faceMap[0], faceMap[1]);
			unsigned axis = faceMap[2];
			Vec2ui startNode = faceToNode(face, axis, 0);
			Vec2ui endNode = faceToNode(face, axis, 1);

			Vec2R startPoint = interpolateInterface(startNode, endNode);

			// Find second vertex
			edge = marchingSquaresTemplate[mcKey][edgeIndex + 1];
			faceMap = cellToFace(cell, edge);

			face =  Vec2ui(faceMap[0], faceMap[1]);
			axis = faceMap[2];

			startNode = faceToNode(face, axis, 0);
			endNode = faceToNode(face, axis, 1);

			Vec2R endPoint = interpolateInterface(startNode, endNode);

			// Store vertices
			Vec2R worldStartPoint = indexToWorld(startPoint);
			Vec2R worldEndPoint = indexToWorld(endPoint);

			verts.push_back(worldStartPoint);
			verts.push_back(worldEndPoint);

			edges.push_back(Vec2ui(verts.size() - 2, verts.size() - 1));
		}
	});

	return Mesh2D(edges, verts);
}

// Extract a mesh representation of the interface using dual contouring
Mesh2D LevelSet2D::buildDCMesh() const
{
	std::vector<Vec2R> verts;
	std::vector<Vec2ui> edges;
	
	// Create grid to store index to dual contouring point. Note that phi is
	// center sampled so the DC grid must be node sampled and one cell shorter
	// in each dimension
	UniformGrid<unsigned> dcPointIndex(size() - Vec2ui(1), -1);

	// Run dual contouring loop
	forEachVoxelRange(Vec2ui(0), dcPointIndex.size(), [&](const Vec2ui& cell)
	{
		std::vector<Vec2R> qefPoints;
		std::vector<Vec2R> qefNormals;

		for (unsigned axis : {0, 1})
			for (unsigned direction : {0, 1})
			{
				Vec2ui face = cellToFace(cell, axis, direction);

				Vec2ui backwardNode = faceToNode(face, axis, 0);
				Vec2ui forwardNode = faceToNode(face, axis, 1);

				if ((myPhiGrid(backwardNode) <= 0 && myPhiGrid(forwardNode) > 0) ||
					(myPhiGrid(backwardNode) > 0 && myPhiGrid(forwardNode) <= 0))
				{
					// Find interface point
					Vec2R interfacePoint = interpolateInterface(backwardNode, forwardNode);
					qefPoints.push_back(interfacePoint);

					// Find associated surface normal
					Vec2R surfaceNormal = normal(indexToWorld(interfacePoint));
					qefNormals.push_back(surfaceNormal);
				}
			}

		if (qefPoints.size() > 0)
		{
			Eigen::MatrixXd A(qefPoints.size(), 2);
			Eigen::VectorXd b(qefPoints.size());
			Eigen::VectorXd pointCOM = Eigen::VectorXd::Zero(2);

			assert(qefPoints.size() > 1);

			for (unsigned pointIndex = 0; pointIndex < qefPoints.size(); ++pointIndex)
			{
				A(pointIndex, 0) = qefNormals[pointIndex][0];
				A(pointIndex, 1) = qefNormals[pointIndex][1];

				b(pointIndex) = dot(qefNormals[pointIndex], qefPoints[pointIndex]);
				
				pointCOM[0] += qefPoints[pointIndex][0];
				pointCOM[1] += qefPoints[pointIndex][1];
			}

			pointCOM /= Real(qefPoints.size());

			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			svd.setThreshold(1E-2);

			Eigen::VectorXd dcPoint = pointCOM + svd.solve(b - A * pointCOM);

			Vec2R vecCOM(pointCOM[0], pointCOM[1]);

			Vec2R boundingBoxMin = floor(vecCOM);
			Vec2R boundingBoxMax = ceil(vecCOM);

			if (dcPoint[0] < boundingBoxMin[0] ||
				dcPoint[1] < boundingBoxMin[1] ||
				dcPoint[0] > boundingBoxMax[0] ||
				dcPoint[1] > boundingBoxMax[1])
					dcPoint = pointCOM;

			verts.push_back(indexToWorld(Vec2R(dcPoint[0], dcPoint[1])));
			dcPointIndex(cell) = verts.size() - 1;
		}
	});
	
	for (unsigned axis : {0, 1})
	{
		Vec2ui start(0); ++start[axis];
		Vec2ui end(dcPointIndex.size());

		forEachVoxelRange(start, end, [&](const Vec2ui& face)
		{
			Vec2ui backwardNode = faceToNode(face, axis, 0);
			Vec2ui forwardNode = faceToNode(face, axis, 1);

			if ((myPhiGrid(backwardNode) <= 0 && myPhiGrid(forwardNode) > 0) ||
				(myPhiGrid(backwardNode) > 0 && myPhiGrid(forwardNode) <= 0))
			{
				Vec2i backwardCell = faceToCell(Vec2i(face), axis, 0);
				Vec2i forwardCell = faceToCell(Vec2i(face), axis, 1);

				assert(dcPointIndex(Vec2ui(backwardCell)) >= 0 && dcPointIndex(Vec2ui(forwardCell)) >= 0);

				Vec2ui edge;
				if (myPhiGrid(backwardNode) <= 0.)
				{
					if (axis == 0)
						edge = Vec2ui(dcPointIndex(Vec2ui(backwardCell)), dcPointIndex(Vec2ui(forwardCell)));
					else
						edge = Vec2ui(dcPointIndex(Vec2ui(forwardCell)), dcPointIndex(Vec2ui(backwardCell)));
				}
				else
				{
					if (axis == 0)
						edge = Vec2ui(dcPointIndex(Vec2ui(forwardCell)), dcPointIndex(Vec2ui(backwardCell)));
					else
						edge = Vec2ui(dcPointIndex(Vec2ui(backwardCell)), dcPointIndex(Vec2ui(forwardCell)));
				}

				edges.push_back(edge);
			}
		});
	}

	return Mesh2D(edges, verts);
}

Vec2R LevelSet2D::interpolateInterface(const Vec2ui& startPoint, const Vec2ui& endPoint) const
{
	assert(myPhiGrid(startPoint[0], startPoint[1]) <= 0 && myPhiGrid(endPoint[0], endPoint[1]) > 0 ||
			myPhiGrid(startPoint[0], startPoint[1]) > 0 && myPhiGrid(endPoint[0], endPoint[1]) <= 0);

	//Find weight to zero isosurface
	Real s = myPhiGrid(startPoint[0], startPoint[1]) / (myPhiGrid(startPoint[0], startPoint[1]) - myPhiGrid(endPoint[0], endPoint[1]));

	if (s < 0.0) s = 0.0;
	else if (s > 1.0) s = 1.0;

	Vec2R dx = Vec2R(endPoint) - Vec2R(startPoint);
	return Vec2R(startPoint[0], startPoint[1]) + s*dx;
}

void LevelSet2D::unionSurface(const LevelSet2D& unionPhi)
{
	forEachVoxelRange(Vec2ui(0), size(), [&](const Vec2ui& cell)
	{
		myPhiGrid(cell) = std::min(myPhiGrid(cell), unionPhi.interp(indexToWorld(Vec2R(cell))));
	});
}
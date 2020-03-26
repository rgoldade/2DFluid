#include "GeometricMultigridOperators.h"

namespace FluidSim2D::SimTools::GeometricMultigridOperators
{

Vec2i getChildCell(const Vec2i& cell, const int childIndex)
{
	assert(childIndex < 4);

	Vec2i childCell(cell);
	childCell *= 2;
	for (int axis : {0, 1})
	{
		if (childIndex & (1 << axis))
			++childCell[axis];
	}

	return childCell;
}

UniformGrid<CellLabels> buildCoarseCellLabels(const UniformGrid<CellLabels>& sourceCellLabels)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 && sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<CellLabels> destinationCellLabels(sourceCellLabels.size() / 2, CellLabels::EXTERIOR_CELL);

	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			// Iterate over the destination cell's children.
			Vec2i childCell = 2 * cell;

			bool hasDirichletChild = false;
			bool hasInteriorChild = false;

			for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
			{
				Vec2i localCell = cellToNode(childCell, cellIndex);

				if (sourceCellLabels(localCell) == CellLabels::DIRICHLET_CELL)
				{
					hasDirichletChild = true;
					break;
				}
				else if (sourceCellLabels(localCell) == CellLabels::INTERIOR_CELL ||
							sourceCellLabels(localCell) == CellLabels::BOUNDARY_CELL)
					hasInteriorChild = true;
			}

			if (hasDirichletChild)
				destinationCellLabels(cell) = CellLabels::DIRICHLET_CELL;
			else if (hasInteriorChild)
				destinationCellLabels(cell) = CellLabels::INTERIOR_CELL;
			else
				assert(destinationCellLabels(cell) == CellLabels::EXTERIOR_CELL);
		}
	});

	// Set boundary cells
	tbb::parallel_for(tbb::blocked_range<int>(0, destinationCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = destinationCellLabels.unflatten(cellIndex);

			if (destinationCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
				bool hasBoundary = false;

				for (int axis = 0; axis < 2 && !hasBoundary; ++axis)
					for (int direction : {0, 1})
					{
						Vec2i adjacentCell = cellToCell(cell, axis, direction);

						assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destinationCellLabels.size()[axis]);

						if (destinationCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL ||
							destinationCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
						{
							hasBoundary = true;
							break;
						}
					}

				if (hasBoundary)
					destinationCellLabels(cell) = CellLabels::BOUNDARY_CELL;
			}
		}
	});

	return destinationCellLabels;
}

std::vector<Vec2i> buildBoundaryCells(const UniformGrid<CellLabels>& sourceCellLabels,
										int boundaryWidth)
{
	assert(sourceCellLabels.size()[0] % 2 == 0 && sourceCellLabels.size()[1] % 2 == 0);

	UniformGrid<VisitedCellLabels> visitedCells(sourceCellLabels.size(), VisitedCellLabels::UNVISITED_CELL);

	tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelBoundaryCellList;

	// Build initial layer
	tbb::parallel_for(tbb::blocked_range<int>(0, sourceCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localBoundaryCellList = parallelBoundaryCellList.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = sourceCellLabels.unflatten(cellIndex);

			if (sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				localBoundaryCellList.push_back(cell);
		}
	});

	std::vector<Vec2i> boundaryCellList;
	mergeLocalThreadVectors(boundaryCellList, parallelBoundaryCellList);

	for (int layer = 1; layer < boundaryWidth; ++layer)
	{
		// Set cells to visited
		tbb::parallel_for(tbb::blocked_range<int>(0, boundaryCellList.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i cell = boundaryCellList[cellIndex];
				visitedCells(cell) = VisitedCellLabels::FINISHED_CELL;
			}
		});

		if (layer < boundaryWidth - 1)
		{
			parallelBoundaryCellList.clear();

			tbb::parallel_for(tbb::blocked_range<int>(0, boundaryCellList.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
			{
				auto &localBoundaryCellList = parallelBoundaryCellList.local();

				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec2i cell = boundaryCellList[cellIndex];

					assert(visitedCells(cell) == VisitedCellLabels::FINISHED_CELL);

					for (int axis : {0, 1})
						for (int direction : {0, 1})
						{
							Vec2i adjacentCell = cellToCell(cell, axis, direction);

							if (sourceCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL &&
								visitedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
							{
								localBoundaryCellList.push_back(adjacentCell);
							}
						}
				}
			});

			//
			// Collect new layer
			//

			boundaryCellList.clear();
			mergeLocalThreadVectors(boundaryCellList, parallelBoundaryCellList);
		}
	}

	parallelBoundaryCellList.clear();

	// Build initial layer
	tbb::parallel_for(tbb::blocked_range<int>(0, sourceCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto &localBoundaryCellList = parallelBoundaryCellList.local();
		
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = visitedCells.unflatten(cellIndex);

			if (visitedCells(cell) == VisitedCellLabels::FINISHED_CELL)
			{
				assert(sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL ||
						sourceCellLabels(cell) == CellLabels::INTERIOR_CELL);

				localBoundaryCellList.push_back(cell);
			}
		}
	});

	boundaryCellList.clear();
	mergeLocalThreadVectors(boundaryCellList, parallelBoundaryCellList);

	auto vecCompare = [](const Vec2i& vec0, const Vec2i& vec1)
	{
		if (vec0[0] < vec1[0])
			return true;
		else if (vec0[0] == vec1[0] && vec0[1] < vec1[1])
			return true;

		return false;
	};

	tbb::parallel_sort(boundaryCellList.begin(), boundaryCellList.end(), vecCompare);

	return boundaryCellList;
}

std::pair<Vec2i, int> buildExpandedDomainLabels(UniformGrid<CellLabels>& expandedDomainCellLabels,
												const UniformGrid<CellLabels>& baseDomainCellLabels)
{
	// Build domain labels with the appropriate padding to apply
	// geometric multigrid directly without a wasteful transfer
	// for each v-cycle.

	// Cap MG levels at 4 voxels in the smallest dimension
	float minLog = std::min(std::log2(float(baseDomainCellLabels.size()[0])),
								std::log2(float(baseDomainCellLabels.size()[1])));

	int mgLevels = std::ceil(minLog) - std::log2(2);

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = std::pow(2, mgLevels - 1);

	Vec2i expandedGridSize = baseDomainCellLabels.size() + 2 * Vec2i(exteriorPadding);

	for (int axis : {0, 1})
	{
		float logSize = std::log2(expandedGridSize[axis]);
		logSize = std::ceil(logSize);

		expandedGridSize[axis] = std::exp2(logSize);
	}

	Vec2i exteriorOffset = Vec2i(exteriorPadding);

	expandedDomainCellLabels.resize(expandedGridSize, CellLabels::EXTERIOR_CELL);

	// Copy initial domain labels to interior domain labels with padding
	tbb::parallel_for(tbb::blocked_range<int>(0, baseDomainCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = baseDomainCellLabels.unflatten(cellIndex);

			auto baseLabel = baseDomainCellLabels(cell);
			if (baseLabel != CellLabels::EXTERIOR_CELL)
			{
				Vec2i expandedCell = cell + exteriorOffset;
				expandedDomainCellLabels(expandedCell) = baseLabel;
			}
		}
	});

	return std::pair<Vec2i, int>(exteriorOffset, mgLevels);
}

bool unitTestCoarsening(const UniformGrid<CellLabels>& coarseCellLabels,
						const UniformGrid<CellLabels>& fineCellLabels)
{
	// The coarse cell grid must be exactly have the size of the fine cell grid.
	if (2 * coarseCellLabels.size() != fineCellLabels.size())
		return false;

	if (coarseCellLabels.size()[0] % 2 != 0 ||
		coarseCellLabels.size()[1] % 2 != 0 ||
		fineCellLabels.size()[0] % 2 != 0 ||
		fineCellLabels.size()[1] % 2 != 0)
		return false;

	{
		bool testPassed = true;

		tbb::parallel_for(tbb::blocked_range<int>(0, fineCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			if (!testPassed) return;

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i fineCell = fineCellLabels.unflatten(cellIndex);
				Vec2i coarseCell = fineCell / 2;

				// If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
				if (fineCellLabels(fineCell) == CellLabels::DIRICHLET_CELL)
				{
					if (coarseCellLabels(coarseCell) != CellLabels::DIRICHLET_CELL)
					{
						testPassed = false;
						return;
					}
				}
				else if (fineCellLabels(fineCell) == CellLabels::INTERIOR_CELL ||
							fineCellLabels(fineCell) == CellLabels::BOUNDARY_CELL)
				{
					// If the fine cell is interior, the coarse cell can be either
					// interior or Dirichlet (if a sibling cell is Dirichlet).
					if (coarseCellLabels(coarseCell) == CellLabels::EXTERIOR_CELL)
					{
						testPassed = false;
						return;
					}
				}
			}
		});

		if(!testPassed) return false;
	}
	{
		bool testPassed = true;

		tbb::parallel_for(tbb::blocked_range<int>(0, coarseCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			if (!testPassed) return;

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec2i coarseCell = coarseCellLabels.unflatten(cellIndex);

				bool foundDirichletChild = false;
				bool foundInteriorChild = false;
				bool foundExteriorChild = false;

				for (int childIndex = 0; childIndex < 4; ++childIndex)
				{
					Vec2i fineCell = getChildCell(coarseCell, childIndex);

					auto fineLabel = fineCellLabels(fineCell);

					if (fineLabel == CellLabels::DIRICHLET_CELL)
						foundDirichletChild = true;
					else if (fineLabel == CellLabels::INTERIOR_CELL ||
								fineLabel == CellLabels::BOUNDARY_CELL)
						foundInteriorChild = true;
					else if (fineLabel == CellLabels::EXTERIOR_CELL)
						foundExteriorChild = true;
				}

				auto coarseLabel = coarseCellLabels(coarseCell);
				if (coarseLabel == CellLabels::DIRICHLET_CELL)
				{
					if (!foundDirichletChild)
						testPassed = false;
				}
				else if (coarseLabel == CellLabels::INTERIOR_CELL ||
							coarseLabel == CellLabels::BOUNDARY_CELL)
				{
					if (foundDirichletChild || !foundInteriorChild)
						testPassed = false;
				}
				else if (coarseLabel == CellLabels::EXTERIOR_CELL)
				{
					if (foundDirichletChild || foundInteriorChild || !foundExteriorChild)
						testPassed = false;
				}
			}
		});

		if(!testPassed) return false;
	}

	return true;
}

bool unitTestExteriorCells(const UniformGrid<CellLabels>& cellLabels)
{
	Vec2i startCell(0);
	Vec2i endCell = cellLabels.size();

	bool exteriorCellTestPassed = true;
	for (int axis : {0, 1})
		for (int direction : {0, 1})
		{
			Vec2i localStartCell = startCell;
			Vec2i localEndCell = endCell;

			if (direction == 0)
				localEndCell[axis] = 1;
			else
				localStartCell[axis] = endCell[axis] - 1;

			forEachVoxelRange(localStartCell, localEndCell, [&](const Vec2i& cell)
			{
				if (!exteriorCellTestPassed) return;

				if (cellLabels(cell) != CellLabels::EXTERIOR_CELL)
					exteriorCellTestPassed = false;
			});
		}

	return exteriorCellTestPassed;
}

}
#include "gtest/gtest.h"

#include "SparseUniformGrid.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim2D;

//
// SparseTile tests
//

TEST(SPARSE_TILE_TESTS, DEFAULT_CONSTRUCTOR_TEST)
{
	SparseTile<int, 10> tile;
	EXPECT_TRUE(tile.constant());
}

TEST(SPARSE_TILE_TESTS, CONSTRUCTOR_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);
	EXPECT_TRUE(tile.constant());

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});
}

TEST(SPARSE_TILE_TESTS, EXPAND_TILE_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);
	tile.expand();
	EXPECT_FALSE(tile.constant());

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});

	const double value2 = value * value;
	tile.makeConstant(value2);

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value2);
	});
}

TEST(SPARSE_TILE_TESTS, COLLAPSE_TILE_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);
	tile.expand();
	EXPECT_FALSE(tile.constant());

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});

	tile.collapseIfConstant();

	EXPECT_TRUE(tile.constant());

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});
}

TEST(SPARSE_TILE_TESTS, WRITE_TILE_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);

	const double value2 = value * value;
	tile.setVoxel(Vec2i::Zero(), value2);
	EXPECT_FALSE(tile.constant());

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		if (cell == Vec2i::Zero())
		{
			EXPECT_EQ(tile.getVoxel(cell), value2);
		}
		else
		{
			EXPECT_EQ(tile.getVoxel(cell), value);
		}		
	});

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		tile.setVoxel(cell, cell[0] + cell[1] * 10);
	});

	forEachVoxelRangeReverse(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), cell[0] + cell[1] * 10);
	});
}

TEST(SPARSE_TILE_TESTS, FLATTEN_ROUND_TRIP_TEST)
{
	SparseTile<double, 10> tile;
	tile.expand();

	forEachVoxelRange(Vec2i::Zero(), Vec2i::Constant(10), [&](const Vec2i& cell)
	{
		EXPECT_TRUE(tile.unflatten(tile.flatten(cell)) == cell);
	});
}

//
// SparseUniformGrid tests
//

TEST(SPARSE_UNIFORM_GRID_TESTS, DEFAULT_CONSTRUCTOR_TEST)
{
	SparseUniformGrid<int, 10> sparseGrid;
	EXPECT_TRUE(sparseGrid.tileSize() == Vec2i::Zero());
	EXPECT_TRUE(sparseGrid.gridSize() == Vec2i::Zero());
}

TEST(SPARSE_UNIFORM_GRID_TESTS, CONSTRUCTOR_TEST)
{
	Vec2i size(100, 200);
	SparseUniformGrid<double, 10> sparseGrid(size);

	EXPECT_TRUE(sparseGrid.gridSize() == size);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.gridSize(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(sparseGrid.getVoxel(cell), double(0));
	});

	EXPECT_TRUE(sparseGrid.tileSize() == Vec2i(10, 20));

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.tile(tileCoord).constant());
	});

	EXPECT_TRUE(sparseGrid.tileCount() == 10 * 20);
}


TEST(SPARSE_UNIFORM_GRID_TESTS, VALUE_CONSTRUCTOR_TEST)
{
	Vec2i size(100, 200);
	const double value = PI;
	SparseUniformGrid<double, 10> sparseGrid(size, value);

	EXPECT_TRUE(sparseGrid.gridSize() == size);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.gridSize(), [&](const Vec2i& cell)
	{
		EXPECT_EQ(sparseGrid.getVoxel(cell), value);
	});

	EXPECT_TRUE(sparseGrid.tileSize() == Vec2i(10, 20));

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.tile(tileCoord).constant());
	});

	EXPECT_TRUE(sparseGrid.tileCount() == 10 * 20);
}

TEST(SPARSE_UNIFORM_GRID_TESTS, TILE_FLATTEN_ROUNDTRIP_TEST)
{
	Vec2i size(100, 200);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.unflattenTileIndex(sparseGrid.flattenTileIndex(tileCoord)) == tileCoord);
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, TILE_VOXEL_RANGE_TEST)
{
	Vec2i size(111, 222);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		auto tileRange = sparseGrid.tileVoxelRange(tileCoord);
		forEachVoxelRange(tileRange.first, tileRange.second, [&](const Vec2i& cell)
		{
			EXPECT_TRUE(cell[0] < size[0] && cell[1] < size[1]);
		});
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, WRITE_READ_TEST)
{
	Vec2i size(111, 222);
	SparseUniformGrid<double, 10> sparseGrid(size);

	UniformGrid<double> uniformGrid(size);

	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		uniformGrid(cell) = cell[0] + cell[1] * size[0];
		sparseGrid.setVoxel(cell, uniformGrid(cell));
	});

	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		EXPECT_EQ(uniformGrid(cell), sparseGrid.getVoxel(cell));
	});

	SparseUniformGrid<double, 10> tileWrittenSparseGrid(size);

	forEachVoxelRange(Vec2i::Zero(), tileWrittenSparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		auto tileRange = tileWrittenSparseGrid.tileVoxelRange(tileCoord);
		forEachVoxelRange(tileRange.first, tileRange.second, [&](const Vec2i& cell)
		{
			tileWrittenSparseGrid.setVoxel(cell, uniformGrid(cell));
		});
	});

	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		EXPECT_EQ(uniformGrid(cell), tileWrittenSparseGrid.getVoxel(cell));
	});

	forEachVoxelRange(Vec2i::Zero(), tileWrittenSparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		auto tileRange = tileWrittenSparseGrid.tileVoxelRange(tileCoord);
		forEachVoxelRange(tileRange.first, tileRange.second, [&](const Vec2i& cell)
		{
			EXPECT_EQ(uniformGrid(cell), tileWrittenSparseGrid.getVoxel(cell));
		});
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, WRITE_EXPAND_TEST)
{
	Vec2i size(111, 222);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});

	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		sparseGrid.setVoxel(cell, cell[0] + cell[1] * size[0]);
	});

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_FALSE(sparseGrid.isTileConstant(tileCoord));
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, COLLAPSE_TILE_TEST)
{
	Vec2i size(111, 222);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		sparseGrid.expandTile(tileCoord);
		EXPECT_FALSE(sparseGrid.isTileConstant(tileCoord));
	});

	sparseGrid.collapseTiles();

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});

	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		sparseGrid.setVoxel(cell, cell[0] + cell[1] * size[0]);
	});

	sparseGrid.collapseTiles();

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_FALSE(sparseGrid.isTileConstant(tileCoord));
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, RESIZE_TEST)
{
	Vec2i size(111, 222);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec2i::Zero(), size, [&](const Vec2i& cell)
	{
		sparseGrid.setVoxel(cell, cell[0] + cell[1] * size[0]);
	});

	Vec2i largerSize = 2 * size;
	sparseGrid.resize(largerSize);

	EXPECT_TRUE(sparseGrid.gridSize() == largerSize);

	forEachVoxelRange(Vec2i::Zero(), sparseGrid.tileSize(), [&](const Vec2i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});
}
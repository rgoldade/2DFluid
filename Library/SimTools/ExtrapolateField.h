#pragma once

#include <queue>

#include "Common.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ExtrapolateField.h/cpp
// Ryan Goldade 2016
//
// Extrapolates field from the boundary
// of a mask outward based on a simple
// BFS flood fill approach. Values
// are averaged from FINISHED neighbours.
// Note that because this process happens
// "in order" there could be some bias
// based on which boundary locations
// are inserted into the queue first.
//
//
////////////////////////////////////

template<typename Field>
class ExtrapolateField
{
public:
	ExtrapolateField(Field& field)
		: m_field(field)
		{}

	void extrapolate(const Field& mask, unsigned bandwidth = std::numeric_limits<unsigned>::max());

private:
	Field& m_field;
};

template<>
void ExtrapolateField<VectorGrid<Real>>::extrapolate(const VectorGrid<Real>& mask, unsigned bandwidth);

template<typename Field>
void ExtrapolateField<Field>::extrapolate(const Field& mask, unsigned bandwidth)
{
	// Run a BFS flood outwards from masked cells and average the values of the neighbouring "finished" cells
	// It could be made more accurate if we used the value of the "closer" cell (smaller SDF value)
	// It could be made more efficient if we truncated the BFS after a large enough distance (max SDF value)
	assert(m_field.is_matched(mask));

	UniformGrid<MarkedCells> marked_cells(m_field.size(), MarkedCells::UNVISITED);

	using Voxel = std::pair<Vec2ui, unsigned>;

	// Initialize flood fill queue
	std::queue<Voxel> marker_q;

	// If a face has fluid through it, the weight should be greater than zero
	// and so it should be marked
	for_each_voxel_range(Vec2ui(0), m_field.size(), [&](const Vec2ui& cell)
	{
		if (mask(cell) > 0.) marked_cells(cell) = MarkedCells::FINISHED;
	});

	// Load up neighbouring faces and push into queue
	for_each_voxel_range(Vec2ui(0), m_field.size(), [&](const Vec2ui& cell)
	{
		if (marked_cells(cell) == MarkedCells::FINISHED)
		{
			for (unsigned dir = 0; dir < 4; ++dir)
			{
				Vec2i cidx = Vec2i(cell) + cell_to_cell[dir];

				unsigned axis = dir / 2;
				// Boundary check
				if (cidx[axis] < 0 ||
					cidx[axis] >= marked_cells.size()[axis]) continue;

				Vec2ui ucidx(cidx);
				if (marked_cells(ucidx) == MarkedCells::UNVISITED)
				{
					marker_q.push(Voxel(ucidx, 1));
					marked_cells(ucidx) = MarkedCells::VISITED;
				}
			}
		}
	});

	while (!marker_q.empty())
	{
		// Store reference to updated faces to set as finished after
		// this layer is completed
		std::queue<Voxel> temp_q = marker_q;
		// Store references to next layer of faces
		std::queue<Voxel> new_layer;

		while (!marker_q.empty())
		{
			Voxel voxel = marker_q.front();
			Vec2ui idx = voxel.first;
			marker_q.pop();

			assert(marked_cells(idx) == MarkedCells::VISITED);

			Real val = 0.;
			Real count = 0.;

			if (voxel.second < bandwidth)
			{
				for (unsigned dir = 0; dir < 4; ++dir)
				{
					Vec2i cidx = Vec2i(idx) + cell_to_cell[dir];

					unsigned axis = dir / 2;
					//Boundary check
					if (cidx[axis] < 0 || cidx[axis] >= marked_cells.size()[axis]) continue;

					Vec2ui ucidx(cidx);
					if (marked_cells(ucidx) == MarkedCells::FINISHED)
					{
						val += m_field(ucidx);
						++count;
					}
					else if (marked_cells(ucidx) == MarkedCells::UNVISITED)
					{
						new_layer.push(Voxel(ucidx, voxel.second + 1));
						marked_cells(ucidx) = MarkedCells::VISITED;
					}
				}

				assert(count > 0);
				m_field(idx) = val / count;
			}
		}

		//Set update cells to finished
		while (!temp_q.empty())
		{
			Voxel voxel = temp_q.front();
			Vec2ui idx = voxel.first;
			temp_q.pop();
			assert(marked_cells(idx) == MarkedCells::VISITED);
			marked_cells(idx) = MarkedCells::FINISHED;
		}

		//Copy new queue
		marker_q = new_layer;
	}
}
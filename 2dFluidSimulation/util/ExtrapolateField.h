#pragma once
#include <queue>
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

	void extrapolate(const Field& mask, size_t bandwidth = std::numeric_limits<size_t>::max());

private:
	Field& m_field;
};

template<>
void ExtrapolateField<VectorGrid<Real>>::extrapolate(const VectorGrid<Real>& mask, size_t bandwidth);

template<typename Field>
void ExtrapolateField<Field>::extrapolate(const Field& mask, size_t bandwidth)
{
	// Run a BFS flood outwards from masked cells and average the values of the neighbouring "finished" cells
	// It could be made more accurate if we used the value of the "closer" cell (smaller SDF value)
	// It could be made more efficient if we truncated the BFS after a large enough distance (max SDF value)

	assert(m_field.size()[0] == mask.size()[0] &&
			m_field.size()[1] == mask.size()[1]);

	UniformGrid<marked> marked_cells(m_field.size(), UNVISITED);

	typedef std::pair<Vec2st, size_t> Voxel;
	// Initialize flood fill queue
	std::queue<Voxel> marker_q;

	// If a face has fluid through it, the weight should be greater than zero
	// and so it should be marked
	for (size_t x = 0; x < m_field.size()[0]; ++x)
		for (size_t y = 0; y < m_field.size()[1]; ++y)
		{
			if (mask(x, y) > 0.) marked_cells(x, y) = FINISHED;
		}

	// Load up neighbouring faces and push into queue
	for (size_t x = 0; x < m_field.size()[0]; ++x)
		for (size_t y = 0; y < m_field.size()[1]; ++y)
		{
			if (marked_cells(x, y) == FINISHED)
			{
				for (size_t c = 0; c < 4; ++c)
				{
					Vec2i cidx = Vec2i(x, y) + cell_offset[c];

					// Boundary check
					if (cidx[0] < 0 || cidx[1] < 0 ||
						cidx[0] >= marked_cells.size()[0] ||
						cidx[1] >= marked_cells.size()[1]) continue;

					if (marked_cells(cidx[0], cidx[1]) == UNVISITED)
					{
						marker_q.push(Voxel(Vec2st(cidx[0], cidx[1]), 1));
						marked_cells(cidx[0], cidx[1]) = VISITED;
					}
				}
			}
		}

	while (!marker_q.empty())
	{
		// Store reference to updated faces to set as finished after
		// this layer is completed
		std::queue<Voxel> temp_q = marker_q;
		// Store references to next layer of faces
		std::queue<Voxel> new_layer;

		while (!marker_q.empty())
		{
			auto voxel = marker_q.front();
			Vec2st idx = voxel.first;
			marker_q.pop();

			assert(marked_cells(idx[0], idx[1]) == VISITED);

			Real val = 0.;
			Real count = 0.;

			if (voxel.second < bandwidth)
			{
				for (size_t c = 0; c < 4; ++c)
				{
					Vec2i cidx = Vec2i(idx) + cell_offset[c];
					//Boundary check
					if (cidx[0] < 0 || cidx[1] < 0 ||
						cidx[0] >= marked_cells.size()[0] ||
						cidx[1] >= marked_cells.size()[1]) continue;

					if (marked_cells(cidx[0], cidx[1]) == FINISHED)
					{
						val += m_field(cidx[0], cidx[1]);
						++count;
					}
					else if (marked_cells(cidx[0], cidx[1]) == UNVISITED)
					{
						new_layer.push(Voxel(Vec2st(cidx), voxel.second + 1));
						marked_cells(cidx[0], cidx[1]) = VISITED;
					}
				}

				assert(count > 0);
				m_field(idx[0], idx[1]) = val / count;
			}
		}

		//Set update cells to finished
		while (!temp_q.empty())
		{
			auto voxel = temp_q.front();
			Vec2st idx = voxel.first;
			temp_q.pop();
			assert(marked_cells(idx[0], idx[1]) == VISITED);
			marked_cells(idx[0], idx[1]) = FINISHED;
		}

		//Copy new queue
		marker_q = new_layer;
	}
}
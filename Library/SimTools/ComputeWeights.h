#pragma once

#include "Common.h"
#include "ScalarGrid.h"
#include "VectorGrid.h"
#include "LevelSet2D.h"

///////////////////////////////////
//
// ComputeWeights.h/cpp
// Ryan Goldade 2017
//
// Useful collection of tools to compute
// control volume weights for use in both
// pressure projection and viscosity solves.
//
////////////////////////////////////

class ComputeWeights
{
public:
	ComputeWeights(const LevelSet2D &surface, const LevelSet2D &collision)
		: m_surface(surface)
		, m_collision(collision)
		{
			assert(m_surface.is_matched(m_collision));
		}

	void compute_gf_weights(VectorGrid<Real>& liquid_weights);
	void compute_cutcell_weights(VectorGrid<Real>& cc_weights);
	void compute_supersampled_volumes(ScalarGrid<Real>& volume_weights, unsigned samples, bool use_collision = false);

private:

	const LevelSet2D &m_surface, &m_collision;
};

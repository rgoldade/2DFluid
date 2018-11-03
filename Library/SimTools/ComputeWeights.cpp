#include "ComputeWeights.h"

Real length_fraction(Real phi0, Real phi1)
{
	Real theta = 0.;
	if (phi0 < 0 && phi1 < 0)
		theta = 1.;
	else if (phi0 < 0 && phi1 >= 0)
		theta = phi0 / (phi0 - phi1);
	else if (phi0 >= 0 && phi1 < 0)
		theta = phi1 / (phi1 - phi0);

	return theta;
}

void ComputeWeights::compute_gf_weights(VectorGrid<Real>& gf_weights)
{
	assert(gf_weights.sample_type() == VectorGridSettings::SampleType::STAGGERED &&
			gf_weights.size(0)[0] - 1 == m_surface.size()[0] &&
			gf_weights.size(0)[1] == m_surface.size()[1] &&
			gf_weights.size(1)[0] == m_surface.size()[0] &&
			gf_weights.size(1)[1] - 1 == m_surface.size()[1] &&
			gf_weights.xform() == m_surface.xform());

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		for_each_voxel_range(Vec2ui(0), gf_weights.size(axis), [&](const Vec2ui& face)
		{
			Vec2i backward_cell = Vec2i(face) + face_to_cell[axis][0];
			Vec2i forward_cell = Vec2i(face) + face_to_cell[axis][1];

			bool out_of_bounds = false;
			if (backward_cell[axis] < 0 || forward_cell[axis] >= m_surface.size()[axis])
				gf_weights.grid(axis)(face) = 0.;
			else
			{
				Real phib = m_surface(Vec2ui(backward_cell));
				Real phif = m_surface(Vec2ui(forward_cell));

				gf_weights.grid(axis)(face) = length_fraction(phib, phif);
			}
		});
	}
}

void ComputeWeights::compute_cutcell_weights(VectorGrid<Real>& cc_weights)
{
	// Make sure that the cut cell weights are the staggered equivalent of
	// the underlying collision surface

	assert(cc_weights.sample_type() == VectorGridSettings::SampleType::STAGGERED &&
			cc_weights.size(0)[0] - 1 == m_collision.size()[0] &&
			cc_weights.size(0)[1] == m_collision.size()[1] &&
			cc_weights.size(1)[0] == m_collision.size()[0] &&
			cc_weights.size(1)[1] - 1 == m_collision.size()[1] &&
			cc_weights.xform() == m_collision.xform());

	for (unsigned axis = 0; axis < 2; ++axis)
	{
		for_each_voxel_range(Vec2ui(0), cc_weights.size(axis), [&](const Vec2ui& face)
		{
			unsigned other_axis = (axis + 1) % 2;

			Vec2R offset(0); offset[other_axis] = .5;

			Vec2R pos0 = cc_weights.idx_to_ws(Vec2R(face) - offset, axis);
			Vec2R pos1 = cc_weights.idx_to_ws(Vec2R(face) + offset, axis);

			Real weight = 1. - length_fraction(m_collision.interp(pos0), m_collision.interp(pos1));
			weight = clamp(weight, 0., 1.);

			cc_weights(face, axis) = weight;
		});
	}
}

// There is no assumption about grid alignment for this method because
// we're computing weights for centers, faces, nodes, etc. that each
// have their internal index space cell offsets. We can't make any
// easy general assumptions about indices between grids anymore.
void ComputeWeights::compute_supersampled_volumes(ScalarGrid<Real>& volume_weights, unsigned samples, bool use_collision)
{
	Real sample_dx = 1. / Real(samples);

	const LevelSet2D *surface = use_collision ? &m_collision : &m_surface;
	Real sign = use_collision ? -1. : 1.;

	// Loop over each cell in the grid
	for_each_voxel_range(Vec2ui(0), volume_weights.size(), [&](const Vec2ui& cell)
	{
		if (sign * surface->interp(volume_weights.idx_to_ws(Vec2R(cell))) > surface->dx() * 2.)
		{
			volume_weights(cell) = 0.;
			return;
		}

		// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
		// first sample point is the .5 * sample_dx closer to (i,j).
		int volcount = 0;
		for (Real x = Real(cell[0]) - .5 + .5 * sample_dx; x < Real(cell[0]) + .5; x += sample_dx)
			for (Real y = Real(cell[1]) - .5 + .5 * sample_dx; y < Real(cell[1]) + .5; y += sample_dx)
			{
				Vec2R wpos = volume_weights.idx_to_ws(Vec2R(x, y));
				if (sign * surface->interp(wpos) <= 0.) ++volcount;
			}

		volume_weights(cell) = Real(volcount * sample_dx * sample_dx);
	});
}
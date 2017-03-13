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
	assert(gf_weights.sample_type() == VectorGridSettings::STAGGERED &&
			gf_weights.size(0)[0] - 1 == m_surface.size()[0] &&
			gf_weights.size(0)[1] == m_surface.size()[1] &&
			gf_weights.size(1)[0] == m_surface.size()[0] &&
			gf_weights.size(1)[1] - 1 == m_surface.size()[1] &&
			gf_weights.xform() == m_surface.xform());

	for (int y = 0; y < gf_weights.size(0)[1]; ++y)
	{
		gf_weights(0, y, 0) = 0.;
		gf_weights(gf_weights.size(0)[0] - 1, y, 0) = 0.;
	}

	for (int x = 1; x < gf_weights.size(0)[0] - 1; ++x)
		for (int y = 0; y < gf_weights.size(0)[1]; ++y)
		{
			Vec2i cb = Vec2i(x - 1, y);
			Vec2i cf = Vec2i(x, y);

			Real phib = m_surface(cb[0], cb[1]);
			Real phif = m_surface(cf[0], cf[1]);

			gf_weights(x, y, 0) = length_fraction(phib, phif);
		}

	for (int x = 0; x < gf_weights.size(1)[0]; ++x)
	{
		gf_weights(x, 0, 1) = 0.;
		gf_weights(x, gf_weights.size(1)[1] - 1, 1) = 0.;
	}

	for (int x = 0; x < gf_weights.size(1)[0]; ++x)
		for (int y = 1; y < gf_weights.size(1)[1] - 1; ++y)
		{
			Vec2i cb = Vec2i(x, y - 1);
			Vec2i cf = Vec2i(x, y);

			Real phib = m_surface(cb[0], cb[1]);
			Real phif = m_surface(cf[0], cf[1]);

			gf_weights(x, y, 1) = length_fraction(phib, phif);
		}
}

void ComputeWeights::compute_cutcell_weights(VectorGrid<Real>& cc_weights)
{
	// Make sure that the cut cell weights are the staggered equivalent of
	// the underlying collision surface

	assert(cc_weights.sample_type() == VectorGridSettings::STAGGERED &&
			cc_weights.size(0)[0] - 1 == m_surface.size()[0] &&
			cc_weights.size(0)[1] == m_surface.size()[1] &&
			cc_weights.size(1)[0] == m_surface.size()[0] &&
			cc_weights.size(1)[1] - 1 == m_surface.size()[1] &&
			cc_weights.xform() == m_surface.xform());

	for (int y = 0; y < cc_weights.size(0)[1]; ++y)
	{
		cc_weights(0, y, 0) = 0.;
		cc_weights(cc_weights.size(0)[0] - 1, y, 0) = 0.;
	}

	for (int x = 1; x < cc_weights.size(0)[0] - 1; ++x)
		for (int y = 0; y < cc_weights.size(0)[1]; ++y)
		{
			Vec2R offset(0., .5);
			Vec2R pos0 = cc_weights.idx_to_ws(Vec2R(x, y) - offset, 0);
			Vec2R pos1 = cc_weights.idx_to_ws(Vec2R(x, y) + offset, 0);

			Real weight = 1. - length_fraction(m_collision.interp(pos0), m_collision.interp(pos1));
			weight = clamp(weight, 0., 1.);

			cc_weights(x, y, 0) = weight;
		}

	for (int x = 0; x < cc_weights.size(1)[0]; ++x)
	{
		cc_weights(x, 0, 1) = 0.;
		cc_weights(x, cc_weights.size(1)[1] - 1, 1) = 0.;
	}

	for (int x = 0; x < cc_weights.size(1)[0]; ++x)
		for (int y = 1; y < cc_weights.size(1)[1] - 1; ++y)
		{
			Vec2R offset(.5, 0.);
			Vec2R pos0 = cc_weights.idx_to_ws(Vec2R(x, y) - offset, 1);
			Vec2R pos1 = cc_weights.idx_to_ws(Vec2R(x, y) + offset, 1);

			Real weight = 1. - length_fraction(m_collision.interp(pos0), m_collision.interp(pos1));
			weight = clamp(weight, 0., 1.);

			cc_weights(x, y, 1) = weight;
		}

}

// There is no assumption about grid alignment for this method because
// we're computing weights for centers, faces, nodes, etc. that each
// have their internal index space cell offsets. We can't make any
// easy general assumptions about indices between grids anymore.
void ComputeWeights::compute_supersampled_volumes(ScalarGrid<Real>& volume_weights, size_t samples, bool use_collision)
{
	Real sample_dx = 1. / (Real)samples;

	const LevelSet2D *surface = use_collision ? &m_collision : &m_surface;
	Real sign = use_collision ? -1. : 1.;

	// Loop over each cell in the grid
	for (int i = 0; i < volume_weights.size()[0]; ++i)
		for (int j = 0; j < volume_weights.size()[1]; ++j)
		{
			// Early exit if we're far enough away from the surface that we can't possibly get anything
			Real tempphi = sign * surface->interp(volume_weights.idx_to_ws(Vec2R(i, j)));
			Real compare_val = surface->dx() * 2.;

			if (sign * surface->interp(volume_weights.idx_to_ws(Vec2R(i, j))) > surface->dx() * 2.)
			{
				volume_weights(i, j) = 0.;
				continue;
			}

			// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
			// first sample point is the .5 * sample_dx closer to (i,j).
			int volcount = 0;
			for (Real x = ((Real)i - .5) + (.5 * sample_dx); x < (Real)i + .5; x += sample_dx)
				for (Real y = ((Real)j - .5) + (.5 * sample_dx); y < (Real)j + .5; y += sample_dx)
				{
					Vec2R wpos = volume_weights.idx_to_ws(Vec2R(x, y));
					Real tempinterp = surface->interp(wpos);
					if (tempinterp > 0.)
						int a = 2;
					if (sign * surface->interp(wpos) <= 0.) ++volcount;
				}

			volume_weights(i, j) = (Real)volcount * sample_dx * sample_dx;
		}
}
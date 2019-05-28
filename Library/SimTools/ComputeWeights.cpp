#include "ComputeWeights.h"

VectorGrid<Real> computeGhostFluidWeights(const LevelSet& surface)
{
	VectorGrid<Real> ghostFluidWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);
	
	for (int axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), ghostFluidWeights.size(axis), [&](const Vec2i& face)
		{
			Vec2i backwardCell = faceToCell(face, axis, 0);
			Vec2i forwardCell = faceToCell(face, axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] >= surface.size()[axis])
				ghostFluidWeights.grid(axis)(face) = 0.;
			else
			{
				Real phiBackward = surface(backwardCell);
				Real phiForward = surface(forwardCell);
				
				if (phiBackward < 0 && phiForward < 0)
					ghostFluidWeights(face, axis) = 1;
				else if ((phiBackward < 0. && phiForward >= 0.) ||
							(phiBackward >= 0. && phiForward < 0.))
					ghostFluidWeights(face, axis) = lengthFraction(phiBackward, phiForward);
			}
		});
	}

	return ghostFluidWeights;
}

VectorGrid<Real> computeCutCellWeights(const LevelSet& surface, bool invert)
{
	VectorGrid<Real> cutCellWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (unsigned axis : {0, 1})
	{
		forEachVoxelRange(Vec2i(0), cutCellWeights.size(axis), [&](const Vec2i& face)
		{
			int otherAxis = (axis + 1) % 2;

			Vec2R offset(0); offset[otherAxis] = .5;

			Vec2R pos0 = cutCellWeights.indexToWorld(Vec2R(face) - offset, axis);
			Vec2R pos1 = cutCellWeights.indexToWorld(Vec2R(face) + offset, axis);

			Real weight = lengthFraction(surface.interp(pos0), surface.interp(pos1));

			if (invert) weight = 1. - weight;

			// Clamp below zero and above one.
			weight = Util::clamp(weight, 0., 1.);

			cutCellWeights(face, axis) = weight;
		});
	}

	return cutCellWeights;
}

// There is no assumption about grid alignment for this method because
// we're computing weights for centers, faces, nodes, etc. that each
// have their internal index space cell offsets. We can't make any
// easy general assumptions about indices between grids anymore.
ScalarGrid<Real> computeSuperSampledAreas(const LevelSet& surface, ScalarGridSettings::SampleType sampleType, int samples)
{
	ScalarGrid<Real> volumes(surface.xform(), surface.size(), 0, sampleType);

	Real dx = 1. / Real(samples);
	Real sampleArea = Util::sqr(dx);

	// Loop over each cell in the grid
	forEachVoxelRange(Vec2i(0), volumes.size(), [&](const Vec2i& cell)
	{
		if (surface.interp(volumes.indexToWorld(Vec2R(cell))) > 2. * surface.dx())
			return;

		// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
		// first sample point is the .5 * sample_dx closer to (i,j).
		int sampleCount = 0;
		Vec2R start = Vec2R(cell) - Vec2R(.5 - .5 * dx);
		Vec2R end = Vec2R(cell) + Vec2R(.5);

		Vec2R sample;

		for (sample[0] = start[0]; sample[0] < end[0]; sample[0] += dx)
			for (sample[1] = start[1]; sample[1] < end[1]; sample[1] += dx)
			{
				Vec2R worldPosition = volumes.indexToWorld(sample);
				if (surface.interp(worldPosition) <= 0.) ++sampleCount;
			}

		volumes(cell) = Real(sampleCount * sampleArea);
	});

	return volumes;
}

VectorGrid<Real> computeSuperSampledFaceAreas(const LevelSet& surface, int samples)
{
	VectorGrid<Real> volumes(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	Real dx = 1. / Real(samples);
	Real sampleArea = Util::sqr(dx);

	for (auto axis : { 0,1 })
	{
		// Loop over each cell in the grid
		forEachVoxelRange(Vec2i(0), volumes.size(axis), [&](const Vec2i& face)
		{
			if (surface.interp(volumes.indexToWorld(Vec2R(face), axis)) > 2. * surface.dx())
				return;

			// Loop over super samples internally. i -.5 is the index space boundary of the sample. The 
			// first sample point is the .5 * sample_dx closer to (i,j).
			int sampleCount = 0;
			Vec2R start = Vec2R(face) - Vec2R(.5 - .5 * dx);
			Vec2R end = Vec2R(face) + Vec2R(.5);

			Vec2R sample;

			for (sample[0] = start[0]; sample[0] < end[0]; sample[0] += dx)
				for (sample[1] = start[1]; sample[1] < end[1]; sample[1] += dx)
				{
					Vec2R worldPosition = volumes.indexToWorld(sample, axis);
					if (surface.interp(worldPosition) <= 0.) ++sampleCount;
				}

			volumes(face, axis) = Real(sampleCount * sampleArea);
		});
	}

	return volumes;
}
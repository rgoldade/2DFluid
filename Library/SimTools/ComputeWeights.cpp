#include "ComputeWeights.h"

#include <array>

#include "tbb/tbb.h"

namespace FluidSim2D::SimTools
{

VectorGrid<float> computeGhostFluidWeights(const LevelSet& surface)
{
	VectorGrid<float> ghostFluidWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, ghostFluidWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = ghostFluidWeights.grid(axis).unflatten(faceIndex);

				Vec2i backwardCell = faceToCell(face, axis, 0);
				Vec2i forwardCell = faceToCell(face, axis, 1);

				if (backwardCell[axis] < 0 || forwardCell[axis] >= surface.size()[axis])
					continue;
				else
				{
					float phiBackward = surface(backwardCell);
					float phiForward = surface(forwardCell);

					ghostFluidWeights(face, axis) = lengthFraction(phiBackward, phiForward);
				}
			}
		});
	}

	return ghostFluidWeights;
}

VectorGrid<float> computeCutCellWeights(const LevelSet& surface, bool invert)
{
	VectorGrid<float> cutCellWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, cutCellWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = cutCellWeights.grid(axis).unflatten(faceIndex);

				int otherAxis = (axis + 1) % 2;

				Vec2f offset(0); offset[otherAxis] = .5;

				Vec2f backwardNodePoint = cutCellWeights.indexToWorld(Vec2f(face) - offset, axis);
				Vec2f forwardNodePoint = cutCellWeights.indexToWorld(Vec2f(face) + offset, axis);

				float weight = lengthFraction(surface.biLerp(backwardNodePoint), surface.biLerp(forwardNodePoint));

				if (invert)
					weight = 1. - weight;

				if (weight > 0)
					cutCellWeights(face, axis) = weight;
			}
		});
	}

	return cutCellWeights;
}

// There is no assumption about grid alignment for this method because
// we're computing weights for centers, faces, nodes, etc. that each
// have their internal index space cell offsets. We can't make any
// easy general assumptions about indices between grids anymore.
ScalarGrid<float> computeSupersampledAreas(const LevelSet& surface, ScalarGridSettings::SampleType sampleType, int samples)
{
	assert(samples > 0);

	ScalarGrid<float> areas(surface.xform(), surface.size(), 0, sampleType);

	float dx = 1. / float(samples);
	float sampleArea = sqr(dx);

	tbb::parallel_for(tbb::blocked_range<int>(0, areas.voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
		{
			Vec2i sampleCoord = areas.unflatten(sampleIndex);

			float sdf = surface.biLerp(areas.indexToWorld(Vec2f(sampleCoord)));
			
			if (sdf > 2. * surface.dx())
				continue;

			if (sdf <= -2. * surface.dx())
			{
				areas(sampleCoord) = 1;
				continue;
			}

			Vec2f start = Vec2f(sampleCoord) - Vec2f(.5) + Vec2f(.5 * dx);
			Vec2f end = Vec2f(sampleCoord) + Vec2f(.5) - Vec2f(.5 * dx);

			Vec2f sample;
			float insideMaterialCount = 0;

			for (sample[0] = start[0]; sample[0] <= end[0]; sample[0] += dx)
				for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += dx)
				{
					Vec2f worldSamplePoint = areas.indexToWorld(sample);

					if (surface.biLerp(worldSamplePoint) <= 0.)
						++insideMaterialCount;
				}

			if (insideMaterialCount > 0)
			{
				float supersampledArea = insideMaterialCount * sampleArea;
				supersampledArea = clamp(supersampledArea, float(0), float(1));
				areas(sampleCoord) = supersampledArea;
			}
		}
	});

	return areas;
}

VectorGrid<float> computeSupersampledFaceAreas(const LevelSet& surface, int samples)
{
	assert(samples > 0);

	VectorGrid<float> areas(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	float dx = 1. / float(samples);
	float sampleArea = sqr(dx);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, areas.grid(axis).voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
			{
				Vec2i sampleCoord = areas.grid(axis).unflatten(sampleIndex);

				float sdf = surface.biLerp(areas.indexToWorld(Vec2f(sampleCoord), axis));

				if (sdf > 2. * surface.dx())
					continue;

				if (sdf <= -2. * surface.dx())
				{
					areas(sampleCoord, axis) = 1;
					continue;
				}

				Vec2f start = Vec2f(sampleCoord) - Vec2f(.5) + Vec2f(.5 * dx);
				Vec2f end = Vec2f(sampleCoord) + Vec2f(.5) - Vec2f(.5 * dx);

				Vec2f sample;
				float insideMaterialCount = 0;

				for (sample[0] = start[0]; sample[0] <= end[0]; sample[0] += dx)
					for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += dx)
					{
						Vec2f worldSamplePoint = areas.indexToWorld(sample, axis);

						if (surface.biLerp(worldSamplePoint) <= 0.)
							++insideMaterialCount;
					}

				if (insideMaterialCount > 0)
				{
					float supersampledArea = insideMaterialCount * sampleArea;
					supersampledArea = clamp(supersampledArea, float(0), float(1));
					areas(sampleCoord, axis) = supersampledArea;
				}
			}
		});
	}

	return areas;
}

}
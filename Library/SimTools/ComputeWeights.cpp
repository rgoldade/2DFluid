#include "ComputeWeights.h"

#include <array>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

namespace FluidSim2D
{

VectorGrid<double> computeGhostFluidWeights(const LevelSet& surface)
{
	VectorGrid<double> ghostFluidWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

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
					double phiBackward = surface(backwardCell);
					double phiForward = surface(forwardCell);

					ghostFluidWeights(face, axis) = lengthFraction(phiBackward, phiForward);
				}
			}
		});
	}

	return ghostFluidWeights;
}

VectorGrid<double> computeCutCellWeights(const LevelSet& surface, bool invert)
{
	VectorGrid<double> cutCellWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, cutCellWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec2i face = cutCellWeights.grid(axis).unflatten(faceIndex);

				int otherAxis = (axis + 1) % 2;

				Vec2d offset = Vec2d::Zero(); offset[otherAxis] = .5;

				Vec2d backwardNodePoint = cutCellWeights.indexToWorld(face.cast<double>() - offset, axis);
				Vec2d forwardNodePoint = cutCellWeights.indexToWorld(face.cast<double>() + offset, axis);

				double weight = lengthFraction(surface.biLerp(backwardNodePoint), surface.biLerp(forwardNodePoint));

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
ScalarGrid<double> computeSupersampledAreas(const LevelSet& surface, ScalarGridSettings::SampleType sampleType, int samples)
{
	assert(samples > 0);

	ScalarGrid<double> areas(surface.xform(), surface.size(), 0, sampleType);

	double sampleDx = 1. / double(samples);
    double sampleCount = std::pow(double(samples), 2);
	tbb::parallel_for(tbb::blocked_range<int>(0, areas.voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
		{
			Vec2i sampleCoord = areas.unflatten(sampleIndex);
			double sdf = surface.biLerp(areas.indexToWorld(sampleCoord.cast<double>()));

			if (sdf > 2. * surface.dx())
				continue;

			if (sdf <= -2. * surface.dx())
			{
				areas(sampleCoord) = 1;
				continue;
			}

			Vec2d start = sampleCoord.cast<double>() - .5 * Vec2d::Ones() + .5 * Vec2d(sampleDx, sampleDx);
            Vec2d end = sampleCoord.cast<double>() + .5 * Vec2d::Ones();

			int insideMaterialCount = 0;

			for (Vec2d sample = start; sample[0] <= end[0]; sample[0] += sampleDx)
                for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += sampleDx)
				{
					Vec2d worldSamplePoint = areas.indexToWorld(sample);

					if (surface.biLerp(worldSamplePoint) <= 0.)
						++insideMaterialCount;
				}

			if (insideMaterialCount > 0)
			{
				double supersampledArea = double(insideMaterialCount) / sampleCount;
				supersampledArea = std::clamp(supersampledArea, 0., 1.);
				areas(sampleCoord) = supersampledArea;
			}
		}
	});

	return areas;
}

VectorGrid<double> computeSupersampledFaceAreas(const LevelSet& surface, int samples)
{
	assert(samples > 0);

	VectorGrid<double> areas(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	double sampleDx = 1. / double(samples);
    double sampleCount = std::pow(double(samples), 2);

	for (int axis : {0, 1})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, areas.grid(axis).voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
			{
				Vec2i face = areas.grid(axis).unflatten(sampleIndex);
                double sdf = surface.biLerp(areas.indexToWorld(face.cast<double>(), axis));

				if (sdf > 2. * surface.dx())
					continue;

				if (sdf <= -2. * surface.dx())
				{
                    areas(face, axis) = 1;
					continue;
				}

				Vec2d start = face.cast<double>() - .5 * Vec2d::Ones() + .5 * Vec2d(sampleDx, sampleDx);
                Vec2d end = face.cast<double>() + .5 * Vec2d::Ones();

				double insideMaterialCount = 0;

				for (Vec2d point = start; point[0] <= end[0]; point[0] += sampleDx)
                    for (point[1] = start[1]; point[1] <= end[1]; point[1] += sampleDx)
					{
                        Vec2d worldSamplePoint = areas.indexToWorld(point, axis);

						if (surface.biLerp(worldSamplePoint) <= 0.)
							++insideMaterialCount;
					}

				if (insideMaterialCount > 0)
				{
					double supersampledArea = insideMaterialCount / sampleCount;
					supersampledArea = std::clamp(supersampledArea, 0., 1.);
                    areas(face, axis) = supersampledArea;
				}
			}
		});
	}

	return areas;
}

}
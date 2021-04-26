#include "FluidParticles.h"

#include <random>

namespace FluidSim2D
{

static Vec2d randomizer(const Vec2i& coord, size_t count, double seed)
{
	int pos0 = int((5915587277 * coord[0]) ^ (3367900313 * count) ^ size_t(3267000013. * seed));
	int pos1 = int((2860486313 * coord[1]) ^ (9576890767 * count) ^ size_t(5463458053. * seed));

	pos0 = abs(pos0 % 100);
	pos1 = abs(pos1 % 100);

	return Vec2d(double(pos0) / 100. - .5, double(pos1) / 100. - .5);
};

void FluidParticles::drawPoints(Renderer& renderer, const Vec3d& colour, double pointSize) const
{
	renderer.addPoints(myParticles, colour, pointSize);
}

void FluidParticles::drawVelocity(Renderer& renderer, const Vec3d& colour, double length) const
{
	assert(myTrackVelocity);

	VecVec2d startPoints;
	VecVec2d endPoints;

	for (size_t particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		startPoints.push_back(myParticles[particleIndex]);
		endPoints.push_back(myParticles[particleIndex] + myVelocity[particleIndex] * length);
	}

	renderer.addLines(startPoints, endPoints, colour);
}

void FluidParticles::init(const LevelSet& surface)
{
	myParticles.clear();

	forEachVoxelRange(Vec2i::Zero(), surface.size(), [&](const Vec2i& cell)
	{
		if (surface(cell) < surface.dx())
		{
			double sampleCount = (surface(cell) > -2. * surface.dx() - myParticleRadius) ? myParticleDensity * myOversampleRate : myParticleDensity;

			for (int seedCount = 0; seedCount < sampleCount; ++seedCount)
			{
				Vec2d randOffset = randomizer(cell, seedCount, 15);
				Vec2d indexPoint = cell.cast<double>() + randOffset;
				Vec2d worldPoint = surface.indexToWorld(indexPoint);

				if (surface.biLerp(worldPoint) <= -myParticleRadius)
					myParticles.push_back(worldPoint);
			}
		}
	});

	myVelocity.resize(myParticles.size(), Vec2d::Zero());
}

void FluidParticles::setVelocity(const VectorGrid<double>& vel)
{
	assert(myTrackVelocity);
	myVelocity.resize(myParticles.size());

	for (int pointIndex = 0; pointIndex < myParticles.size(); ++pointIndex)
		myVelocity[pointIndex] = vel.biLerp(myParticles[pointIndex]);
}

void FluidParticles::applyVelocity(VectorGrid<double>& velocity)
{
	assert(myTrackVelocity);
	for (int axis : {0, 1})
	{
		UniformGrid<double> denominator(velocity.size(axis), 0);
		UniformGrid<double> numerator(velocity.size(axis), 0);

		for (size_t particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
		{
			// Iterate over nearby voxels
			AlignedBox2i bbox;
			bbox.extend(velocity.worldToIndex(myParticles[particleIndex], axis).array().floor().matrix().cast<int>());
			bbox.extend(velocity.worldToIndex(myParticles[particleIndex], axis).array().ceil().matrix().cast<int>());

			AlignedBox2i clampBox;
			clampBox.extend(Vec2i::Zero());
			clampBox.extend(velocity.size(axis) - Vec2i::Ones());
			
			bbox.clamp(clampBox);

			for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i)
				for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j)
				{
					if (i < 0 || j < 0 || i >= velocity.size(axis)[0]
						|| j >= velocity.size(axis)[1]) continue;

					Vec2d gridPoint = velocity.indexToWorld(Vec2d(i, j), axis);

					if ((gridPoint - myParticles[particleIndex]).norm() <= std::pow(velocity.dx(), 2))
					{
						double k = 1.f - (myParticles[particleIndex] - gridPoint).norm() / velocity.dx();
						numerator(i, j) += k * myVelocity[particleIndex][axis];
						denominator(i, j) += k;
					}
				}
		}

		forEachVoxelRange(Vec2i::Zero(), velocity.size(axis), [&](const Vec2i& cell)
		{
			if (denominator(cell) > 0.)
			{
				velocity(cell, axis) = numerator(cell) / denominator(cell);
			}
		});
	}
}
void FluidParticles::incrementVelocity(VectorGrid<double>& velocity)
{
	assert(myVelocity.size() == myParticles.size() && myTrackVelocity);

	for (int particleIndex = 0; particleIndex < int(myParticles.size()); ++particleIndex)
		myVelocity[particleIndex] += velocity.biLerp(myParticles[particleIndex]);
}

void FluidParticles::blendVelocity(const VectorGrid<double>& oldVelocity,
	const VectorGrid<double>& newVelocity,
	double blend)
{
	assert(myVelocity.size() == myParticles.size() && myTrackVelocity);

	for (int particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		Vec2d particleVelocity = myVelocity[particleIndex];
		Vec2d picVelocity = newVelocity.biLerp(myParticles[particleIndex]);
		Vec2d flipVelocity = picVelocity - oldVelocity.biLerp(myParticles[particleIndex]);

		myVelocity[particleIndex] = (1. - blend) * picVelocity + (blend) * (particleVelocity + flipVelocity);
	}
}

LevelSet FluidParticles::surfaceParticles(const Transform& xform, const Vec2i& size, int narrowBand) const
{
	LevelSet surface(xform, size, narrowBand);
	double dx = xform.dx();

	for (const Vec2d& particlePoint : myParticles)
	{
		// Iterate over nearby voxels
		AlignedBox2i bbox;

		Vec2i minIndex = surface.worldToIndex(particlePoint - Vec2d(3. * dx, 3. * dx)).array().floor().matrix().cast<int>();
		bbox.extend(minIndex);

		Vec2i maxIndex = surface.worldToIndex(particlePoint + Vec2d(3. * dx, 3. * dx)).array().floor().matrix().cast<int>();
		bbox.extend(maxIndex);

		AlignedBox2i clampBox;
		clampBox.extend(Vec2i::Zero());
		clampBox.extend(surface.size() - Vec2i::Ones());

		bbox.clamp(clampBox);

		for (int i = bbox.min()[0]; i <= bbox.max()[0]; ++i)
			for (int j = bbox.min()[1]; j <= bbox.max()[1]; ++j)
			{
				if (i < 0 || j < 0 || i >= surface.size()[0]
					|| j >= surface.size()[1]) continue;

				Vec2d gridPoint = surface.indexToWorld(Vec2d(i, j));

				surface(i, j) = std::min((gridPoint - particlePoint).norm() - myParticleRadius, surface(i, j));
			}
	}

	surface.reinit();

	return surface;
}

void FluidParticles::reseed(const LevelSet& surface, double minDensity, double maxDensity, const VectorGrid<double>* velocity, double seed)
{
	myNewParticles.clear();

	// Load up particles into grid cells
	UniformGrid<std::vector<int>> particleIndexGrid(surface.size());

	VecVec2d addParticlesList;
	std::vector<int> deleteParticlesList;

	for (int particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		Vec2d particlePoint = surface.worldToIndex(myParticles[particleIndex]);
		Vec2i gridPoint = particlePoint.array().round().cast<int>();

		if (gridPoint[0] < 0 || gridPoint[1] < 0
			|| gridPoint[0] >= surface.size()[0]
			|| gridPoint[1] >= surface.size()[1])
		{
			deleteParticlesList.push_back(particleIndex);
			continue;
		}

		// Add index to particle grid to reference later 
		particleIndexGrid(gridPoint).push_back(particleIndex);
	}

	maxDensity = std::max(maxDensity, myOversampleRate);
	minDensity = std::min(maxDensity, myOversampleRate);

	forEachVoxelRange(Vec2i::Zero(), surface.size(), [&](const Vec2i& cell)
	{
		// If there are more particles in a cell than the max value, delete down to the target density
		if (particleIndexGrid(cell).size() > maxDensity)
		{
			// Only reseed near/in the surface
			int targetParticleCount = (surface(cell) > -2. * surface.dx()) ? int(myParticleDensity * myOversampleRate) : myParticleDensity;

			// Delete particles until we're down to the right amount
			for (int deleteCount = int(particleIndexGrid(cell).size()); deleteCount > targetParticleCount; --deleteCount)
				deleteParticlesList.push_back(particleIndexGrid(cell)[deleteCount - 1]);
		}

		// If there are too few particles in a cell than the max value, seed up to the target.
		if (surface(cell) < 2. * surface.dx() && particleIndexGrid(cell).size() < minDensity)
		{
			// Only reseed near/in the surface
			int targetParticleCount = (surface(cell) > -2 * surface.dx()) ? int(myParticleDensity * myOversampleRate) : myParticleDensity;

			// Add particles until we're up to the right amount
			for (int addCount = int(particleIndexGrid(cell).size()); addCount < targetParticleCount; ++addCount)
			{
				// TODO: build a single random generator
				Vec2d newPoint = cell.cast<double>() + randomizer(cell, addCount, seed);
				Vec2d worldPoint = surface.indexToWorld(newPoint);

				if (surface.biLerp(worldPoint) <= -myParticleRadius)
					addParticlesList.push_back(worldPoint);
			}
		}
	});

	// Reverse sort the parts to be deleted so we don't accidentally swap and delete the wrong particles
	std::sort(deleteParticlesList.begin(), deleteParticlesList.end(), std::greater<int>());

	for (auto deleteIndex : deleteParticlesList)
	{
		int particleCount = int(myParticles.size());
		std::swap(myParticles[deleteIndex], myParticles[particleCount - 1]);

		if (myTrackVelocity) std::swap(myVelocity[deleteIndex], myVelocity[particleCount - 1]);

		myParticles.resize(particleCount - 1);
	}

	if (myTrackVelocity) myVelocity.resize(myParticles.size());

	myParticles.insert(myParticles.end(), addParticlesList.begin(), addParticlesList.end());

	// Sample velocity field if tracked
	if (myTrackVelocity)
	{
		if (velocity != nullptr)
		{
			for (unsigned addParticleIndex = 0; addParticleIndex < addParticlesList.size(); ++addParticleIndex)
			{
				Vec2d particleVelocity = velocity->biLerp(addParticlesList[addParticleIndex]);
				myVelocity.push_back(particleVelocity);
			}
		}
		else
		{
			for (unsigned addParticleIndex = 0; addParticleIndex < addParticlesList.size(); ++addParticleIndex)
				myVelocity.push_back(Vec2d::Zero());
		}
		assert(myVelocity.size() == myParticles.size());
	}

	myNewParticles = addParticlesList;
}

void FluidParticles::bumpParticles(const LevelSet& solidSurface)
{
	for (int particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		if (solidSurface.biLerp(myParticles[particleIndex]) <= 0.)
			myParticles[particleIndex] -= .9 * solidSurface.biLerp(myParticles[particleIndex]) * solidSurface.normal(myParticles[particleIndex]);

	}
}

void FluidParticles::advect(double dt, const VectorGrid<double>& vel, const IntegrationOrder order)
{
	auto velFunc = [vel](double, const Vec2d& world_pos) -> Vec2d
	{
		return vel.biLerp(world_pos);
	};

	assert(dt >= 0);
	for (Vec2d& point : myParticles)
		point = Integrator(dt, point, velFunc, order);
}

}
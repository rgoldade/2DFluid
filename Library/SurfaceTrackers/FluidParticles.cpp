#include "FluidParticles.h"

#include <random>

namespace FluidSim2D::SurfaceTrackers
{

Vec2f randomizer(const Vec2i& coord, unsigned count, float seed)
{
	int pos0 = (5915587277 * coord[0]) ^ (3367900313 * count) ^ int(3267000013. * seed);
	int pos1 = (2860486313 * coord[1]) ^ (9576890767 * count) ^ int(5463458053. * seed);

	pos0 = abs(pos0 % 100);
	pos1 = abs(pos1 % 100);

	return Vec2f(float(pos0) / 100. - .5, float(pos1) / 100. - .5);
};

void FluidParticles::drawPoints(Renderer& renderer, const Vec3f& colour, float pointSize) const
{
	renderer.addPoints(myParticles, colour, pointSize);
}

void FluidParticles::drawVelocity(Renderer& renderer, const Vec3f& colour, float length) const
{
	assert(myTrackVelocity);

	std::vector<Vec2f> startPoints;
	std::vector<Vec2f> endPoints;

	unsigned particleCount = myParticles.size();
	for (unsigned particleIndex = 0; particleIndex < particleCount; ++particleIndex)
	{
		startPoints.push_back(myParticles[particleIndex]);
		endPoints.push_back(myParticles[particleIndex] + myVelocity[particleIndex] * length);
	}

	renderer.addLines(startPoints, endPoints, colour);
}

void FluidParticles::init(const LevelSet& surface)
{
	myParticles.clear();

	forEachVoxelRange(Vec2i(0), surface.size(), [&](const Vec2i& cell)
	{
		if (surface(cell) < surface.dx())
		{
			float sampleCount = (surface(cell) > -2. * surface.dx() - myParticleRadius) ? myParticleDensity * myOversampleRate : myParticleDensity;

			unsigned seedCount = 0;

			for (unsigned seedCount = 0; seedCount < sampleCount; ++seedCount)
			{
				Vec2f randOffset = randomizer(cell, seedCount, 15);
				Vec2f indexPoint = Vec2f(cell) + randOffset;
				Vec2f worldPoint = surface.indexToWorld(indexPoint);

				if (surface.biLerp(worldPoint) <= -myParticleRadius)
					myParticles.push_back(worldPoint);
			}
		}
	});

	myVelocity.resize(myParticles.size(), Vec2f(0));
}

void FluidParticles::setVelocity(const VectorGrid<float>& vel)
{
	assert(myTrackVelocity);
	myVelocity.resize(myParticles.size());

	for (int pointIndex = 0; pointIndex < myParticles.size(); ++pointIndex)
		myVelocity[pointIndex] = vel.biLerp(myParticles[pointIndex]);
}

void FluidParticles::applyVelocity(VectorGrid<float>& velocity)
{
	assert(myTrackVelocity);
	for (int axis : {0, 1})
	{
		UniformGrid<float> denominator(velocity.size(axis), 0);
		UniformGrid<float> numerator(velocity.size(axis), 0);

		const int particleCount = myParticles.size();
		for (int particleIndex = 0; particleIndex < particleCount; ++particleIndex)
		{
			// Iterate over nearby voxels
			Vec2f minBoundingBox = floor(velocity.worldToIndex(myParticles[particleIndex], axis));
			Vec2f maxBoundingBox = ceil(velocity.worldToIndex(myParticles[particleIndex], axis));

			maxUnion(minBoundingBox, Vec2f(0));
			minUnion(maxBoundingBox, Vec2f(velocity.size(axis)) - Vec2f(1));

			for (int i = minBoundingBox[0]; i <= maxBoundingBox[0]; ++i)
				for (int j = minBoundingBox[1]; j <= maxBoundingBox[1]; ++j)
				{
					if (i < 0 || j < 0 || i >= velocity.size(axis)[0]
						|| j >= velocity.size(axis)[1]) continue;

					Vec2f gridPoint = velocity.indexToWorld(Vec2f(i, j), axis);

					if (dist2(gridPoint, myParticles[particleIndex]) <= sqr(velocity.dx()))
					{
						float k = 1. - dist(myParticles[particleIndex], gridPoint) / velocity.dx();
						numerator(i, j) += k * myVelocity[particleIndex][axis];
						denominator(i, j) += k;
					}
				}
		}

		forEachVoxelRange(Vec2i(0), velocity.size(axis), [&](const Vec2i& cell)
		{
			if (denominator(cell) > 0.)
			{
				velocity(cell, axis) = numerator(cell) / denominator(cell);
			}
		});
	}
}
void FluidParticles::incrementVelocity(VectorGrid<float>& velocity)
{
	assert(myVelocity.size() == myParticles.size() && myTrackVelocity);

	const int particleCount = myParticles.size();

	for (int particleIndex = 0; particleIndex < particleCount; ++particleIndex)
		myVelocity[particleIndex] += velocity.biLerp(myParticles[particleIndex]);
}

void FluidParticles::blendVelocity(const VectorGrid<float>& oldVelocity,
	const VectorGrid<float>& newVelocity,
	float blend)
{
	assert(myVelocity.size() == myParticles.size() && myTrackVelocity);

	for (int particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		Vec2f particleVelocity = myVelocity[particleIndex];
		Vec2f picVelocity = newVelocity.biLerp(myParticles[particleIndex]);
		Vec2f flipVelocity = picVelocity - oldVelocity.biLerp(myParticles[particleIndex]);

		myVelocity[particleIndex] = (1. - blend) * picVelocity + (blend) * (particleVelocity + flipVelocity);
	}
}

LevelSet FluidParticles::surfaceParticles(const Transform& xform, const Vec2i& size, int narrowBand) const
{
	LevelSet surface(xform, size, narrowBand);
	float dx = xform.dx();

	for (auto particlePoint : myParticles)
	{
		// Iterate over nearby voxels
		Vec2f minBoundingBox = floor(surface.worldToIndex(particlePoint - Vec2f(3. * dx)));
		Vec2f maxBoundingBox = ceil(surface.worldToIndex(particlePoint + Vec2f(3. * dx)));

		maxUnion(minBoundingBox, Vec2f(0));
		minUnion(maxBoundingBox, Vec2f(surface.size()) - Vec2f(1));

		for (int i = minBoundingBox[0]; i <= maxBoundingBox[0]; ++i)
			for (int j = minBoundingBox[1]; j <= maxBoundingBox[1]; ++j)
			{
				if (i < 0 || j < 0 || i >= surface.size()[0]
					|| j >= surface.size()[1]) continue;

				Vec2f gridPoint = surface.indexToWorld(Vec2f(i, j));

				surface(i, j) = std::min(dist(gridPoint, particlePoint) - myParticleRadius, surface(i, j));
			}
	}

	surface.reinit();

	return surface;
}

void FluidParticles::reseed(const LevelSet& surface, float minDensity, float maxDensity, const VectorGrid<float>* velocity, float seed)
{
	myNewParticles.clear();

	// Load up particles into grid cells
	UniformGrid<std::vector<int>> particleIndexGrid(surface.size());

	std::vector<Vec2f> addParticlesList;
	std::vector<int> deleteParticlesList;

	for (int particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		Vec2f particlePoint = surface.worldToIndex(myParticles[particleIndex]);
		Vec2f gridPoint = round(particlePoint);

		if (gridPoint[0] < 0 || gridPoint[1] < 0
			|| gridPoint[0] >= surface.size()[0]
			|| gridPoint[1] >= surface.size()[1])
		{
			deleteParticlesList.push_back(particleIndex);
			continue;
		}

		// Add index to particle grid to reference later 
		particleIndexGrid(gridPoint[0], gridPoint[1]).push_back(particleIndex);
	}

	maxDensity = std::max(maxDensity, myOversampleRate);
	minDensity = std::min(maxDensity, myOversampleRate);
	forEachVoxelRange(Vec2i(0), surface.size(), [&](const Vec2i& cell)
	{
		// If there are more particles in a cell than the max value, delete down to the target density
		if (particleIndexGrid(cell).size() > maxDensity)
		{
			// Only reseed near/in the surface
			int targetParticleCount = (surface(cell) > -2 * surface.dx()) ? myParticleDensity * myOversampleRate : myParticleDensity;

			// Delete particles until we're down to the right amount
			for (int deleteCount = particleIndexGrid(cell).size(); deleteCount > targetParticleCount; --deleteCount)
				deleteParticlesList.push_back(particleIndexGrid(cell)[deleteCount - 1]);
		}

		// If there are too few particles in a cell than the max value, seed up to the target.
		if (surface(cell) < 2. * surface.dx() && particleIndexGrid(cell).size() < minDensity)
		{
			// Only reseed near/in the surface
			unsigned targetParticleCount = (surface(cell) > -2 * surface.dx()) ? myParticleDensity * myOversampleRate : myParticleDensity;

			// Add particles until we're up to the right amount
			for (unsigned addCount = particleIndexGrid(cell).size(); addCount < targetParticleCount; ++addCount)
			{
				// TODO: build a single random generator
				Vec2f newPoint = Vec2f(cell) + randomizer(cell, addCount, seed);
				Vec2f worldPoint = surface.indexToWorld(newPoint);

				if (surface.biLerp(worldPoint) <= -myParticleRadius)
					addParticlesList.push_back(worldPoint);
			}
		}
	});

	// Reverse sort the parts to be deleted so we don't accidentally swap and delete the wrong particles
	std::sort(deleteParticlesList.begin(), deleteParticlesList.end(), std::greater<int>());

	for (auto deleteIndex : deleteParticlesList)
	{
		int particleCount = myParticles.size();
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
				Vec2f particleVelocity = velocity->biLerp(addParticlesList[addParticleIndex]);
				myVelocity.push_back(particleVelocity);
			}
		}
		else
		{
			for (unsigned addParticleIndex = 0; addParticleIndex < addParticlesList.size(); ++addParticleIndex)
				myVelocity.push_back(Vec2f(0));
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

void FluidParticles::advect(float dt, const VectorGrid<float>& vel, const IntegrationOrder order)
{
	auto velFunc = [vel](float, const Vec2f& world_pos)
	{
		return vel.biLerp(world_pos);
	};

	assert(dt >= 0);
	for (auto& point : myParticles)
		point = Integrator(dt, point, velFunc, order);
}

}
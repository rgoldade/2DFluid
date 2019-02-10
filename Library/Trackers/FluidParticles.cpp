#include <random>

#include "FluidParticles.h"

static Vec2R randomizer(const Vec2ui& coord, unsigned count, Real seed)
{
	int pos0 = (5915587277 * coord[0]) ^ (3367900313 * count) ^ int(3267000013. * seed);
	int pos1 = (2860486313 * coord[1]) ^ (9576890767 * count) ^ int(5463458053. * seed);

	pos0 = abs(pos0 % 100);
	pos1 = abs(pos1 % 100);

	return Vec2R(Real(pos0) / 100. - .5, Real(pos1) / 100. - .5);
};

void FluidParticles::drawPoints(Renderer& renderer, const Vec3f& colour, unsigned size) const
{
	renderer.addPoints(myParticles, colour, size);
}

void FluidParticles::drawVelocity(Renderer& renderer, const Vec3f& colour, Real length) const
{
	assert(myTrackVelocity);

	std::vector<Vec2R> startPoints;
	std::vector<Vec2R> endPoints;

	unsigned particleCount = myParticles.size();
	for (unsigned particleIndex = 0; particleIndex < particleCount; ++particleIndex)
	{
		startPoints.push_back(myParticles[particleIndex]);
		endPoints.push_back(myParticles[particleIndex] + myVelocity[particleIndex] * length);
	}
	
	renderer.addLines(startPoints, endPoints, colour);
}

void FluidParticles::init(const LevelSet2D& surface)
{
	myParticles.clear();

	forEachVoxelRange(Vec2ui(0), surface.size(), [&](const Vec2ui& cell)
	{
		if (surface(cell) < surface.dx())
		{
			Real sampleCount = (surface(cell) > -2. * surface.dx() - myParticleRadius) ? myParticleDensity * myOversampleRate : myParticleDensity;

			unsigned seedCount = 0;

			for (unsigned seedCount = 0; seedCount < sampleCount; ++seedCount)
			{
				Vec2R randOffset = randomizer(Vec2ui(cell), seedCount, 15);
				Vec2R indexPoint = Vec2R(cell) + randOffset;
				Vec2R worldPoint = surface.indexToWorld(indexPoint);

				Real tempphi = surface.interp(worldPoint);
				if (surface.interp(worldPoint) <= -myParticleRadius)
					myParticles.push_back(worldPoint);
			}
		}
	});

	myVelocity.resize(myParticles.size(), Vec2R(0));
}

void FluidParticles::setVelocity(const VectorGrid<Real>& vel)
{
	assert(myTrackVelocity);
	myVelocity.resize(myParticles.size());

	for (unsigned p = 0; p < myParticles.size(); ++p)
		myVelocity[p] = vel.interp(myParticles[p]);
}

void FluidParticles::applyVelocity(VectorGrid<Real>& velocity)
{
	assert(myTrackVelocity);
	for (unsigned axis : {0, 1})
	{
		UniformGrid<Real> denominator(velocity.size(axis), 0);
		UniformGrid<Real> numerator(velocity.size(axis), 0);

		unsigned particleCount = myParticles.size();
		for (unsigned particleIndex = 0; particleIndex < particleCount; ++particleIndex)
		{
			// Iterate over nearby voxels
			Vec2R minBoundingBox = floor(velocity.worldToIndex(myParticles[particleIndex], axis));
			Vec2R maxBoundingBox = ceil(velocity.worldToIndex(myParticles[particleIndex], axis));

			maxUnion(minBoundingBox, Vec2R(0));
			minUnion(maxBoundingBox, Vec2R(velocity.size(axis)) - Vec2R(1));

			for (int i = minBoundingBox[0]; i <= maxBoundingBox[0]; ++i)
				for (int j = minBoundingBox[1]; j <= maxBoundingBox[1]; ++j)
				{
					if (i < 0 || j < 0 || i >= velocity.size(axis)[0]
						|| j >= velocity.size(axis)[1]) continue;

					Vec2R gridPoint = velocity.indexToWorld(Vec2R(i, j), axis);

					if (dist2(gridPoint, myParticles[particleIndex]) <= Util::sqr(velocity.dx()))
					{
						Real k = 1. - dist(myParticles[particleIndex], gridPoint)/ velocity.dx();
						numerator(i, j) += k * myVelocity[particleIndex][axis];
						denominator(i, j) += k;
					}
				}
		}

		forEachVoxelRange(Vec2ui(0), velocity.size(axis), [&](const Vec2ui& cell)
		{
			if (denominator(cell) > 0.)
			{
				velocity(cell, axis) = numerator(cell) / denominator(cell);
			}
		});
	}
}
void FluidParticles::incrementVelocity(VectorGrid<Real>& velocity)
{
	assert(myVelocity.size() == myParticles.size() && myTrackVelocity);

	unsigned particleCount = myParticles.size();

	for (unsigned particleIndex = 0; particleIndex < particleCount; ++particleIndex)
		myVelocity[particleIndex] += velocity.interp(myParticles[particleIndex]);
}

void FluidParticles::blendVelocity(const VectorGrid<Real>& oldVelocity,
										const VectorGrid<Real>& newVelocity,
										Real blend)
{
	assert(myVelocity.size() == myParticles.size() && myTrackVelocity);

	for (unsigned particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		Vec2R particleVelocity = myVelocity[particleIndex];
		Vec2R picVelocity = newVelocity.interp(myParticles[particleIndex]);
		Vec2R flipVelocity = picVelocity - oldVelocity.interp(myParticles[particleIndex]);

		myVelocity[particleIndex] = (1. - blend) * picVelocity + (blend) * (particleVelocity + flipVelocity);
	}
}

LevelSet2D FluidParticles::surfaceParticles(const Transform& xform, const Vec2ui& size, unsigned narrowBand) const
{
	LevelSet2D surface(xform, size, narrowBand);
	Real dx = xform.dx();

	for (auto particlePoint : myParticles)
	{
		// Iterate over nearby voxels
		Vec2R minBoundingBox = floor(surface.worldToIndex(particlePoint - Vec2R(3. * dx)));
		Vec2R maxBoundingBox = ceil(surface.worldToIndex(particlePoint + Vec2R(3. * dx)));

		maxUnion(minBoundingBox, Vec2R(0));
		minUnion(maxBoundingBox, Vec2R(surface.size()) - Vec2R(1));

		for (int i = minBoundingBox[0]; i <= maxBoundingBox[0]; ++i)
			for (int j = minBoundingBox[1]; j <= maxBoundingBox[1]; ++j)
			{
				if (i < 0 || j < 0 || i >= surface.size()[0]
					|| j >= surface.size()[1]) continue;

				Vec2R gridPoint = surface.indexToWorld(Vec2R(i, j));
				
				surface(i, j) = std::min(dist(gridPoint, particlePoint) - myParticleRadius, surface(i, j));
			}
	}

	surface.reinit();

	return surface;
}

void FluidParticles::reseed(const LevelSet2D& surface, Real minDensity, Real maxDensity, const VectorGrid<Real>* velocity, Real seed)
{
	myNewParticles.clear();

	// Load up particles into grid cells
	UniformGrid<std::vector<size_t>> particleIndexGrid(surface.size());

	std::vector<Vec2R> addParticlesList;
	std::vector<unsigned> deleteParticlesList;

	for (unsigned particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		Vec2R particlePoint = surface.worldToIndex(myParticles[particleIndex]);
		Vec2R gridPoint = round(particlePoint);
		
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
	forEachVoxelRange(Vec2ui(0), surface.size(), [&](const Vec2ui& cell)
	{
		// If there are more particles in a cell than the max value, delete down to the target density
		if (particleIndexGrid(cell).size() > maxDensity)
		{
			// Only reseed near/in the surface
			unsigned targetParticleCount = (surface(cell) > -2 * surface.dx()) ? myParticleDensity * myOversampleRate : myParticleDensity;

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
				Vec2R newPoint = Vec2R(cell) + randomizer(cell, addCount, seed);
				Vec2R worldPoint = surface.indexToWorld(newPoint);

				if (surface.interp(worldPoint) <= -myParticleRadius) addParticlesList.push_back(worldPoint);
			}
		}
	});

	// Reverse sort the parts to be deleted so we don't accidentally swap and delete the wrong particles
	std::sort(deleteParticlesList.begin(), deleteParticlesList.end(), std::greater<unsigned>());

	for (auto deleteIndex : deleteParticlesList)
	{
		unsigned particleCount = myParticles.size();
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
				Vec2R particleVelocity = velocity->interp(addParticlesList[addParticleIndex]);
				myVelocity.push_back(particleVelocity);
			}
		}
		else
		{
			for (unsigned addParticleIndex = 0; addParticleIndex < addParticlesList.size(); ++addParticleIndex)
				myVelocity.push_back(Vec2R(0));
		}
		assert(myVelocity.size() == myParticles.size());
	}

	myNewParticles = addParticlesList;
}

void FluidParticles::bumpParticles(const LevelSet2D& collision)
{
	for (unsigned particleIndex = 0; particleIndex < myParticles.size(); ++particleIndex)
	{
		if (collision.interp(myParticles[particleIndex]) <= 0.)
			myParticles[particleIndex] -= .9 *collision.interp(myParticles[particleIndex]) * collision.normal(myParticles[particleIndex]);

	}
}

void FluidParticles::advect(Real dt, const VectorGrid<Real>& vel, const IntegrationOrder order)
{
	auto velFunc = [vel](Real, const Vec2R& world_pos)
	{
		return vel.interp(world_pos);
	};

	assert(dt >= 0);
	for (auto& point : myParticles)
		point = Integrator(dt, point, velFunc, order);
}
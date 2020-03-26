#ifndef LIBRARY_FIELD_ADVECTOR_H
#define LIBRARY_FIELD_ADVECTOR_H

#include "tbb/tbb.h"

#include "Integrator.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "Vec.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// FieldAdvector.h/cpp
// Ryan Goldade 2017
//
// A versatile advection class to handle
// forward advection and semi-Lagrangian
// backtracing.
//
////////////////////////////////////

namespace FluidSim2D::SimTools
{

using namespace Utilities;

enum class InterpolationOrder { LINEAR, CUBIC };

template<typename Field, typename VelocityField>
void advectField(float dt, Field& destinationField, const Field& sourceField, const VelocityField& velocity, IntegrationOrder order, InterpolationOrder interpOrder)
{
	assert(&destinationField != &sourceField);

	tbb::parallel_for(tbb::blocked_range<int>(0, sourceField.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec2i cell = sourceField.unflatten(cellIndex);

			Vec2f worldPoint = sourceField.indexToWorld(Vec2f(cell));
			worldPoint = Integrator(-dt, worldPoint, velocity, order);

			switch (interpOrder)
			{
			case InterpolationOrder::LINEAR:
				destinationField(cell) = sourceField.biLerp(worldPoint);
				break;
			case InterpolationOrder::CUBIC:
				destinationField(cell) = sourceField.biCubicInterp(worldPoint, false, true);
				break;
			default:
				assert(false);
				break;
			}
		}
	});
}

}

#endif
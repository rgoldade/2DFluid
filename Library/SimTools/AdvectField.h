#ifndef LIBRARY_ADVECTFIELD_H
#define LIBRARY_ADVECTFIELD_H

#include "Common.h"
#include "Integrator.h"
#include "ScalarGrid.h"
#include "Vec.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// AdvectField.h/cpp
// Ryan Goldade 2017
//
// A versatile advection class to handle
// forward advection and semi-Lagrangian
// backtracing.
//
////////////////////////////////////

enum class InterpolationOrder { LINEAR, CUBIC };

template<typename Field>
class AdvectField
{
public:

	AdvectField(const Field& source)
		: myField(source)
		{}

	template<typename VelocityField>
	void advectField(Real dt, Field& field, const VelocityField& velocity, const IntegrationOrder order, const InterpolationOrder interpOrder = InterpolationOrder::LINEAR);

private:
	const Field &myField;
};

template<typename Field>
template<typename VelocityField>
void AdvectField<Field>::advectField(Real dt, Field& field, const VelocityField& velocity, const IntegrationOrder order, const InterpolationOrder interpOrder)
{
	assert(&field != &myField);

	forEachVoxelRange(Vec2i(0), field.size(), [&](const Vec2i& cell)
	{
		Vec2R worldPoint = field.indexToWorld(Vec2R(cell));
		worldPoint = Integrator(-dt, worldPoint, velocity, order);

		switch (interpOrder)
		{
		case InterpolationOrder::LINEAR:
			field(cell) = myField.interp(worldPoint);
			break;
		case InterpolationOrder::CUBIC:
			field(cell) = myField.cubicInterp(worldPoint, false, true);
			break;
		default:
			assert(false);
			break;
		}
	});
}

#endif
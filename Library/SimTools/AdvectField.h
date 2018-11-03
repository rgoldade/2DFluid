#pragma once

#include "VectorGrid.h"
#include "ScalarGrid.h"
#include "Integrator.h"

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

class LevelSet2D;

enum class InterpolationOrder { LINEAR, CUBIC };

template<typename Field>
class AdvectField
{
public:

	AdvectField(const Field& source)
		: m_field(source)
		{}

	template<typename VelocityField>
	void advect_field(Real dt, Field& field, const VelocityField& vel, const IntegrationOrder order, const InterpolationOrder interp_order = InterpolationOrder::LINEAR);

private:
	const Field &m_field;
};

template<typename Field>
template<typename VelocityField>
void AdvectField<Field>::advect_field(Real dt, Field& field, const VelocityField& vel, const IntegrationOrder order, const InterpolationOrder interp_order)
{
	assert(&field != &m_field);

	for_each_voxel_range(Vec2ui(0), field.size(), [&](const Vec2ui& cell)
	{
		Vec2R pos = field.idx_to_ws(Vec2R(cell));
		pos = Integrator(-dt, pos, vel, order);

		switch (interp_order)
		{
		case InterpolationOrder::LINEAR:
			field(cell) = m_field.interp(pos);
			break;
		case InterpolationOrder::CUBIC:
			field(cell) = m_field.cubic_interp(pos, false, true);
			break;
		default:
			assert(false);
			break;
		}
	});
}
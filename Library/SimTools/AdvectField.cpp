#include "AdvectField.h"

// Warning: if you're backtrace advecting the velocity field, make sure "field" is a temporary copy so you
// don't pollute your velocity field whilst advecting.
template<>
void AdvectField<VectorGrid<Real>>::advect_field(Real dt, VectorGrid<Real>& field, IntegratorSettings::Integrator order, IntegratorSettings::Interpolator interp)
{
	AdvectField<ScalarGrid<Real>> u_field(m_vel, m_field.grid(0));
	AdvectField<ScalarGrid<Real>> v_field(m_vel, m_field.grid(1));

	switch (order)
	{
	case IntegratorSettings::FE:
		u_field.advect_field(dt, field.grid(0), Integrator::forward_euler<Vec2R, AdvectField<ScalarGrid<Real>>>());
		v_field.advect_field(dt, field.grid(1), Integrator::forward_euler<Vec2R, AdvectField<ScalarGrid<Real>>>());
		break;
	case IntegratorSettings::RK3:
		u_field.advect_field(dt, field.grid(0), Integrator::rk3<Vec2R, AdvectField<ScalarGrid<Real>>>());
		v_field.advect_field(dt, field.grid(1), Integrator::rk3<Vec2R, AdvectField<ScalarGrid<Real>>>());
		break;
	default:
		assert(false);
	}
}
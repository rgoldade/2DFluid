#pragma once

#include "VectorGrid.h"
#include "ScalarGrid.h"
#include "LevelSet2D.h"
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

namespace IntegratorSettings
{
	enum Integrator { FE, RK3 };
}

template<typename Field>
class AdvectField
{
public:
	AdvectField(VectorGrid<Real>& vel, const Field& source)
		: m_vel(vel), m_field(source), m_boundary(NULL)
		{}

	void set_collision_volumes(const LevelSet2D& boundary)
	{
		m_boundary = &boundary;
	}

	void advect_field(Real dt, Field& field, IntegratorSettings::Integrator order);

	template<typename Integrator>
	void advect_field(Real dt, Field& field, const Integrator& f);

	// Operator overload to sample the velocity field while taking into account
	// solid objects
	Vec2R operator()(Real dt, const Vec2R& wpos) const;

private:
	VectorGrid<Real>& m_vel;
	
	const Field &m_field;
	// The solid volume is used to *bump* back to the surface
	const LevelSet2D* m_boundary;
};

template<typename Field>
void AdvectField<Field>::advect_field(Real dt, Field& field, IntegratorSettings::Integrator order)
{
	switch (order)
	{
	case IntegratorSettings::FE:
		advect_field(dt, field, Integrator::forward_euler<Vec2R, AdvectField<Field>>());
		break;
	case IntegratorSettings::RK3:
		advect_field(dt, field, Integrator::rk3<Vec2R, AdvectField<Field>>());
		break;
	default:
		assert(false);
	}
}

template<>
void AdvectField<VectorGrid<Real>>::advect_field(Real dt, VectorGrid<Real>& field, IntegratorSettings::Integrator order);

// Advect the source field (m_field) into the destination field. The destination field
// should never be the same field as m_field since it would corrupt the information mid-loop
template<typename Field>
template<typename Integrator>
void AdvectField<Field>::advect_field(Real dt, Field& dest_field, const Integrator& f)
{
	size_t x_size = dest_field.size()[0];
	size_t y_size = dest_field.size()[1];

	for (size_t x = 0; x < x_size; ++x)
		for (size_t y = 0; y < y_size; ++y)
		{
			Vec2R pos = dest_field.idx_to_ws(Vec2R(x, y));
			pos = f(pos, -dt, *this);
			dest_field(x, y) = m_field.interp(pos);
		}
}

template<typename Field>
Vec2R AdvectField<Field>::operator()(Real dt, const Vec2R& wpos) const
{
	// Bump to boundary surface if there is a boundary field set
	//if (m_boundary && m_boundary->interp(wpos) <= 0.)
	//{
	//	return - m_boundary->interp(wpos) * m_boundary->normal_const(wpos) / dt;
	//}
	return m_vel.interp(wpos);
}

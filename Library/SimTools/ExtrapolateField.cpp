#include "ExtrapolateField.h"

template<>
void ExtrapolateField<VectorGrid<Real>>::extrapolate(const VectorGrid<Real>& mask, size_t bandwidth)
{
	ExtrapolateField<ScalarGrid<Real>> x_field(m_field.grid(0));
	x_field.extrapolate(mask.grid(0), bandwidth);

	ExtrapolateField<ScalarGrid<Real>> y_field(m_field.grid(1));
	y_field.extrapolate(mask.grid(1), bandwidth);
}
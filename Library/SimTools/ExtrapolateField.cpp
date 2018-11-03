#include "ExtrapolateField.h"

template<>
void ExtrapolateField<VectorGrid<Real>>::extrapolate(const VectorGrid<Real>& mask, unsigned bandwidth)
{
	for (unsigned axis = 0; axis < 2; ++axis)
	{
		ExtrapolateField<ScalarGrid<Real>> field(m_field.grid(axis));
		field.extrapolate(mask.grid(axis), bandwidth);
	}
}
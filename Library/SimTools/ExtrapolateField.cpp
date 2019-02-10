#include "ExtrapolateField.h"

template<>
void ExtrapolateField<VectorGrid<Real>>::extrapolate(const VectorGrid<Real>& mask, unsigned bandwidth)
{
	for (unsigned axis : {0, 1})
	{
		ExtrapolateField<ScalarGrid<Real>> field(myField.grid(axis));
		field.extrapolate(mask.grid(axis), bandwidth);
	}
}
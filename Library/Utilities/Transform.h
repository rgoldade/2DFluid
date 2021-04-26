#ifndef FLUIDSIM2D_TRANSFORM_H
#define FLUIDSIM2D_TRANSFORM_H

#include "Utilities.h"

///////////////////////////////////
//
// Transform.h
// Ryan Goldade 2017
//
// Simple transform container to simplify
// implementation for various transforms
// between grids, etc.
//
////////////////////////////////////

namespace FluidSim2D
{

class Transform
{
public:
	Transform(double dx = 1, const Vec2d& offset = Vec2d::Zero())
		: myDx(dx)
		, myOffset(offset)
	{}

	FORCE_INLINE Vec2d indexToWorld(const Vec2d& indexPoint) const
	{
		return indexPoint * myDx + myOffset;
	}

	FORCE_INLINE Vec2d worldToIndex(const Vec2d& worldPoint) const
	{
		return (worldPoint - myOffset) / myDx;
	}

	FORCE_INLINE double dx() const { return myDx; }
	FORCE_INLINE Vec2d offset() const { return myOffset; }

	FORCE_INLINE bool operator==(const Transform& rhs) const
	{
		if (myDx != rhs.myDx) return false;
		if (myOffset != rhs.myOffset) return false;
		return true;
	}

	FORCE_INLINE bool operator!=(const Transform& rhs) const
	{
		return !(*this == rhs);
	}

private:
	double myDx;
	Vec2d myOffset;
};

}
#endif
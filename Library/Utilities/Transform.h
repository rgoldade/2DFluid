#ifndef LIBRARY_TRANSFORM_H
#define LIBRARY_TRANSFORM_H

#include "Utilities.h"
#include "Vec.h"

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

namespace FluidSim2D::Utilities
{

class Transform
{
public:
	Transform(float dx = 1., const Vec2f& offset = Vec2f(0))
		: myDx(dx)
		, myOffset(offset)
	{}

	Vec2f indexToWorld(const Vec2f& indexPoint) const
	{
		return indexPoint * myDx + myOffset;
	}

	Vec2f worldToIndex(const Vec2f& worldPoint) const
	{
		return (worldPoint - myOffset) / myDx;
	}

	float dx() const { return myDx; }
	Vec2f offset() const { return myOffset; }

	bool operator==(const Transform& rhs) const
	{
		if (myDx != rhs.myDx) return false;
		if (myOffset != rhs.myOffset) return false;
		return true;
	}

	bool operator!=(const Transform& rhs) const
	{
		if (myDx == rhs.myDx) return false;
		if (myOffset == rhs.myOffset) return false;
		return true;
	}

private:
	float myDx;
	Vec2f myOffset;
};

}
#endif
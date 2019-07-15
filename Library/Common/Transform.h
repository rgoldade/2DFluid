#ifndef LIBRARY_TRANSFORM_H
#define LIBRARY_TRANSFORM_H

#include "Common.h"
#include "Util.h"
#include "Vec.h"

///////////////////////////////////
//
// Transform.h
// Ryan Goldade 2017
//
// Simple transform container to simplify
// implementation for various transforms
// in the package
//
////////////////////////////////////

class Transform
{
public:
	Transform(Real dx = 1., const Vec2R& offset = Vec2R(0))
		: myDx(dx)
		, myOffset(offset)
		{}

	Vec2R indexToWorld(const Vec2R& indexPoint) const
	{
		return indexPoint * myDx + myOffset;
	}

	Vec2R worldToIndex(const Vec2R& worldPoint) const
	{
		return (worldPoint - myOffset) / myDx;
	}

	Real dx() const { return myDx; }
	Vec2R offset() const { return myOffset; }

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
	Real myDx;
	Vec2R myOffset;
};

#endif
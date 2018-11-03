#pragma once

#include "Common.h"

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
		: m_dx(dx)
		, m_offset(offset)
		{}

	Vec2R idx_to_ws(Vec2R ipos) const
	{
		return ipos * m_dx + m_offset;
	}

	Vec2R ws_to_idx(Vec2R wpos) const
	{
		return (wpos - m_offset) / m_dx;
	}

	Real dx() const { return m_dx; }
	Vec2R offset() const { return m_offset; }

	bool operator==(const Transform &rhs) const
	{
		if (m_dx != rhs.m_dx) return false;
		if (m_offset != rhs.m_offset) return false;
		return true;
	}

	bool operator!=(const Transform &rhs) const
	{
		if (m_dx == rhs.m_dx) return false;
		if (m_offset == rhs.m_offset) return false;
		return true;
	}

private:
	Real m_dx;
	Vec2R m_offset;
};

#pragma once

///////////////////////////////////
//
// HashGrid2D.h
// Ryan Goldade 2017
//
// Thin wrapper around unordered_map
// to give some specialization for a 
// sparse representation of a 2-D
// uniform grid.
//
////////////////////////////////////

#include <functional>
#include <unordered_map>

#include "Vec.h"

// Template specialization to create a custom 2-D hash function
namespace std {
	template<>
	struct hash<Vec2i>
	{
		size_t operator()(const Vec2i& k) const
		{
			return((hash<int>()(k[0]) ^ hash<int>()(k[1] << 1)) >> 1);
		}
	};
}

template <class T>
class HashGrid2D
{

public:
	HashGrid2D() {};

	HashGrid2D(size_t size)
	{
		m_grid.reserve(size);
	};

	// If hash key is unoccupied, insert value and return true.
	// Return false if already occupied.
	inline bool insert(const Vec2i& key, const T& val)
	{
		if (find(key)) 	return false;

		m_grid[key] = val;
		return true;
	}

	// If hash key is occupied, return by overwriting val.
	// Return false if already occupied.
	inline bool get(const Vec2i& key, T& val)
	{
		bool found = find(key);
		if (found)
			val = m_grid[key];
		return found;
	}

	// Get method with no safety checks
	inline T get(const Vec2i& key)
	{
		return m_grid[key];
	}

	inline T at(const Vec2i& key)
	{
		return m_grid.at(key);
	}

	// If hash key is occupied, return true.
	// Return false if unoccupied.
	inline bool find(const Vec2i& key)
	{
		return !(m_grid.find(key) == m_grid.end());
	}
	
	// If hash key is occupied, overwrite value and return true.
	// Insert value and return false if unoccupied.
	inline bool replace(const Vec2i& key, const T& val)
	{
		bool found = find(key);
		m_grid[key] = val;
		return found;
	}

	// Reset the grid by clearing and resizing the hash table.
	inline void reset(size_t size)
	{
		empty();
		m_grid.reserve(size);
	}

	inline void empty()
	{
		m_grid.clear();
	}

	inline size_t size()
	{
		return m_grid.size();
	}

	// Underlying iterator
	inline void rewind()
	{
		iter = m_grid.begin();
	}

	inline void begin()
	{
		iter = m_grid.begin();
	}

	inline void advance()
	{
		++iter;
	}

	inline bool end()
	{
		return iter == m_grid.end();
	}

	inline T operator*() const
	{
		return (*iter).second;
	}

private:
	std::unordered_map<Vec2i, T> m_grid;
	typename std::unordered_map<Vec2i, T>::iterator iter;
};

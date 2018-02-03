#pragma once

#include <vector>
#include <functional>

#include <GL/glut.h>

#include "Core.h"
#include "Vec.h"

///////////////////////////////////
//
// Renderer.h
// Ryan Goldade 2016
//
// Render machine to handle adding
// primitives from various different
// sources and having a central place
// to run the render loop.
//
////////////////////////////////////

class Renderer
{
public:
	Renderer(const char *title, Vec2i window_size, Vec2R screen_min,
				Real height, int *argc, char **argv);

	void display();
	void mouse(int button, int state, int x, int y);
	void drag(int x, int y);
	void keyboard(unsigned char key, int x, int y);
	void reshape(int w, int h);

	void set_user_mouse_click(std::function<void(int, int, int, int)> clickfunc);
	void set_user_keyboard(std::function<void(unsigned char, int, int)> keyfunc);
	void set_user_mouse_drag(std::function<void(int, int)> dragfunc);
	void set_user_display(std::function<void()> displayfunc);

	void add_point(const Vec2R& point, const Vec3f& colour = Vec3f(0,0,0), Real size = 1.);
	void add_points(const std::vector<Vec2R>& points, const Vec3f& colour = Vec3f(0, 0, 0), Real size = 1.);

	void add_line(const Vec2R& start, const Vec2R& end, const Vec3f& colour = Vec3f(0, 0, 0));
	void add_lines(const std::vector<Vec2R>& start, const std::vector<Vec2R>& end, const Vec3f& colour);
	void add_tris(const std::vector<Vec2R>& points, const std::vector<Vec3st>& faces, const Vec3f& colour);
	void add_quads(const std::vector<Vec2R>& points, const std::vector<Vec4st>& faces, const Vec3f& colour);
	
	void draw_primitives() const;

	void clear();
	void run();
private:

	std::vector<std::vector<Vec2R>> m_points;
	std::vector<Vec3f> m_point_colours;
	std::vector<Real> m_point_size;

	std::vector<std::vector<Vec2R>> m_start_lines;
	std::vector<std::vector<Vec2R>> m_end_lines;
	std::vector<Vec3f> m_line_colours;

	std::vector<std::vector<Vec2R>> m_tri_points;
	std::vector<std::vector<Vec3st>> m_tri_faces;
	std::vector<Vec3f> m_tri_colours;

	std::vector<std::vector<Vec2R>> m_quad_points;
	std::vector<std::vector<Vec4st>> m_quad_faces;
	std::vector<Vec3f> m_quad_colours;

	// width, height
	Vec2i m_wsize;

	Vec2R m_smin;
	Real m_sheight;

	Vec2R m_dmin;
	Real m_dheight;

	// Mouse specific state
	Vec2i m_mouse_pos;
	bool m_mmoved;
	enum MouseAction {INACTIVE, PAN, ZOOM_IN, ZOOM_OUT};
	MouseAction m_maction;

	// User specific extensions for each glut callback
	std::function<void(unsigned char, int, int)> m_user_keyboard;
	std::function<void(int, int, int, int)> m_user_mouse_click;
	std::function<void(int, int)> m_user_mouse_drag;
	std::function<void()> m_user_display;
};

#ifndef LIBRARY_RENDERER_H
#define LIBRARY_RENDERER_H

#include <functional>
#include <vector>

#ifdef __APPLE__
	#include <GLUT/glut.h>
#else
	#include <GL/glut.h>
#endif

#include "Common.h"
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
	Renderer(const char *title, Vec2i windowSize, Vec2R screenOrigin,
				Real screenHeight, int *argc, char **argv);

	void display();
	void mouse(int button, int state, int x, int y);
	void drag(int x, int y);
	void keyboard(unsigned char key, int x, int y);
	void reshape(int w, int h);

	void setUserMouseClick(std::function<void(int, int, int, int)> clickFunction);
	void setUserKeyboard(std::function<void(unsigned char, int, int)> keyFunction);
	void setUserMouseDrag(std::function<void(int, int)> dragFunction);
	void setUserDisplay(std::function<void()> displayFunction);

	void addPoint(const Vec2R& point, const Vec3f& colour = Vec3f(0,0,0), Real size = 1.);
	void addPoints(const std::vector<Vec2R>& points, const Vec3f& colour = Vec3f(0, 0, 0), Real size = 1.);

	void addLine(const Vec2R& start, const Vec2R& end, const Vec3f& colour = Vec3f(0, 0, 0), const Real width = 1);
	void addLines(const std::vector<Vec2R>& start, const std::vector<Vec2R>& end, const Vec3f& colour, const Real width = 1);
	void addTris(const std::vector<Vec2R>& verts, const std::vector<Vec3i>& faces, const std::vector<Vec3f>& colour);
	void addQuads(const std::vector<Vec2R>& verts, const std::vector<Vec4i>& faces, const std::vector<Vec3f>& colours);
	
	void drawPrimitives() const;

	void printImage(const std::string& filename) const;
	
	void clear();
	void run();

	Real pixelScale() const;

private:

	std::vector<std::vector<Vec2R>> myPoints;
	std::vector<Vec3f> myPointColours;
	std::vector<Real> myPointSize;

	std::vector<std::vector<Vec2R>> myLineStartingPoints;
	std::vector<std::vector<Vec2R>> myLineEndingPoints;
	std::vector<Vec3f> myLineColours;
	std::vector<Real> myLineSizes;

	std::vector<std::vector<Vec2R>> myTriVerts;
	std::vector<std::vector<Vec3i>> myTriFaces;
	std::vector<std::vector<Vec3f>> myTriColours;

	std::vector<std::vector<Vec2R>> myQuadVerts;
	std::vector<std::vector<Vec4i>> myQuadFaces;
	std::vector<std::vector<Vec3f>> myQuadColours;

	// width, height
	Vec2i myWindowSize;

	Vec2R myCurrentScreenOrigin;
	Real myCurrentScreenHeight;

	Vec2R myDefaultScreenOrigin;
	Real myDefaultScreenHeight;

	// Mouse specific state
	Vec2i myMousePosition;
	bool myMouseMoved;

	enum class MouseAction {INACTIVE, PAN, ZOOM_IN, ZOOM_OUT};
	MouseAction myMouseAction;

	// User specific extensions for each glut callback
	std::function<void(unsigned char, int, int)> myUserKeyboardFunction;
	std::function<void(int, int, int, int)> myUserMouseClickFunction;
	std::function<void(int, int)> myUserMouseDragFunction;
	std::function<void()> myUserDisplayFunction;
};

#endif
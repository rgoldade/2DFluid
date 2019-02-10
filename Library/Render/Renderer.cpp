#include "Renderer.h"

#include <fstream>

// Helper struct because glut is a pain.
// This is probably a very bad design choice
// but glut doesn't make life easy.
class GlutHelper
{
public:
	static void init(Renderer* render)
	{
		// Safety check to only ever take one instance
		// of a Renderer
		if (!myRender) myRender = render;

		glutReshapeFunc(reshape);
		glutDisplayFunc(display);
		glutMouseFunc(mouse);
		glutMotionFunc(drag);
		glutKeyboardFunc(keyboard);
	}
private:
	static void reshape(int w, int h)
	{
		if (myRender) myRender->reshape(w, h);
	}

	static void display()
	{
		if (myRender) myRender->display();
	}

	static void mouse(int button, int state, int x, int y)
	{
		if (myRender) myRender->mouse(button, state, x, y);
	}

	static void drag(int x, int y)
	{
		if (myRender) myRender->drag(x, y);
	}

	static void keyboard(unsigned char key, int x, int y)
	{
		if (myRender) myRender->keyboard(key, x, y);
	}

	static Renderer* myRender;
};

Renderer* GlutHelper::myRender;

Renderer::Renderer(const char *title, Vec2ui windowSize, Vec2R screenOrigin,
						Real screenHeight, int *argc, char **argv)
	: myWindowSize(windowSize)
	, myCurrentScreenOrigin(screenOrigin)
	, myCurrentScreenHeight(screenHeight)
	, myDefaultScreenOrigin(screenOrigin)
	, myDefaultScreenHeight(screenHeight)
	, myMouseAction(MouseAction::INACTIVE)
{
	glutInit(argc, argv);

	//TODO: review if these flags are needed
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(myWindowSize[0], myWindowSize[1]);
	glutCreateWindow(title);
	glClearColor(1.0, 1.0, 1.0, 0.0);

	GlutHelper::init(this);
}

void Renderer::setUserKeyboard(std::function<void(unsigned char, int, int)> keyFunction)
{
	myUserKeyboardFunction = keyFunction;
}

void Renderer::setUserMouseClick(std::function<void(int, int, int, int)> clickFunction)
{
	myUserMouseClickFunction = clickFunction;
} 

void Renderer::setUserMouseDrag(std::function<void(int, int)> dragFunction)
{
	myUserMouseDragFunction = dragFunction;
}

void Renderer::setUserDisplay(std::function<void()> displayFunction)
{
	myUserDisplayFunction = displayFunction;
}

void Renderer::display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: taken directly from gluvi.. might be able to clean some of this up
	glViewport(0, 0, GLsizei(myWindowSize[0]), GLsizei(myWindowSize[1]));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(myCurrentScreenOrigin[0], myCurrentScreenOrigin[0] + (myCurrentScreenHeight * Real(myWindowSize[0])) / Real(myWindowSize[1]), myCurrentScreenOrigin[1], myCurrentScreenOrigin[1] + myCurrentScreenHeight, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Run user specific display funcion
	if (myUserDisplayFunction) myUserDisplayFunction();
		
	drawPrimitives();
	glutSwapBuffers();
	glutPostRedisplay();
}

void Renderer::mouse(int button, int state, int x, int y)
{
	// If the user provided a custom mouse handler,
	// then use that
	if (myUserMouseClickFunction) myUserMouseClickFunction(button, state, x, y);
	else
	{
		if (state == GLUT_UP)
		{
			Real scale;
			switch (myMouseAction)
			{
				// PAN is only for drag
				// TODO: add click specific movement
			case MouseAction::ZOOM_IN:

				// Zoom in by 2x
				scale = .5;

				myCurrentScreenOrigin[0] -= .5 * (scale - 1.) * Real(myWindowSize[0]) * myCurrentScreenHeight / Real(myWindowSize[1]);
				myCurrentScreenOrigin[1] -= .5 * (scale - 1.) * myCurrentScreenHeight;
				myCurrentScreenHeight *= scale;

				glutPostRedisplay();
				break;

			case MouseAction::ZOOM_OUT:

				scale = 2.;
				myCurrentScreenOrigin[0] -= .5 * (scale - 1.) * Real(myWindowSize[0]) * myCurrentScreenHeight / Real(myWindowSize[1]);
				myCurrentScreenOrigin[1] -= .5 * (scale - 1.) * myCurrentScreenHeight;
				myCurrentScreenHeight *= scale;

				glutPostRedisplay();
			}
		}
		else
		{
			switch (button)
			{
			case GLUT_LEFT_BUTTON:
				myMouseAction = MouseAction::PAN;
				break;
			case GLUT_MIDDLE_BUTTON:
				myMouseAction = MouseAction::ZOOM_IN;
				break;
			case GLUT_RIGHT_BUTTON:
				myMouseAction = MouseAction::ZOOM_OUT;
			}
		}

		myMouseMoved = false; // not really used at the moment
		myMousePosition = Vec2i(x, y);
	}
}

void Renderer::drag(int x, int y)
{
	if (myUserMouseDragFunction) myUserMouseDragFunction(x, y);
	else
	{
		if (x != myMousePosition[0] || y != myMousePosition[1])
		{
			myMouseMoved = true;
			if (myMouseAction == MouseAction::PAN)
			{
				Real pixelRatio = myCurrentScreenHeight / Real(myWindowSize[1]);
				myCurrentScreenOrigin[0] -= pixelRatio * (Real(x) - myMousePosition[0]);
				myCurrentScreenOrigin[1] += pixelRatio * (Real(y) - myMousePosition[1]);
				glutPostRedisplay();
			}

			myMousePosition[0] = x;
			myMousePosition[1] = y;
		}
	}
}

void Renderer::keyboard(unsigned char key, int x, int y)
{
	if (myUserKeyboardFunction) myUserKeyboardFunction(key, x, y);

	// r triggers return to defaults
	if (key == 'r')
	{
		myCurrentScreenOrigin = myDefaultScreenOrigin;
		myCurrentScreenHeight = myDefaultScreenHeight;
		glutPostRedisplay();
	}
}

void Renderer::reshape(int w, int h)
{
	myWindowSize[0] = w;
	myWindowSize[1] = h;
	glutPostRedisplay();
}

// These helpers make it easy to render out basic primitives without having to write
// a custom loop outside of this class
void Renderer::addPoint(const Vec2R& point, const Vec3f& colour, Real size)
{
	std::vector<Vec2R> temp(1, point);
	myPoints.push_back(temp);
	myPointColours.push_back(colour);
	myPointSize.push_back(size);
}

void Renderer::addPoints(const std::vector<Vec2R>& points, const Vec3f& colour, Real size)
{
	myPoints.push_back(points);
	myPointColours.push_back(colour);
	myPointSize.push_back(size);
}

void Renderer::addLine(const Vec2R& start, const Vec2R& end, const Vec3f& colour)
{
	std::vector<Vec2R> lineStarts; lineStarts.push_back(start);
	std::vector<Vec2R> lineEnds; lineEnds.push_back(end);
	myStartLines.push_back(lineStarts);
	myEndLines.push_back(lineEnds);

	myLineColours.push_back(colour);
}

void Renderer::addLines(const std::vector<Vec2R>& start, const std::vector<Vec2R>& end, const Vec3f& colour)
{
	assert(start.size() == end.size());
	myStartLines.push_back(start);
	myEndLines.push_back(end);
	myLineColours.push_back(colour);
}

void Renderer::addTris(const std::vector<Vec2R>& verts, const std::vector<Vec3ui>& faces, const std::vector<Vec3f>& colour)
{
	myTriVerts.push_back(verts);
	myTriFaces.push_back(faces);
	myTriColours.push_back(colour);
}

void Renderer::addQuads(const std::vector<Vec2R>& verts, const std::vector<Vec4ui>& faces, const std::vector<Vec3f>& colours)
{
	myQuadVerts.push_back(verts);
	myQuadFaces.push_back(faces);
	myQuadColours.push_back(colours);
}

void Renderer::clear()
{
	myPoints.clear();
	myPointColours.clear();
	myPointSize.clear();

	myStartLines.clear();
	myEndLines.clear();
	myLineColours.clear();

	myTriVerts.clear();
	myTriFaces.clear();
	myTriColours.clear();

	myQuadVerts.clear();
	myQuadFaces.clear();
	myQuadColours.clear();
}

void Renderer::drawPrimitives() const
{
	// Render quads
	glBegin(GL_QUADS);

	unsigned quadListSize = myQuadFaces.size();
	for (unsigned quadListIndex = 0; quadListIndex < quadListSize; ++quadListIndex)
	{
		unsigned quadSublistSize = myQuadFaces[quadListIndex].size();
		for (unsigned quadIndex = 0; quadIndex < quadSublistSize; ++quadIndex)
		{
			Vec3f quadColour = myQuadColours[quadListIndex][quadIndex];
			glColor3f(quadColour[0], quadColour[1], quadColour[2]);
			
			for (unsigned quadVertexIndex = 0; quadVertexIndex < 4; ++quadVertexIndex)
			{
				unsigned meshVertexIndex = myQuadFaces[quadListIndex][quadIndex][quadVertexIndex];
				Vec2R vertex = myQuadVerts[quadListIndex][meshVertexIndex];
				glVertex2d(vertex[0], vertex[1]);
			}
		}
	}

	glEnd();

	// Render tris
	glBegin(GL_TRIANGLES);

	unsigned triListSize = myTriFaces.size();
	for (unsigned triListIndex = 0; triListIndex < triListSize; ++triListIndex)
	{
		unsigned triSublistSize = myTriFaces[triListIndex].size();
		for (unsigned triIndex = 0; triIndex < triSublistSize; ++triIndex)
		{
			Vec3f triColour = myTriColours[triListIndex][triIndex];
			glColor3f(triColour[0], triColour[1], triColour[2]);

			for (unsigned triVertexIndex = 0; triVertexIndex < 3; ++triVertexIndex)
			{
				unsigned meshVertexIndex = myTriFaces[triListIndex][triIndex][triVertexIndex];
				Vec2R vertex = myTriVerts[triListIndex][meshVertexIndex];
				glVertex2d(vertex[0], vertex[1]);
			}
		}
	}
	glEnd();

	glBegin(GL_LINES);

	unsigned lineListSize = myStartLines.size();
	for (unsigned lineListIndex = 0; lineListIndex < lineListSize; ++lineListIndex)
	{
		Vec3f lineColour = myLineColours[lineListIndex];
		glColor3f(lineColour[0], lineColour[1], lineColour[2]);

		unsigned lineSublistSize = myStartLines[lineListIndex].size();
		for (unsigned lineIndex = 0; lineIndex < lineSublistSize; ++lineIndex)
		{
			Vec2R startriVertexIndexoint = myStartLines[lineListIndex][lineIndex];
			Vec2R endPoint = myEndLines[lineListIndex][lineIndex];

			glVertex2d(startriVertexIndexoint[0], startriVertexIndexoint[1]);
			glVertex2d(endPoint[0], endPoint[1]);
		}
	}
	glEnd();	
	
	unsigned pointListSize = myPoints.size();
	for (unsigned pointListIndex = 0; pointListIndex < pointListSize; ++pointListIndex)
	{
		Vec3f pointColour = myPointColours[pointListIndex];
		glColor3f(pointColour[0], pointColour[1], pointColour[2]);
		glPointSize(myPointSize[pointListIndex]);
		
		glBegin(GL_POINTS);

		unsigned pointSublistSize = myPoints[pointListIndex].size();
		for (unsigned pointIndex = 0; pointIndex < pointSublistSize; ++pointIndex)
		{
			Vec2R point = myPoints[pointListIndex][pointIndex];
			glVertex2d(point[0], point[1]);
		}

		glEnd();
	}
}

void Renderer::run()
{
	glutMainLoop();
}
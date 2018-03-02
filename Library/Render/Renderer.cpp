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
		if (!m_render) m_render = render;

		glutReshapeFunc(reshape);
		glutDisplayFunc(display);
		glutMouseFunc(mouse);
		glutMotionFunc(drag);
		glutKeyboardFunc(keyboard);
	}
private:
	static void reshape(int w, int h)
	{
		if (m_render) m_render->reshape(w, h);
	}

	static void display()
	{
		if (m_render) m_render->display();
	}

	static void mouse(int button, int state, int x, int y)
	{
		if (m_render) m_render->mouse(button, state, x, y);
	}

	static void drag(int x, int y)
	{
		if (m_render) m_render->drag(x, y);
	}

	static void keyboard(unsigned char key, int x, int y)
	{
		if (m_render) m_render->keyboard(key, x, y);
	}

	static Renderer* m_render;
};

Renderer* GlutHelper::m_render;

Renderer::Renderer(const char *title, Vec2i window_size, Vec2R screen_min,
						Real height, int *argc, char **argv)
	: m_wsize(window_size), m_smin(screen_min), m_sheight(height),
	m_dmin(screen_min),	m_dheight(height), m_maction(INACTIVE)
{
	glutInit(argc, argv);

	//TODO: review if these flags are needed
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(window_size[0], window_size[1]);
	glutCreateWindow(title);
	glClearColor(1.0, 1.0, 1.0, 0.0);

	GlutHelper::init(this);
}

void Renderer::set_user_keyboard(std::function<void(unsigned char, int, int)> keyfunc)
{
	m_user_keyboard = keyfunc;
}

void Renderer::set_user_mouse_click(std::function<void(int, int, int, int)> clickfunc)
{
	m_user_mouse_click = clickfunc;
} 

void Renderer::set_user_mouse_drag(std::function<void(int, int)> dragfunc)
{
	m_user_mouse_drag = dragfunc;
}

void Renderer::set_user_display(std::function<void()> displayfunc)
{
	m_user_display = displayfunc;
}

void Renderer::display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: taken directly from gluvi.. might be able to clean some of this up
	glViewport(0, 0, (GLsizei)m_wsize[0], (GLsizei)m_wsize[1]);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(m_smin[0], m_smin[0] + (m_sheight * m_wsize[0]) / m_wsize[1], m_smin[1], m_smin[1] + m_sheight, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Run user specific display funcion
	if (m_user_display) m_user_display();
		
	draw_primitives();
	glutSwapBuffers();
	glutPostRedisplay();
}

void Renderer::mouse(int button, int state, int x, int y)
{
	// If the user provided a custom mouse handler,
	// then use that
	if (m_user_mouse_click) m_user_mouse_click(button, state, x, y);
	else
	{
		if (state == GLUT_UP)
		{
			switch (m_maction)
			{
				// PAN is only for drag
				// TODO: add click specific movement
				case ZOOM_IN:
				{
					// Zoom in by 2x
					Real scale = .5;

					m_smin[0] -= .5 * (scale - 1.) * m_wsize[0] * m_sheight / m_wsize[1];
					m_smin[1] -= .5 * (scale - 1.) * m_sheight;
					m_sheight *= scale;

					glutPostRedisplay();
					break;
				}
				case ZOOM_OUT:
				{
					Real scale = 2.;
					m_smin[0] -= .5 * (scale - 1.) * m_wsize[0] * m_sheight / m_wsize[1];
					m_smin[1] -= .5 * (scale - 1.) * m_sheight;
					m_sheight *= scale;

					glutPostRedisplay();
				}
			}
		}
		else
		{
			switch (button)
			{
			case GLUT_LEFT_BUTTON:
				m_maction = PAN;
				break;
			case GLUT_MIDDLE_BUTTON:
				m_maction = ZOOM_IN;
				break;
			case GLUT_RIGHT_BUTTON:
				m_maction = ZOOM_OUT;
			}
		}

		m_mmoved = false; // not really used at the moment
		m_mouse_pos = Vec2i(x, y);
	}
}

void Renderer::drag(int x, int y)
{
	if (m_user_mouse_drag) m_user_mouse_drag(x, y);
	else
	{
		if (x != m_mouse_pos[0] || y != m_mouse_pos[1])
		{
			m_mmoved = true;
			if (m_maction == PAN)
			{
				Real r = m_sheight / (Real)m_wsize[1];
				m_smin[0] -= r * ((Real)x - m_mouse_pos[0]);
				m_smin[1] += r * ((Real)y - m_mouse_pos[1]);
				glutPostRedisplay();
			}
			m_mouse_pos[0] = x;
			m_mouse_pos[1] = y;
		}
	}
}

void Renderer::keyboard(unsigned char key, int x, int y)
{
	if (m_user_keyboard) m_user_keyboard(key, x, y);

	// r triggers return to defaults
	if (key == 'r')
	{
		m_smin = m_dmin;
		m_sheight = m_dheight;
		glutPostRedisplay();
	}
}

void Renderer::reshape(int w, int h)
{
	m_wsize[0] = w;
	m_wsize[1] = h;
	glutPostRedisplay();
}

// These helpers make it easy to render out basic primitives without having to write
// a custom loop outside of this class
void Renderer::add_point(const Vec2R& point, const Vec3f& colour, Real size)
{
	std::vector<Vec2R> temp(1, point);
	m_points.push_back(temp);
	m_point_colours.push_back(colour);
	m_point_size.push_back(size);
}

void Renderer::add_points(const std::vector<Vec2R>& points, const Vec3f& colour, Real size)
{
	m_points.push_back(points);
	m_point_colours.push_back(colour);
	m_point_size.push_back(size);
}

void Renderer::add_line(const Vec2R& start, const Vec2R& end, const Vec3f& colour)
{
	std::vector<Vec2R> start_vec; start_vec.push_back(start);
	std::vector<Vec2R> end_vec; end_vec.push_back(end);
	m_start_lines.push_back(start_vec);
	m_end_lines.push_back(end_vec);

	m_line_colours.push_back(colour);
}

void Renderer::add_lines(const std::vector<Vec2R>& start, const std::vector<Vec2R>& end, const Vec3f& colour)
{
	assert(start.size() == end.size());
	m_start_lines.push_back(start);
	m_end_lines.push_back(end);
	m_line_colours.push_back(colour);
}

void Renderer::add_tris(const std::vector<Vec2R>& verts, const std::vector<Vec3st>& faces, const std::vector<Vec3f>& colour)
{
	m_tri_verts.push_back(verts);
	m_tri_faces.push_back(faces);
	m_tri_colours.push_back(colour);
}

void Renderer::add_quads(const std::vector<Vec2R>& verts, const std::vector<Vec4st>& faces, const std::vector<Vec3f>& colours)
{
	m_quad_verts.push_back(verts);
	m_quad_faces.push_back(faces);
	m_quad_colours.push_back(colours);
}

void Renderer::clear()
{
	m_points.clear();
	m_point_colours.clear();
	m_point_size.clear();

	m_start_lines.clear();
	m_end_lines.clear();
	m_line_colours.clear();

	m_tri_verts.clear();
	m_tri_faces.clear();
	m_tri_colours.clear();

	m_quad_verts.clear();
	m_quad_faces.clear();
	m_quad_colours.clear();
}

void Renderer::draw_primitives() const
{
	// Render quads
	glBegin(GL_QUADS);

	for (size_t qs = 0; qs < m_quad_faces.size(); ++qs)
	{
		for (size_t f = 0; f < m_quad_faces[qs].size(); ++f)
		{
			Vec3f colour = m_quad_colours[qs][f];
			glColor3f(colour[0], colour[1], colour[2]);

			
			for (size_t qp = 0; qp < 4; ++qp)
			{
				size_t face = m_quad_faces[qs][f][qp];
				Vec2R vert = m_quad_verts[qs][face];
				glVertex2d(vert[0], vert[1]);
			}
		}
	}
	glEnd();

	// Render tris
	glBegin(GL_TRIANGLES);
	for (size_t ts = 0; ts < m_tri_faces.size(); ++ts)
	{
		for (size_t f = 0; f < m_tri_faces[ts].size(); ++f)
		{
			Vec3f colour = m_tri_colours[ts][f];
			glColor3f(colour[0], colour[1], colour[2]);

			for (size_t tp = 0; tp < 3; ++tp)
			{
				size_t face = m_tri_faces[ts][f][tp];
				Vec2R p = m_tri_verts[ts][face];
				glVertex2d(p[0], p[1]);
			}
		}
	}
	glEnd();

	glBegin(GL_LINES);
	for (size_t ls = 0; ls < m_start_lines.size(); ++ls)
	{
		Vec3f c = m_line_colours[ls];
		glColor3f(c[0], c[1], c[2]);
		for (size_t l = 0; l < m_start_lines[ls].size(); ++l)
		{
			Vec2R sp = m_start_lines[ls][l];
			Vec2R ep = m_end_lines[ls][l];
			glVertex2d(sp[0], sp[1]);
			glVertex2d(ep[0], ep[1]);
		}
	}
	glEnd();	
	
	for (size_t ps = 0; ps < m_points.size(); ++ps)
	{
		Vec3f c = m_point_colours[ps];
		glColor3f(c[0], c[1], c[2]);
		glPointSize(m_point_size[ps]);
		glBegin(GL_POINTS);

		for (size_t p = 0; p < m_points[ps].size(); ++p)
		{
			Vec2R pp = m_points[ps][p];
			glVertex2d(pp[0], pp[1]);
		}

		glEnd();
	}
}

void Renderer::run()
{
	glutMainLoop();
}
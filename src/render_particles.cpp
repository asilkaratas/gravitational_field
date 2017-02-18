#include <GL/glew.h>

#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_particleRadius(0.125f * 0.5f),
      m_program(0),
      m_vbo(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
}

void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints()
{
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);



	glDrawArrays(GL_POINTS, 0, m_numParticles);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

void ParticleRenderer::DrawCircle(float cx, float cy, float r, int num_segments) {
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++)   {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle
        float x = r * cosf(theta);//calculate the x component
        float y = r * sinf(theta);//calculate the y component
        glVertex2f(x + cx, y + cy);//output vertex
    }
    glEnd();
}



void ParticleRenderer::display()
{

	for(int i = 0; i < m_numParticles*3; i += 3)
	{
		float x = m_pos[i];
		float y = m_pos[i+1];

		DrawCircle(x, y, m_particleRadius, 8);
	}

	/*
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(m_program);
	glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
	glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

	glColor3f(1, 1, 1);
	_drawPoints();

	glUseProgram(0);
	glDisable(GL_POINT_SPRITE_ARB);
	*/

}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}

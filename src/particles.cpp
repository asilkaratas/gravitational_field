#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"

//#define MAX_EPSILON_ERROR 5.00f
//#define THRESHOLD         0.10f //0.30f

#define GRID_SIZE       128 //64
#define NUM_PARTICLES   10000

const uint width = 600, height = 600;


bool bPause = false;

#define PI 3.1415926535898

uint numParticles = 0;
uint2 gridSize;

float timestep = 1.0f;
float damping = 1.0f;
float gravity = 0.00005f; // 0.0003f;
int iterations = 1;

float collideSpring = 0.4f;
float collideDamping = 0.02f;
float collideShear = 0.05f;
float collideAttraction = 0.0f;

ParticleSystem *psystem = 0;

static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

ParticleRenderer *renderer = 0;

const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

void initParticleSystem(int numParticles, uint2 gridSize)
{
    psystem = new ParticleSystem(numParticles, gridSize);
    psystem->reset();

    renderer = new ParticleRenderer;
    renderer->setParticleRadius(psystem->getParticleRadius());

    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (psystem)
    {
        delete psystem;
    }
    cudaDeviceReset();
    return;
}

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Gravitational Field");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    float cameraPosX = 0.5f;
    float cameraPosY = 0.5f;
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(cameraPosX,
	            cameraPosX + width,
	            cameraPosY + height,
	            cameraPosY,
	            -1,
	            1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    glutReportErrors();
}


void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Gravitational Field (%d circles): %3.1f fps", numParticles, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void display()
{
    sdkStartTimer(&timer);

    if (!bPause)
    {
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideSpring(collideSpring);
        psystem->setCollideDamping(collideDamping);
        psystem->setCollideShear(collideShear);
        psystem->setCollideAttraction(collideAttraction);
        psystem->update(timestep);

        renderer->setPositions(psystem->getPositionArray(), psystem->getNumParticles());
        //renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();
    gluLookAt(0.0, 0.0, 1.73, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    renderer->display();
    glPopMatrix();

    sdkStopTimer(&timer);

    glFlush();
    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}



inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void reshape(int w, int h)
{
	glutReshapeWindow(600, 600);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);


    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    renderer->setWindowSize(w, h);
    renderer->setFOV(60.0);
}



void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case ' ':
            bPause = !bPause;
            break;

        case 13:
            psystem->update(timestep);

            if (renderer)
            {
                renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
            }

            break;

        case '\033':
        case 'q':
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif

        case '1':
            psystem->reset();
            break;

    }

    glutPostRedisplay();
}

void idle(void)
{
    glutPostRedisplay();
}



void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset", '1');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


int
main(int argc, char **argv)
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;

    gridSize.x = gridSize.y = gridDim;
    printf("grid: %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.x*gridSize.y);
    printf("particles: %d\n", numParticles);

    initGL(&argc, argv);
    cudaGLInit(argc, argv);

    initParticleSystem(numParticles, gridSize);
    initMenus();

    glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);

	glutCloseFunc(cleanup);

	glutMainLoop();

    if (psystem)
    {
        delete psystem;
    }

    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}


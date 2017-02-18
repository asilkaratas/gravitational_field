#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint2 gridSize) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_dPos(0),
    m_dVel(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y;

    m_gridSortBits = 18;

    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 128.0f;

    m_params.worldOrigin = make_float2(-1.0f, -1.0f);
    float cellSize = m_params.particleRadius * 2.0f;
    m_params.cellSize = make_float2(cellSize, cellSize);

    m_params.boundaryDamping = -0.5f;

    /*
    m_params.spring = 0.3f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float2(0.0f, -0.0003f);
    m_params.globalDamping = 1.0f;
    */

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*DIM];
    m_hVel = new float[m_numParticles*DIM];
    memset(m_hPos, 0, m_numParticles*DIM*sizeof(float));
    memset(m_hVel, 0, m_numParticles*DIM*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * DIM * m_numParticles;

    m_posVbo = createVBO(memSize);
    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);

    allocateArray((void **)&m_dVel, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));


    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dVel);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

	unregisterGLBufferObject(m_cuda_posvbo_resource);
	glDeleteBuffers(1, (const GLuint *)&m_posVbo);
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);

    setParameters(&m_params);

    integrateSystem(
        dPos,
        m_dVel,
        deltaTime,
        m_numParticles);

    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_numParticles);

    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_dVel,
        m_numParticles,
        m_numGridCells);

    collide(
        m_dVel,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);

    unmapGLBufferObject(m_cuda_posvbo_resource);
}

float *
ParticleSystem::getPositionArray()
{
	assert(m_bInitialized);
	float *hdata = m_hPos;
	float *ddata = m_dPos;
	struct cudaGraphicsResource *cuda_vbo_resource = m_cuda_posvbo_resource;
	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*DIM*sizeof(float));
	return hdata;
}

void
ParticleSystem::setPositionArray(const float *data, int start, int count)
{
	assert(m_bInitialized);

	unregisterGLBufferObject(m_cuda_posvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
	glBufferSubData(GL_ARRAY_BUFFER, start*DIM*sizeof(float), count*DIM*sizeof(float), data);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
}

float *
ParticleSystem::getVelocityArray()
{
	assert(m_bInitialized);
	float *hdata = m_hPos;
	float *ddata = m_dVel;
	struct cudaGraphicsResource *cuda_vbo_resource = 0;
	copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*DIM*sizeof(float));
	return hdata;
}

void
ParticleSystem::setVelocityArray(const float *data, int start, int count)
{
	assert(m_bInitialized);
	copyArrayToDevice(m_dVel, data, start*DIM*sizeof(float), count*DIM*sizeof(float));
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}


void
ParticleSystem::reset()
{
	int p = 0, v = 0;

	for (uint i=0; i < m_numParticles; i++)
	{
		float point[2];
		point[0] = frand();
		point[1] = frand();
		m_hPos[p++] = 2 * (point[0] - 0.5f);
		m_hPos[p++] = 2 * (point[1] - 0.5f);
		m_hPos[p++] = 0.0f;
		m_hVel[v++] = 0.0f;
		m_hVel[v++] = 0.0f;
		m_hVel[v++] = 0.0f;
	}

	setPositionArray(m_hPos, 0, m_numParticles);
	setVelocityArray(m_hVel, 0, m_numParticles);
}


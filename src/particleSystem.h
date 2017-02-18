#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint2 gridSize);
        ~ParticleSystem();

        static const int DIM = 3;

        void update(float deltaTime);
        void reset();


        float *getPositionArray();
        void setPositionArray(const float *data, int start, int count);

        float *getVelocityArray();
        void setVelocityArray(const float *data, int start, int count);

        int getNumParticles() const
        {
            return m_numParticles;
        }

        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }

        void *getCudaPosVBO()              const
        {
            return (void *)m_cudaPosVBO;
        }

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x)
        {
            m_params.gravity = make_float2(0.0f, x);
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }

        uint2 getGridSize()
        {
            return m_params.gridSize;
        }
        float2 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float2 getCellSize()
        {
            return m_params.cellSize;
        }

    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);

        void _initialize(int numParticles);
        void _finalize();



    protected: // data
        bool m_bInitialized;
        uint m_numParticles;

        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVel;

        float *m_dSortedPos;
        float *m_dSortedVel;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;

        uint   m_posVbo;            // vertex buffer object for particle positions

        float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos

        struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange

        // params
        SimParams m_params;
        uint2 m_gridSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__

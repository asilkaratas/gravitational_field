#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H


#define FETCH(t, i) t[i]

#include "vector_types.h"
typedef unsigned int uint;

struct SimParams
{
    float2 gravity;
    float globalDamping;
    float particleRadius;

    uint2 gridSize;
    uint numCells;
    float2 worldOrigin;
    float2 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
};

#endif

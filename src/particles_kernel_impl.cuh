#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

__constant__ SimParams params;

struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float3 posData = thrust::get<0>(t);
        volatile float3 velData = thrust::get<1>(t);
        float2 pos = make_float2(posData.x, posData.y);
        float2 vel = make_float2(velData.x, velData.y);

        vel += params.gravity * deltaTime;
        vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }



#endif

        if (pos.y < -1.0f + params.particleRadius)
        {
            pos.y = -1.0f + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float3(pos, posData.z);
        thrust::get<1>(t) = make_float3(vel, velData.z);
    }
};

// calculate position in uniform grid

__device__ int2 calcGridPos(float2 p)
{
    int2 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)

__device__ uint calcGridHash(int2 gridPos)
{
	//printf("A x:%d y:%d\n", gridPos.x, gridPos.y);
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);  // ensures each point in the grid rectangle
    //printf("B x:%d y:%d\n\n", gridPos.x, gridPos.y);



    return __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle

__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float3 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float3 p = pos[index];

    // get address in grid
    int2 gridPos = calcGridPos(make_float2(p.x, p.y));
    uint hash = calcGridHash(gridPos);
   // printf("A x:%d y:%d hash:%d\n", gridPos.x, gridPos.y, hash);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array

__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float3 *sortedPos,        // output: sorted positions
                                  float3 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float3 *oldPos,           // input: sorted position array
                                  float3 *oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float3 pos = FETCH(oldPos, sortedIndex);       // macro does global read
        float3 vel = FETCH(oldVel, sortedIndex);

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }


}


__device__
float2 collideCircles(float2 posA, float2 posB,
                      float2 velA, float2 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    // calculate relative position
    float2 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float2 force = make_float2(0.0f);

    if (dist < collideDist)
    {
        float2 norm = relPos / dist;

        // relative velocity
        float2 relVel = velB - velA;

        // relative tangential velocity
        float2 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}



// collide a particle against all other particles in a given cell
__device__
float2 collideCell(int2    gridPos,
                   uint    index,
                   float2  pos,
                   float2  vel,
                   float3 *oldPos,
                   float3 *oldVel,
                   uint   *cellStart,
                   uint   *cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    float2 force = make_float2(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float2 pos2 = make_float2(FETCH(oldPos, j));
                float2 vel2 = make_float2(FETCH(oldVel, j));

                force += collideCircles(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
            }
        }
    }

    return force;
}


__global__
void collideD(float3 *newVel,               // output: new velocity
              float3 *oldPos,               // input: sorted positions
              float3 *oldVel,               // input: sorted velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float2 pos = make_float2(FETCH(oldPos, index));
    float2 vel = make_float2(FETCH(oldVel, index));

    // get address in grid
    int2 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float2 force = make_float2(0.0f);

    for (int y=-1; y<=1; y++)
	{
		for (int x=-1; x<=1; x++)
		{
			int2 neighbourPos = gridPos + make_int2(x, y);
			force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
		}
	}

    // write new velocity back to original unsorted location
    uint originalIndex = gridParticleIndex[index];
    newVel[originalIndex] = make_float3(vel + force, 0.0f);
}

#endif

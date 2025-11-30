/**
 * This kernel calculates cell index for each partical.
 *
 * TODO Documentation
 * @param positions             (in/out) Global buffer of particle positions (vec2). This is shared with an OpenGL VBO.
 * @param particalCellIndex     (in/out) Global buffer of particle cell indexes
 * @param gridNx                (in)     Grid X size.
 * @param gridNy                (in)     Grid Y size.
 * @param cellSizeInvX          (in)     Cell Inv X size.
 * @param cellSizeInvY          (in)     Cell Inv Y size.
 * @param worldMinX             (in)     World minimum X coordinate.
 * @param worldMinY             (in)     World minimum Y coordinate.
 */

__kernel void computeParticleCellIndex(
    __global const float4* posVel, 
    __global int* particleCellIndex,
    const int gridNx,
    const int gridNy,
	const float cellSizeInvX,
    const float cellSizeInvY,
	const float worldMinX,
    const float worldMinY,
    const int numParticles)
{
    int pid = get_global_id(0);
    if (pid >= numParticles) return;
    float2 pos = (float2)(posVel[pid].x, posVel[pid].y);

    int cellX = (int)((pos.x - worldMinX) * cellSizeInvX);
    int cellY = (int)((pos.y - worldMinY) * cellSizeInvY);

    cellX = clamp(cellX, 0, gridNx - 1);
    cellY = clamp(cellY, 0, gridNy - 1);

    particleCellIndex[pid] = cellX + cellY * gridNx;
}

/**
 * This kernel calculates COM for each cell.
 *
 * TODO documentation
 * @param 
 */
__kernel void computeCellCOM(
    __global const float4* posVel,
    __global const float* masses,
    __global const int* particleCellIndex,
    __global float* cellMass,
    __global float2* cellCOM,
    const int numParticles,
    const int totalCells,
    __local float* s_m,        
    __local float* s_mpx,      
    __local float* s_mpy      
)
{
    int cell  = get_group_id(0);
    if (cell >= totalCells) return;

    int lid   = get_local_id(0);
    int lsize = get_local_size(0);

    float my_m   = 0.0f;
    float my_mpx = 0.0f;
    float my_mpy = 0.0f;

    for (int pid = lid; pid < numParticles; pid += lsize) {
        if (particleCellIndex[pid] == cell) {
            float m = masses[pid];
            float2 p = (float2)(posVel[pid].x, posVel[pid].y);
            my_m   += m;
            my_mpx += p.x * m;
            my_mpy += p.y * m;
        }
    }

    s_m[lid]   = my_m;
    s_mpx[lid] = my_mpx;
    s_mpy[lid] = my_mpy;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        if (lid < offset) {
            s_m[lid]   += s_m[lid + offset];
            s_mpx[lid] += s_mpx[lid + offset];
            s_mpy[lid] += s_mpy[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        float mass = s_m[0];
        cellMass[cell]   = mass;
        cellCOM[cell] = (float2)(s_mpx[0], s_mpy[0]);
    }
}

/**
 * TODO Documentation
 * This kernel calculates the gravitational forces between a system of particles
 * to simulate their motion over time.
 *
 *
 * @param velocities      (in/out) Global buffer of particle velocities (vec2).
 * @param positions       (in/out) Global buffer of particle positions (vec2). This is shared with an OpenGL VBO.
 * @param masses          (in)     Global buffer of particle masses (float).
 * @param deltaTime       (in)     The time step for the simulation frame.
 */
__kernel void update(
    __global float4* posVel,
    __global const float* masses,
    __global const int* particleCellIndex,
    __global const float* cellMass,       
    __global const float2* cellCOM,
    const int gridNx,
    const int gridNy,
    const int totalCells,
    const int numParticles,
    const float deltaTime)
{
    const float G = 0.0001f;
    const float softening = 0.001f;

    int pid = get_global_id(0);
    if (pid >= numParticles) return;

    float4 pv = posVel[pid];
    float2 p = (float2)(pv.x, pv.y);
    float2 v = (float2)(pv.z, pv.w);

    int myCell = particleCellIndex[pid];
    int myCx = myCell % gridNx;
    int myCy = myCell / gridNx;

    float2 acc = (float2)(0.0f, 0.0f);

    for (int c = 0; c < totalCells; ++c) {
        float cm = cellMass[c];
        if (cm <= 0.0f) continue;
        float2 cpos = (float2)(cellCOM[c].x / cm, cellCOM[c].y / cm);

        float2 d = cpos - p;
        float r2 = d.x * d.x + d.y * d.y + softening;
        float invr = 1.0f / sqrt(r2);
        float invr3 = invr * invr * invr;
        acc += d * (G * cm * invr3);
    }

    float2 newV = v + acc * deltaTime;
    float2 newP = p + newV * deltaTime;
    posVel[pid] = (float4)(newP.x, newP.y, newV.x, newV.y);
}
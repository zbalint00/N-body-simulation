/**
 * For each particle, this kernel calculates which grid cell it belongs to.
 * The world is split into a 3D grid with gridNx * gridNy * gridNz cells.
 *
 * @param pos                   (in/out) Global buffer of particle positions (float4). This is shared with an OpenGL VBO.
 * @param particalCellIndex     (in/out) Global buffer of particle's cell index
 * @param gridNx                (in)     Number of cells in X direction.
 * @param gridNy                (in)     Number of cells in Y direction.
 * @param gridNz                (in)     Number of cells in Z direction.
 * @param cellSizeInvX          (in)     Inverse cell size in X.
 * @param cellSizeInvY          (in)     Inverse cell size in Y.
 * @param cellSizeInvZ          (in)     Inverse cell size in Z.
 * @param worldMinX             (in)     World minimum X coordinate.
 * @param worldMinY             (in)     World minimum Y coordinate.
 * @param worldMinZ             (in)     World minimum Z coordinate.
 * @param numParticles          (in)     Number of particles.
 */

__kernel void computeParticleCellIndex(
    __global const float3* pos, 
    __global int* particleCellIndex,
    const int gridNx,
    const int gridNy,
    const int gridNz,
	const float cellSizeInvX,
    const float cellSizeInvY,
    const float cellSizeInvZ,
	const float worldMinX,
    const float worldMinY,
    const float worldMinZ,
    const int numParticles)
{
    // Global thread id is the particle index
    int pid = get_global_id(0);
    if (pid >= numParticles) return;
    
    float3 position = pos[pid];

    // Compute cell coordinates in floating point, then cast to int
    int cellX = (int)((position.x - worldMinX) * cellSizeInvX);
    int cellY = (int)((position.y - worldMinY) * cellSizeInvY);
    int cellZ = (int)((position.z - worldMinZ) * cellSizeInvZ);
    
    // Clamp cell indexes to the valid grid range
    cellX = clamp(cellX, 0, gridNx - 1);
    cellY = clamp(cellY, 0, gridNy - 1);
    cellZ = clamp(cellZ, 0, gridNz - 1);

    // Store which cell this particle belongs to (Converting 3D cell coordinates to a single 1D index)
    particleCellIndex[pid] = cellX + cellY * gridNx + cellZ * gridNx * gridNy;
}

/**
 * For each cell, this kernel computes:
 *   - the total mass inside the cell
 *   - the sum of (mass * position) inside the cell.
 *
 * NOTE
 *   - cellCOM[cell] stores (sum(m * x), sum(m * y)) for all particles in the cell.
 *   - The actual center of mass (COM) is computed later as:
 *         COM = cellCOM[cell] / cellMass[cell]
 *
 * Work distribution:
 *   - One work-group works on one cell.
 *   - Inside the group, each thread processes a subset of particles.
 *   - Threads write their partial sums to local memory.
 *   - A local reduction combines all partial sums into a single result per cell.
 *
 * @param pos             (in/out)        Global buffer of particle position (x,y,z).
 * @param masses             (in/out)     Global buffer of particle masses.
 * @param particleCellIndex  (in/out)     Global buffer of particle's cell index.
 * @param cellMass           (in/out)     For each cell, the total mass of particles in that cell.
 * @param cellCOM            (in/out)     For each cell, the mass center position.
 * @param numParticles       (in)         Number of particles.
 * @param totalCells         (in)         Total number of cells in the world.
 * @param localMass          (local)      Per-thread partial mass sums, then reduced to total mass.
 * @param localCOMX          (local)      Per-thread partial sums of (mass * pos.x), then reduced.
 * @param localCOMY          (local)      Per-thread partial sums of (mass * pos.y), then reduced.
 * @param localCOMZ          (local)      Per-thread partial sums of (mass * pos.z), then reduced.
 */
__kernel void computeCellCOM(
    __global const float3* pos,
    __global const float* masses,
    __global const int* particleCellIndex,
    __global float* cellMass,
    __global float3* cellCOM,
    const int numParticles,
    const int totalCells,
    __local float* localMass,        
    __local float* localCOMX,      
    __local float* localCOMY,      
    __local float* localCOMZ      
)
{
    // This work-group is responsible for this cell.
    int cellId  = get_group_id(0);
    if (cellId >= totalCells) return;

    // Local thread index and group size.
    int localId   = get_local_id(0);
    int localSize = get_local_size(0);

    // Per-thread partial sums.
    float threadMass  = 0.0f;
    float threadCOMX  = 0.0f;
    float threadCOMY  = 0.0f;
    float threadCOMZ  = 0.0f;

    // Each thread visits particles in a strided way:
    // particleId = localId, localId + localSize, localId + 2*localSize, ...
    for (int particleId = localId; particleId < numParticles; particleId += localSize) {
        // Only count particles that belong to this cell.
        if (particleCellIndex[particleId] == cellId) {
            float mass = masses[particleId];
            float3 position = pos[particleId];
            threadMass += mass;
            threadCOMX += position.x * mass;
            threadCOMY += position.y * mass;
            threadCOMZ += position.z * mass;
        }
    }

    // Store partial sums in local (shared) memory.
    localMass[localId]  = threadMass;
    localCOMX[localId]  = threadCOMX;
    localCOMY[localId]  = threadCOMY;
    localCOMZ[localId]  = threadCOMZ;

    // Wait every thread to finish
    barrier(CLK_LOCAL_MEM_FENCE);

    // Local reduction: binary tree pattern.
    // On each step, the first half of threads add values from the second half.
    for (int offset = localSize >> 1; offset > 0; offset >>= 1) {
        if (localId < offset) {
            localMass[localId] += localMass[localId + offset];
            localCOMX[localId] += localCOMX[localId + offset];
            localCOMY[localId] += localCOMY[localId + offset];
            localCOMZ[localId] += localCOMZ[localId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // After reduction, index 0 holds the full sums for this cell.
    if (localId == 0) {
        float totalMass = localMass[0];

        // Store total mass for this cell.
        cellMass[cellId] = totalMass;

        // Store mass position for this cell.
        // The actual COM is computed in the update kernel as cellCOM / cellMass.
        cellCOM[cellId] = (float3)(localCOMX[0], localCOMY[0], localCOMZ[0]);
    }
}

/**
 * This kernel updates particle positions and velocities using a space-partitioned
 * model with a grid-based approximation.
 *
 *
 * For each particle:
 *   - determine its grid cell from particleCellIndex,
 *   - loop over all other particles and:
 *       * compute exact particle-to-particle forces for particles
 *         in the same cell or in one of the 25 neighboring cells
 *         (the local 26 block),
 *   - loop over all grid cells and:
 *       * skip empty cells (cellMass[cell] <= 0),
 *       * skip cells in the local 26 neighborhood (already handled exactly),
 *       * for all other (distant) cells, treat the whole cell as a single
 *         mass located at its center of mass, computed from cellMass and cellCOM,
 *         and add this approximate contribution to the acceleration,
 *   - integrate the total acceleration to update velocity and position.
 *
 * @param pos               (in/out)     Global buffer of particle state: x,y,z = position.
 * @param vel               (in/out)     Global buffer of particle state: x,y,z = velocity.
 * @param masses            (in)         Global buffer of particle masses (float).
 * @param particleCellIndex (in/out)     Global buffer of particle's cell index.
 * @param cellMass          (in/out)     For each cell, total mass in that cell.
 * @param cellCOM           (in/out)     For each cell, sum of (mass * position) in that cell.
 * @param gridNx            (in)         Number of cells in X direction.
 * @param gridNy            (in)         Number of cells in Y direction.
 * @param gridNz            (in)         Number of cells in Z direction.
 * @param totalCells        (in)         Total number of cells (gridNx * gridNy).
 * @param numParticles      (in)         Number of particles.
 * @param G                 (in)         A physically-motivated gravitational constant. (float)
 * @param deltaTime         (in)         Time step for integration.
 */
__kernel void update(
    __global float3* pos,
    __global float3* vel,
    __global const float* masses,
    __global const int* particleCellIndex,
    __global const float* cellMass,       
    __global const float3* cellCOM,
    const int gridNx,
    const int gridNy,
    const int gridNz,
    const int totalCells,
    const int numParticles,
    const float G,
    const float deltaTime)
{
          
    // A small factor to prevent forces from becoming infinite during close encounters, improving stability.
    const float softening = 0.001f;

    // One thread updates one particle.
    int particleId = get_global_id(0);
    if (particleId >= numParticles) return;

    // Load particle state: position and velocity.
    float3 position   = pos[particleId];
    float3 velocity   = vel[particleId];
    float currentMass = masses[particleId];

    // Actual particle's cell
    int myCellIndex = particleCellIndex[particleId];
    int myCellX     = myCellIndex % gridNx;
    int myCellY     = myCellIndex / gridNx;
    int myCellZ     = (myCellIndex / gridNx) / gridNy;

    // Start with zero acceleration.
    float3 totalAcceleration  = (float3)(0.0f, 0.0f, 0.0f);

    for (int otherId = 0; otherId < numParticles; ++otherId)
    {
        if (otherId == particleId)
            continue; // no self-interaction

        int otherCellIndex = particleCellIndex[otherId];
        int otherCellX     = otherCellIndex % gridNx;
        int otherCellY     = otherCellIndex / gridNx;
        int otherCellZ     = (otherCellIndex / gridNx) / gridNy;

        int dxCell = otherCellX - myCellX;
        int dyCell = otherCellY - myCellY;
        int dzCell = otherCellZ - myCellZ;
        if (dxCell < 0) dxCell = -dxCell;
        if (dyCell < 0) dyCell = -dyCell;
        if (dzCell < 0) dzCell = -dzCell;

        // Only exact interaction if the other particle is in the same cell
        // or in one of the 25 neighboring cells (26 block).
        if (dxCell <= 1 && dyCell <= 1 && dzCell <= 1)
        {
            float3 otherPos   = pos[otherId];

            // Vector from current particle to other particle.
            float3 vectorToOther = otherPos - position;

            // Same distance computation as in the original update kernel.
            float distanceSquared = vectorToOther.x * vectorToOther.x
                                  + vectorToOther.y * vectorToOther.y
                                  + vectorToOther.z * vectorToOther.z
                                  + softening;

            float invDist        = 1.0f / sqrt(distanceSquared);
            float invDistCube    = invDist * invDist * invDist; // 1 / r^3

            // Same force magnitude formula as original:
            // forceMagnitude = G * m_other * invDistCube
            float otherMass      = masses[otherId];
            float forceMagnitude = (G * otherMass) * invDistCube;

            // Accumulate acceleration (a = F/m, own mass cancels out here).
            totalAcceleration += vectorToOther * forceMagnitude;
        }
    }


    // Loop over all cells and add their contribution.
    for (int cellIndex = 0; cellIndex < totalCells; ++cellIndex) {
        float cellMassValue = cellMass[cellIndex];
        if (cellMassValue <= 0.0f) continue; // skip empty cells

        int cellX = cellIndex % gridNx;
        int cellY = cellIndex / gridNx;
        int cellZ = (cellIndex / gridNx) / gridNy;

        int dxCell = cellX - myCellX;
        int dyCell = cellY - myCellY;
        int dzCell = cellZ - myCellZ;
        if (dxCell < 0) dxCell = -dxCell;
        if (dyCell < 0) dyCell = -dyCell;
        if (dzCell < 0) dzCell = -dzCell;

        // Skip cells in our 3x3 neighborhood (own + 8 neighbors),
        // because their particles were already handled exactly above.
        if (dxCell <= 1 && dyCell <= 1 && dzCell <= 1)
            continue;

        // Compute center of mass of this cell:
        float3 cellCOMHelper = cellCOM[cellIndex];
        float3 cellCOMPosition  = (float3)(cellCOMHelper.x / cellMassValue, cellCOMHelper.y / cellMassValue, cellCOMHelper.z / cellMassValue);

        // Direction vector from particle to cell COM.
        float3 direction = cellCOMPosition - position;

         // Distance squared + softening.
        float distanceSquared = direction.x * direction.x
                              + direction.y * direction.y
                              + direction.z * direction.z
                              + softening;

        float invDistance      = 1.0f / sqrt(distanceSquared);
        float invDistanceCubed = invDistance * invDistance * invDistance;

        // Gravitational acceleration contribution from this cell.
        // Proportional to G * cellMass / r^2, with direction.
        totalAcceleration  += direction * (G * cellMassValue * invDistanceCubed);
    }

    // Integrate motion: update velocity, then position.
    float3 newVelocity = velocity + totalAcceleration  * deltaTime;
    float3 newPosition = position + newVelocity * deltaTime;

    // Store updated state back to global buffer.
    pos[particleId] = (float3)(newPosition.x, newPosition.y, newPosition.z);
    vel[particleId] = (float3)(newVelocity.x, newVelocity.y, newVelocity.z);
}
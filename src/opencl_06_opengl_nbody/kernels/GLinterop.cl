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
 * This kernel counts how many particles fall into each grid cell.
 *
 * For each particle:
 *   - read its precomputed cell index from particleCellIndex,
 *   - atomically increment the corresponding entry in cellCount.
 *
 * cellCount must be zero-initialized before launching this kernel.
 *
 * @param particleCellIndex (in/out) Global buffer of particle's cell index.
 * @param cellCount         (out) Per-cell particle count.
 * @param numParticles      (in)  Number of particles.
 */
__kernel void countParticlesPerCell(
    __global const int* particleCellIndex,
    __global int*       cellCount,
    const int           numParticles)
{
    int particleId  = get_global_id(0);
    if (particleId  >= numParticles) return;

    int cellId = particleCellIndex[particleId];
    
    // Atomic add to handle thread race
    atomic_inc(&cellCount[cellId]);
}

/**
 * This kernel performs an exclusive prefix sum (scan) over the cellCount array
 * to compute the starting index of each cell in the "sorted by cell" index array.
 *
 * After the kernel cellStart[c] will hold the starting index (offset) for cell c in the sorted index array
 *
 * A single work-item (gid == 0) performs the scan sequentially.
 *
 * @param cellCount   (in)  Per-cell particle counts.
 * @param cellStart   (out) Per-cell start indices in the sorted index array.
 * @param totalCells  (in)  Total number of cells in the grid.
 */
__kernel void generateCellStartPrefix(
    __global int* cellCount,
    __global int* cellStart,
    const int     totalCells)
{
    int gid = get_global_id(0);
    if (gid != 0) return;

    int sum = 0;
    // For each cell, calculate where the beginning of the block of particles belonging to the cell begins in the sortedIndex array.
    for (int cellId = 0; cellId < totalCells; ++cellId) {
        int count = cellCount[cellId];
        cellStart[cellId] = sum;
        sum += count;
    }
    
    cellStart[totalCells] = sum;
}

/**
 * This kernel builds a sorted by cell particle index array.
 *
 * It will group particles by their cell, so that all particles
 * belonging to cell "c" will be one after another in the sortedIndex buffer:
 *
 * @param particleCellIndex (in/out) Global buffer of particle's cell index.
 * @param cellCount         (in/out) Per-cell particle counts. (Note: It will be modified due to the decreasing)
 * @param cellStart         (in)  For each cell, its starting offset into sortedIndex.
 * @param sortedIndex       (out) Sorted by cell particle indexes.
 * @param numParticles      (in)  Number of particles.
 */
__kernel void sortParticlesByCell(
    __global const int* particleCellIndex,
    __global int*       cellCount,
    __global const int* cellStart,
    __global int*       sortedIndex,
    const int           numParticles)
{
    int particleId  = get_global_id(0);
    if (particleId  >= numParticles) return;

    int cellId  = particleCellIndex[particleId];

    // Get unique index for particles inside cell. Atomic decrement to handle thread race
    int indexWithinCell = atomic_dec(&cellCount[cellId]) - 1;
    // Actual index in the sorted array
    int destIndex  = cellStart[cellId] + indexWithinCell;

    // Store the particle id into its position in the cell-grouped array.
    sortedIndex[destIndex] = particleId;
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
 *   - One work-group works on one cell.
 *   - Inside the group, each thread processes a subset of particles.
 *   - Threads write their partial sums to local memory.
 *   - A local reduction combines all partial sums into a single result per cell.
 *
 * @param pos             (in/out)        Global buffer of particle position (x,y,z).
 * @param masses             (in/out)     Global buffer of particle masses.
 * @param cellMass           (in/out)     For each cell, the total mass of particles in that cell.
 * @param cellCOM            (in/out)     For each cell, the mass center position.
 * @param totalCells         (in)         Total number of cells in the world.
 * @param cellStart          (in/out)     Per-cell start indexes in the sorted index array.
 * @param sortedIndex        (in/out)     Sorted by cell particle indexes.
 * @param localMass          (local)      Per-thread partial mass sums, then reduced to total mass.
 * @param localCOMX          (local)      Per-thread partial sums of (mass * pos.x), then reduced.
 * @param localCOMY          (local)      Per-thread partial sums of (mass * pos.y), then reduced.
 * @param localCOMZ          (local)      Per-thread partial sums of (mass * pos.z), then reduced.
 */
__kernel void computeCellCOM(
    __global const float3* pos,
    __global const float* masses,
    __global float* cellMass,
    __global float3* cellCOM,
    const int totalCells,
    __global const int* cellStart,
    __global const int* sortedIndex, 
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

    // Range in sortedIndex belonging to this cell:
    int startIndex = cellStart[cellId];
    int endIndex   = cellStart[cellId + 1]; 

    // Per-thread partial sums.
    float threadMass  = 0.0f;
    float threadCOMX  = 0.0f;
    float threadCOMY  = 0.0f;
    float threadCOMZ  = 0.0f;

    // Each thread visits particles in a strided way
    // It will only visit particles inside the current cell
    for (int id = startIndex + localId; id < endIndex; id += localSize) {
        int particleId = sortedIndex[id];
        float mass = masses[particleId];
        float3 position = pos[particleId];
        
        threadMass += mass;
        threadCOMX += position.x * mass;
        threadCOMY += position.y * mass;
        threadCOMZ += position.z * mass;       
    }

    // Store partial sums in local (shared) memory.
    localMass[localId]  = threadMass;
    localCOMX[localId]  = threadCOMX;
    localCOMY[localId]  = threadCOMY;
    localCOMZ[localId]  = threadCOMZ;

    // Wait every thread to finish
    barrier(CLK_LOCAL_MEM_FENCE);

    // Local reduction:
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
 *   - loop over particles in the same cell and in the 26 neighboring cells,
 *   - loop over all grid cells and:
 *       * skip empty cells (cellMass[cell] <= 0),
 *       * skip cells in the local 26 neighborhood,
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
 * @param cellStart         (in/out)     Per-cell start indexes in the sorted index array.
 * @param sortedIndex       (in/out)     Sorted by cell particle indexes.
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
    __global const int* cellStart,
    __global const int* sortedIndex,
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
    int myCellY     = (myCellIndex / gridNx) % gridNy;
    int myCellZ     = myCellIndex / (gridNx * gridNy);

    // Start with zero acceleration.
    float3 totalAcceleration  = (float3)(0.0f, 0.0f, 0.0f);

 for (int neighborOffsetZ = -1; neighborOffsetZ <= 1; ++neighborOffsetZ) {
        int neighborCellZ = myCellZ + neighborOffsetZ;
        if (neighborCellZ < 0 || neighborCellZ >= gridNz) continue;

        for (int neighborOffsetY = -1; neighborOffsetY <= 1; ++neighborOffsetY) {
            int neighborCellY = myCellY + neighborOffsetY;
            if (neighborCellY < 0 || neighborCellY >= gridNy) continue;

            for (int neighborOffsetX = -1; neighborOffsetX <= 1; ++neighborOffsetX) {
                int neighborCellX = myCellX + neighborOffsetX;
                if (neighborCellX < 0 || neighborCellX >= gridNx) continue;

                // Neighbor cell 3D coordinates -> 1D index
                int neighborCellIndex = neighborCellX + neighborCellY * gridNx + neighborCellZ * gridNx * gridNy;
                
                // Range in sortedIndex for this neighbor cell.
                int cellRangeStart = cellStart[neighborCellIndex];
                int cellRangeEnd   = cellStart[neighborCellIndex + 1];

                // Loop over all particles in this neighbor cell.
                for (int id = cellRangeStart; id < cellRangeEnd; ++id) {
                    int otherParticleId = sortedIndex[id];
                    if (otherParticleId == particleId) continue;

                    float3 otherPosition = pos[otherParticleId];
                    // Vector from this particle to the neighbor particle.
                    float3 direction = otherPosition - position;

                    // Distance squared + softening factor.
                    float distanceSquared = direction.x * direction.x
                                          + direction.y * direction.y 
                                          + direction.z * direction.z
                                          + softening;

                    float invDistance  = 1.0f / sqrt(distanceSquared);
                    float invDistanceCubed = invDistance * invDistance * invDistance;

                    float otherMass      = masses[otherParticleId];
                    float forceMagnitude = (G * otherMass) * invDistanceCubed;

                    totalAcceleration += direction * forceMagnitude;
                }
            }
        }
    }
    

    // Loop over all cells and add their contribution.
    for (int cellIndex = 0; cellIndex < totalCells; ++cellIndex) {
        float cellMassValue = cellMass[cellIndex];
        if (cellMassValue <= 0.0f) continue; // skip empty cells

        int cellX = cellIndex % gridNx;
        int cellY = (cellIndex / gridNx) % gridNy;
        int cellZ = cellIndex / (gridNx * gridNy);

        int dxCell = cellX - myCellX;
        int dyCell = cellY - myCellY;
        int dzCell = cellZ - myCellZ;
        if (dxCell < 0) dxCell = -dxCell;
        if (dyCell < 0) dyCell = -dyCell;
        if (dzCell < 0) dzCell = -dzCell;

        // Skip cells in our neighborhood,
        if (dxCell <= 1 && dyCell <= 1 && dzCell <= 1)
            continue;

        // Compute center of mass of this cell:
        float3 cellCOMHelper = cellCOM[cellIndex];
        float3 cellCOMPosition  = (float3)(
                                cellCOMHelper.x / cellMassValue, 
                                cellCOMHelper.y / cellMassValue, 
                                cellCOMHelper.z / cellMassValue);

        // Direction vector from particle to cell COM.
        float3 direction = cellCOMPosition - position;

         // Distance squared + softening.
        float distanceSquared = direction.x * direction.x
                              + direction.y * direction.y
                              + direction.z * direction.z
                              + softening;

        float invDistance      = 1.0f / sqrt(distanceSquared);
        float invDistanceCubed = invDistance * invDistance * invDistance;
        float forceMagnitude = (G * cellMassValue) * invDistanceCubed;
        // Gravitational acceleration contribution from this cell.
        // Proportional to G * cellMass / r^2, with direction.
        totalAcceleration  += direction * forceMagnitude;
    }

    // Integrate motion: update velocity, then position.
    float3 newVelocity = velocity + totalAcceleration  * deltaTime;
    float3 newPosition = position + newVelocity * deltaTime;

    // Store updated state back to global buffer.
    pos[particleId] = newPosition;
    vel[particleId] = newVelocity;
}
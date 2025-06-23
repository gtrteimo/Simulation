#include "cuda/grid.cuh"
#include "cuda/params.cuh"
#include "cuda/particles.cuh"
#include "cuda/util.cuh" // Ensure this is included

#include <cmath> // For floorf, ceilf
#include <algorithm> // For std::min, std::max in host functions if needed
#include <limits> // For std::numeric_limits<unsigned int>::max()
#include <cuda_runtime.h> // Include for make_float4 in host code


// Note: The actual sorting step using Thrust or CUB happens on the host
// *between* the two kernels declared here. This file only provides
// the kernels and the memory management/parameter calculation.


// --- Host Functions ---

__host__ void Grid_CalculateParams(GridData* grid_data, const SimulationParams* params) {
    if (!grid_data || !params) {
        fprintf(stderr, "Error: Null pointer passed to Grid_CalculateParams.\n");
        return;
    }

    // Use smoothingRadius as the grid cell size
    grid_data->gridCellSize = params->smoothingRadius;
    if (grid_data->gridCellSize <= 1e-6f) { // Prevent division by zero or tiny cells
         grid_data->gridCellSize = 1e-6f;
    }
    grid_data->invGridCellSize = 1.0f / grid_data->gridCellSize;

    // Store domain min for calculations
    // Now float4
    grid_data->domainMin = params->min;

    // Calculate domain extent (float4 subtraction) - MUST BE ELEMENT-WISE
    float4 domainExtent = make_float4(
        params->max.x - params->min.x,
        params->max.y - params->min.y,
        params->max.z - params->min.z,
        params->max.w - params->min.w
    );

    // Calculate grid dimensions (int4)
    // Apply ceilf element-wise
    grid_data->gridDimensions.x = static_cast<int>(ceilf(domainExtent.x * grid_data->invGridCellSize));
    grid_data->gridDimensions.y = static_cast<int>(ceilf(domainExtent.y * grid_data->invGridCellSize));
    grid_data->gridDimensions.z = static_cast<int>(ceilf(domainExtent.z * grid_data->invGridCellSize));
    grid_data->gridDimensions.w = static_cast<int>(ceilf(domainExtent.w * grid_data->invGridCellSize)); // Add 4th dimension

    // Ensure minimum dimension is 1 (apply std::max element-wise)
    grid_data->gridDimensions.x = std::max(1, grid_data->gridDimensions.x);
    grid_data->gridDimensions.y = std::max(1, grid_data->gridDimensions.y);
    grid_data->gridDimensions.z = std::max(1, grid_data->gridDimensions.z);
    grid_data->gridDimensions.w = std::max(1, grid_data->gridDimensions.w); // Add 4th dimension

    // Calculate total number of cells
    // WARNING: The product of 4 large dimensions can easily exceed unsigned int.
    // For this modification, we stick to unsigned int as per the original struct,
    // but be aware that for large 4D grids, numGridCells might overflow,
    // requiring unsigned long long and potentially changes in hash/indexing logic.
    unsigned long long totalCells_ull =
        static_cast<unsigned long long>(grid_data->gridDimensions.x) *
        static_cast<unsigned long long>(grid_data->gridDimensions.y) *
        static_cast<unsigned long long>(grid_data->gridDimensions.z) *
        static_cast<unsigned long long>(grid_data->gridDimensions.w);

    if (totalCells_ull > std::numeric_limits<unsigned int>::max()) {
         fprintf(stderr, "Warning: Calculated numGridCells (%llu) exceeds unsigned int maximum. Potential overflow!\n", totalCells_ull);
         // This will likely cause issues later with unsigned int arrays/indices.
         // Assigning a clipped value might prevent crashes but indicates a fundamental limit reached.
         grid_data->numGridCells = std::numeric_limits<unsigned int>::max();
    } else {
        grid_data->numGridCells = static_cast<unsigned int>(totalCells_ull);
    }


    // Basic check for potential zero cells (shouldn't happen with std::max(1, ...))
    if (grid_data->numGridCells == 0 && (grid_data->gridDimensions.x > 0 || grid_data->gridDimensions.y > 0 || grid_data->gridDimensions.z > 0 || grid_data->gridDimensions.w > 0)) {
         // This could indicate numGridCells overflowed unsigned int (handled above)
         // or an unexpected dimension calculation error.
         fprintf(stderr, "Warning: numGridCells calculated as 0 but dimensions are positive! Forcing to 1.\n");
         grid_data->numGridCells = 1; // Fallback to a single cell grid
    }
}

__host__ GridData* GridData_Host_Create(int numParticles, const SimulationParams* params) {
    GridData* h_grid_data = new GridData();
    if (!h_grid_data) {
        fprintf(stderr, "Failed to allocate host GridData struct.\n");
        return nullptr;
    }

    // Calculate Createial grid parameters
    Grid_CalculateParams(h_grid_data, params);

    // Note: Host pointers within the struct remain null as memory is on device
    h_grid_data->particle_hashes = nullptr;
    h_grid_data->particle_indices = nullptr;
    h_grid_data->cell_starts = nullptr;
    h_grid_data->cell_ends = nullptr;

    return h_grid_data;
}

__host__ void GridData_Host_Free(GridData* grid_data) {
    if (grid_data) {
        // Host_Free only frees the struct itself, not the device pointers it *held*.
        delete grid_data;
    }
}

__host__ GridData* GridData_Device_Create(int numParticles, const SimulationParams* params) {
    GridData* d_grid_data = nullptr;
    // Allocate the GridData struct on the device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data, sizeof(GridData)));

    // Calculate grid parameters on the host first
    GridData h_grid_params_temp;
    Grid_CalculateParams(&h_grid_params_temp, params);

    // Allocate device arrays based on numParticles and calculated numGridCells
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->particle_hashes, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->particle_indices, numParticles * sizeof(unsigned int)));
    // Allocation size depends on numGridCells, which comes from h_grid_params_temp
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->cell_starts, h_grid_params_temp.numGridCells * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->cell_ends, h_grid_params_temp.numGridCells * sizeof(unsigned int)));

    // Copy the calculated parameters (cellSize, dimensions, etc.) to the device struct
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->gridCellSize, &h_grid_params_temp.gridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->invGridCellSize, &h_grid_params_temp.invGridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    // domainMin is now float4
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->domainMin, &h_grid_params_temp.domainMin, sizeof(float4), cudaMemcpyHostToDevice));
    // gridDimensions is now int4
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->gridDimensions, &h_grid_params_temp.gridDimensions, sizeof(int4), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->numGridCells, &h_grid_params_temp.numGridCells, sizeof(unsigned int), cudaMemcpyHostToDevice));


    // Createialization of cell_starts/ends happens in the build grid step (e.g., Simulation_Device_BuildGrid),
    // typically using Thrust/CUB fill or a dedicated kernel, setting cell_starts to numParticles
    // and cell_ends to 0 *before* Grid_FindCellBoundsKernel runs.

    return d_grid_data;
}

__host__ void GridData_Device_Free(GridData* grid_data) {
    if (grid_data) {
        // Free device arrays (safe to call cudaFree with NULL)
        cudaFree(grid_data->particle_hashes);
        cudaFree(grid_data->particle_indices);
        cudaFree(grid_data->cell_starts);
        cudaFree(grid_data->cell_ends);

        // Free the device GridData struct itself
        cudaFree(grid_data);
    }
}

__host__ void GridData_CopyParamsToDevice(GridData* host_grid_data, GridData* device_grid_data) {
    if (!host_grid_data || !device_grid_data) {
        fprintf(stderr, "Error: Null pointer passed to GridData_CopyParamsToDevice.\n");
        return;
    }
     // Copy the calculated parameters (cellSize, dimensions, etc.) from host struct to device struct
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridCellSize, &host_grid_data->gridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->invGridCellSize, &host_grid_data->invGridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    // domainMin is now float4
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->domainMin, &host_grid_data->domainMin, sizeof(float4), cudaMemcpyHostToDevice));
    // gridDimensions is now int4
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridDimensions, &host_grid_data->gridDimensions, sizeof(int4), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->numGridCells, &host_grid_data->numGridCells, sizeof(unsigned int), cudaMemcpyHostToDevice));
}


// --- Grid Building Kernel Functions ---

__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem* ps,
    // Removed SimulationParams* params as domainMin is now in GridData
    GridData* grid_data
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps->numParticles) return;

    // Position is now float4
    float4 pos = ps->pos[i]; // Access pos directly from float4 array

    // Calculate cell coordinates (now returns int4)
    int4 cell_coords = Grid_GetCellCoords(pos, grid_data); // Pass pos (float4) and grid_data

    // Calculate hash (uses int4 cell_coords)
    unsigned int hash = Grid_GetHashFromCell(cell_coords, grid_data);

    // Store hash and original particle index
    grid_data->particle_hashes[i] = hash;
    grid_data->particle_indices[i] = i; // Store original index before sorting
}

__global__ void Grid_FindCellBoundsKernel(
    const unsigned int* d_sorted_particle_hashes,
    GridData* grid_data,
    int numParticles
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    unsigned int current_hash = d_sorted_particle_hashes[i];

    // Use atomic operations to update cell_starts and cell_ends
    // The logic here does not change as it operates on the scalar hash and particle index `i`.

    // Ensure the hash is valid (within the expected range [0, numGridCells-1])
    // Particles outside the domain (based on clamping behavior in GetCellCoords and GetHashFromCell)
    // might have a hash equal to numGridCells. We exclude these from boundary calculations.
    if (current_hash < grid_data->numGridCells) {

        // Update the start index for this hash: min of current value and thread index i
        atomicMin(&grid_data->cell_starts[current_hash], i);

        // Update the end index for this hash: max of current value and thread index i + 1
        atomicMax(&grid_data->cell_ends[current_hash], i + 1); // i+1 because cell_ends is exclusive
    }

    // Note: This kernel relies on cell_starts being Created to a value >= numParticles
    // and cell_ends being Created to 0 *before* this kernel runs.
    // This Creation should happen in the Simulation_Device_BuildGrid function
    // using a kernel launched over numGridCells or a Thrust/CUB fill.
}
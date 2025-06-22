#include "grid.cuh"
#include "params.cuh"
#include "particles.cuh"
#include "util.cuh" // Ensure this is included

#include <cmath> // For floorf
#include <algorithm> // For std::min, std::max in host functions if needed

// For CUDA memory management
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
    grid_data->domainMin = make_float3(params->min_x, params->min_y, params->min_z);

    // Calculate domain extent
    float3 domainExtent = make_float3(params->max_x - params->min_x,
                                      params->max_y - params->min_y,
                                      params->max_z - params->min_z);

    // Calculate grid dimensions
    // Need to handle cases where domainExtent is zero or negative
    grid_data->gridDimensions.x = static_cast<int>(ceilf(domainExtent.x * grid_data->invGridCellSize));
    grid_data->gridDimensions.y = static_cast<int>(ceilf(domainExtent.y * grid_data->invGridCellSize));
    grid_data->gridDimensions.z = static_cast<int>(ceilf(domainExtent.z * grid_data->invGridCellSize));

    // Ensure minimum dimension is 1
    grid_data->gridDimensions.x = std::max(1, grid_data->gridDimensions.x);
    grid_data->gridDimensions.y = std::max(1, grid_data->gridDimensions.y);
    grid_data->gridDimensions.z = std::max(1, grid_data->gridDimensions.z);

    // Calculate total number of cells
    grid_data->numGridCells = static_cast<unsigned int>(
        grid_data->gridDimensions.x * grid_data->gridDimensions.y * grid_data->gridDimensions.z
    );

    // Basic check for potential overflow if dimensions are huge (unlikely in typical SPH)
    if (grid_data->numGridCells == 0 && (grid_data->gridDimensions.x > 0 || grid_data->gridDimensions.y > 0 || grid_data->gridDimensions.z > 0)) {
         // This could indicate numGridCells overflowed unsigned int.
         // For typical SPH domains/grid sizes, this should not happen.
         // If it does, a different spatial hashing scheme or larger integer type might be needed.
         fprintf(stderr, "Warning: Potential overflow in numGridCells calculation!\n");
         grid_data->numGridCells = 1; // Fallback to a single cell grid
    }
}

__host__ GridData* GridData_Host_Init(int numParticles, const SimulationParams* params) {
    GridData* h_grid_data = new GridData();
    if (!h_grid_data) {
        fprintf(stderr, "Failed to allocate host GridData struct.\n");
        return nullptr;
    }

    // Calculate initial grid parameters
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

__host__ GridData* GridData_Device_Init(int numParticles, const SimulationParams* params) {
    GridData* d_grid_data = nullptr;
    // Allocate the GridData struct on the device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data, sizeof(GridData)));

    // Calculate grid parameters on the host first
    GridData h_grid_params_temp;
    Grid_CalculateParams(&h_grid_params_temp, params);

    // Allocate device arrays based on numParticles and calculated numGridCells
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->particle_hashes, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->particle_indices, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->cell_starts, h_grid_params_temp.numGridCells * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data->cell_ends, h_grid_params_temp.numGridCells * sizeof(unsigned int)));

    // Copy the calculated parameters (cellSize, dimensions, etc.) to the device struct
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->gridCellSize, &h_grid_params_temp.gridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->invGridCellSize, &h_grid_params_temp.invGridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->domainMin, &h_grid_params_temp.domainMin, sizeof(float3), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->gridDimensions, &h_grid_params_temp.gridDimensions, sizeof(int3), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->numGridCells, &h_grid_params_temp.numGridCells, sizeof(unsigned int), cudaMemcpyHostToDevice));


    // Initialize cell_starts and cell_ends arrays on the device.
    // cell_starts is initialized to numParticles (invalid index) to easily find unpopulated cells.
    // cell_ends is initialized to 0. After processing, the first particle index for a cell will update cell_starts,
    // and the last particle index + 1 will update cell_ends.
    // This initialization is done *before* Grid_FindCellBoundsKernel runs.
    // We need a kernel to do this initialization efficiently on the device.
    // Let's add a small kernel or handle this initialization within the build grid step.
    // For now, we'll rely on Grid_FindCellBoundsKernel's logic assuming some initialization
    // or handle the initialization in the main build grid function (Simulation_Device_BuildGrid).
    // A simple memset with a non-zero value for cell_starts is tricky. A kernel is better.
    // Let's assume a kernel or Thrust call handles the initialization in Simulation_Device_BuildGrid.
    // A common alternative: Grid_FindCellBoundsKernel could be launched with numGridCells threads,
    // initializing the cell_starts/ends for its cell index before processing particle hashes.
    // Let's update Grid_FindCellBoundsKernel to handle initialization itself. This requires
    // launching it with enough blocks/threads for numGridCells.

    // --- Re-evaluating Grid_FindCellBoundsKernel ---
    // The original kernel signature takes d_sorted_particle_hashes and operates per particle.
    // Initializing cell_starts/ends (size numGridCells) within this kernel is inefficient
    // as particle threads would clobber each other.
    // A separate kernel or Thrust/CUB fill operation is needed to initialize cell_starts/ends
    // *before* Grid_FindCellBoundsKernel runs.
    // Let's add a host function wrapper to handle this initialization after allocation.
    // This requires knowing numGridCells on the host after allocation, which we do via h_grid_params_temp.

    unsigned int numGridCells = h_grid_params_temp.numGridCells;
    unsigned int numParticles_u = static_cast<unsigned int>(numParticles);

    // Initialize cell_starts to numParticles and cell_ends to 0
    // Using cudaMemset is an option for 0, but not for numParticles.
    // A small kernel is best. Let's add an internal kernel for this.
    // Or, we can perform this initialization step within the main Simulation_Device_BuildGrid function.
    // Let's refine the plan: Simulation_Device_BuildGrid will:
    // 1. Call Grid_CalculateHashesKernel (per particle)
    // 2. Call Thrust sort (host)
    // 3. *Initialize* cell_starts/ends (e.g., kernel or Thrust fill per cell)
    // 4. Call Grid_FindCellBoundsKernel (per particle, using atomics)

    // Okay, allocation is done. The initialization will be handled in Simulation_Device_BuildGrid.

    return d_grid_data;
}

__host__ void GridData_Device_Free(GridData* grid_data) {
    if (grid_data) {
        // Free device arrays
        CHECK_CUDA_ERROR(cudaFree(grid_data->particle_hashes));
        CHECK_CUDA_ERROR(cudaFree(grid_data->particle_indices));
        CHECK_CUDA_ERROR(cudaFree(grid_data->cell_starts));
        CHECK_CUDA_ERROR(cudaFree(grid_data->cell_ends));

        // Free the device GridData struct itself
        CHECK_CUDA_ERROR(cudaFree(grid_data));
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
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->domainMin, &host_grid_data->domainMin, sizeof(float3), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridDimensions, &host_grid_data->gridDimensions, sizeof(int3), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->numGridCells, &host_grid_data->numGridCells, sizeof(unsigned int), cudaMemcpyHostToDevice));
}


// --- Grid Building Kernel Functions ---

__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem* ps,
    const SimulationParams* params,
    GridData* grid_data
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps->numParticles) return;

    // Get particle position
    float3 pos = make_float3(ps->pos_x[i], ps->pos_y[i], ps->pos_z[i]);

    // Calculate cell coordinates
    int3 cell_coords = Grid_GetCellCoords(pos, grid_data, params);

    // Calculate hash
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
    // For cell_starts, we want the *minimum* index for a given hash.
    // For cell_ends, we want the *maximum* index + 1 for a given hash.

    // The indices `i` here are the indices *within the sorted array*.
    // current_hash is in the range [0, numGridCells).
    // We need to write to grid_data->cell_starts[current_hash] and grid_data->cell_ends[current_hash].

    // Ensure the hash is valid (within the expected range [0, numGridCells-1])
    // Particles outside the domain might have a hash equal to numGridCells
    // if using the Grid_GetHashFromCell invalid return value.
    // We only want to process valid hashes for cell boundaries.
    if (current_hash < grid_data->numGridCells) {

        // Update the start index for this hash: min of current value and thread index i
        atomicMin(&grid_data->cell_starts[current_hash], i);

        // Update the end index for this hash: max of current value and thread index i + 1
        atomicMax(&grid_data->cell_ends[current_hash], i + 1); // i+1 because cell_ends is exclusive
    }

    // Note: This kernel relies on cell_starts being initialized to a value > numParticles
    // and cell_ends being initialized to 0 *before* this kernel runs.
    // This initialization should happen in the Simulation_Device_BuildGrid function.
}


/*
// Example of how to initialize cell_starts/ends using a kernel (or Thrust/CUB fill)
// This would typically be called from Simulation_Device_BuildGrid before Grid_FindCellBoundsKernel.
__global__ void Grid_InitCellBoundsKernel(GridData* grid_data) {
    unsigned int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= grid_data->numGridCells) return;

    // Initialize cell_starts with a value larger than any possible particle index
    // (e.g., numParticles, although numParticles isn't directly available here)
    // A value like UINT_MAX works if you know your grid won't hit max int.
    // Using numGridCells *must* be > any valid particle index (0 to numParticles-1) if numParticles <= numGridCells.
    // A robust way is to pass numParticles or use the fact that valid particle indices < numParticles.
    // A simpler way is to initialize with a value guaranteed to be overwritten by any valid index,
    // or use a separate kernel launched with numParticles threads that only updates the first encountered start.

    // Let's assume numParticles was passed to the wrapper launching this kernel.
    // For this example, we'll use numGridCells itself as a sentinel, assuming numParticles < numGridCells.
    // A safer sentinel is UINT_MAX or passing the actual numParticles.
    // Let's modify Simulation_Device_BuildGrid logic instead to handle initialization properly.

    // However, if we were to initialize with a kernel launched across cells:
    // grid_data->cell_starts[cell_idx] = some_sentinel_value; // e.g. UINT_MAX or numParticles
    // grid_data->cell_ends[cell_idx] = 0;
}
*/



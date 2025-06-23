#pragma once

#include "cuda/params.cuh"    // For SimulationParams and domain bounds
#include "cuda/particles.cuh" // For ParticleSystem (mainly numParticles)
#include "cuda/util.cuh"      // For CHECK_CUDA_ERROR, M_PI (indirectly)
#include <cuda_runtime.h>     // For float4, int4, make_float4, make_int4, etc.
#include <cmath>              // For floorf, ceilf
#include <algorithm>          // For std::max (in host functions)

struct GridData {
    // Device pointers for grid structure
    unsigned int* particle_hashes;   // Hash value for each particle
    unsigned int* particle_indices;  // Original particle index, sorted according to hash
    unsigned int* cell_starts;       // Start index (inclusive) in sorted_particle_indices for each cell
    unsigned int* cell_ends;         // End index (exclusive) in sorted_particle_indices for each cell

    // Grid parameters - typically calculated on host, copied to device
    float gridCellSize;             // Typically >= smoothingRadius
    float invGridCellSize;          // 1.0f / gridCellSize
    float4 domainMin;               // Minimum corner of the simulation domain (from SimulationParams)
    int4 gridDimensions;            // Number of grid cells in x, y, z, w
    unsigned int numGridCells;      // Product of gridDimensions
};

// --- Host Functions ---

// Calculates grid parameters (cell size, dimensions) based on simulation domain and smoothing radius.
// Should be called *before* GridData_Host_Create or GridData_Device_Create if params change.
// Populates grid_data->gridCellSize, etc.
__host__ void Grid_CalculateParams(GridData* grid_data, const SimulationParams* params);

// Createializes host-side representation of GridData struct.
// Note: The pointers (particle_hashes, etc.) are expected to be device pointers managed separately.
// This function primarily allocates the GridData struct itself and calculates params.
__host__ GridData* GridData_Host_Create(int numParticles, const SimulationParams* params);

// Frees host-side GridData struct. Does NOT free device memory.
__host__ void GridData_Host_Free(GridData* grid_data);

// --- Device Functions ---

// Createializes device-side GridData struct and allocates device arrays.
__host__ GridData* GridData_Device_Create(int numParticles, const SimulationParams* params);

// Frees device-side GridData struct and allocated device arrays.
__host__ void GridData_Device_Free(GridData* grid_data);

// Utility function to copy host-calculated grid parameters to the device GridData struct.
__host__ void GridData_CopyParamsToDevice(GridData* host_grid_data, GridData* device_grid_data);


// --- Utility Device Functions ---
// These functions are typically called from inside kernels.

// __device__ inline helper function to get grid cell coordinates from a position.
// Returns int4 with coordinates clamped to grid dimensions.
__device__ inline int4 Grid_GetCellCoords(float4 pos, const GridData* grid_data) {
    // Shift position by domain minimum - MUST BE ELEMENT-WISE
    float4 shifted_pos = make_float4(
        pos.x - grid_data->domainMin.x,
        pos.y - grid_data->domainMin.y,
        pos.z - grid_data->domainMin.z,
        pos.w - grid_data->domainMin.w
    );

    // Calculate floating point cell coordinates (element-wise multiplication by scalar) - MUST BE ELEMENT-WISE
    float4 cell_f = make_float4(
        shifted_pos.x * grid_data->invGridCellSize,
        shifted_pos.y * grid_data->invGridCellSize,
        shifted_pos.z * grid_data->invGridCellSize,
        shifted_pos.w * grid_data->invGridCellSize
    );

    // Floor to get integer cell coordinates (element-wise floor)
    int4 cell_i = make_int4(static_cast<int>(floorf(cell_f.x)),
                            static_cast<int>(floorf(cell_f.y)),
                            static_cast<int>(floorf(cell_f.z)),
                            static_cast<int>(floorf(cell_f.w)));

    // Clamp cell coordinates to grid dimensions
    // This handles particles slightly outside the defined domain due to floating point inaccuracies,
    // or particles that might be pushed temporarily outside boundary conditions.
    // Use CUDA's built-in max/min
    cell_i.x = max(0, min(grid_data->gridDimensions.x - 1, cell_i.x));
    cell_i.y = max(0, min(grid_data->gridDimensions.y - 1, cell_i.y));
    cell_i.z = max(0, min(grid_data->gridDimensions.z - 1, cell_i.z));
    cell_i.w = max(0, min(grid_data->gridDimensions.w - 1, cell_i.w));


    return cell_i;
}

// __device__ inline helper function to calculate a spatial hash from cell coordinates.
// Simple 4D to 1D mapping.
__device__ inline unsigned int Grid_GetHashFromCell(int4 cell_coords, const GridData* grid_data) {
    // Ensure coordinates are non-negative and within bounds (should be handled by GetCellCoords, but defensive)
    // Note: Grid_GetCellCoords clamps, so this check might be redundant if only called after clamping.
    // However, keeping it makes this function more robust if called with raw coords.
    // We use numGridCells as a sentinel for invalid cells.
    if (cell_coords.x < 0 || cell_coords.y < 0 || cell_coords.z < 0 || cell_coords.w < 0 ||
        cell_coords.x >= grid_data->gridDimensions.x ||
        cell_coords.y >= grid_data->gridDimensions.y ||
        cell_coords.z >= grid_data->gridDimensions.z ||
        cell_coords.w >= grid_data->gridDimensions.w)
    {
         return grid_data->numGridCells; // Return an 'invalid' hash outside the valid range [0, numGridCells-1]
    }

    // Simple spatial hash: w * Dx*Dy*Dz + z * Dx*Dy + y * Dx + x
    // Assuming linear indexing matches gridDimensions order (W then Z then Y then X changing fastest)
    // Need to use unsigned int for dimensions in calculation to avoid overflow on intermediate products IF dimensions are large.
    // However, gridDimensions is int4. Casting is necessary.
    // WARNING: Product of gridDimensions can easily exceed unsigned int if dimensions are large (e.g., 100x100x100x100 ~ 10^8 fits, 1000x1000x1000x1000 ~ 10^12 does not).
    // For extremely large 4D grids, unsigned long long might be needed for hash and numGridCells.
    // Assuming for this modification that unsigned int is sufficient for the specific problem scale.
    unsigned int Dx = static_cast<unsigned int>(grid_data->gridDimensions.x);
    unsigned int Dy = static_cast<unsigned int>(grid_data->gridDimensions.y);
    unsigned int Dz = static_cast<unsigned int>(grid_data->gridDimensions.z);
    //unsigned int Dw = static_cast<unsigned int>(grid_data->gridDimensions.w); // Not needed for calc order below

    unsigned int hash = static_cast<unsigned int>(
        static_cast<unsigned int>(cell_coords.w) * Dx * Dy * Dz +
        static_cast<unsigned int>(cell_coords.z) * Dx * Dy +
        static_cast<unsigned int>(cell_coords.y) * Dx +
        static_cast<unsigned int>(cell_coords.x)
    );

    // The hash must be less than numGridCells if the cell is valid and numGridCells = Dx*Dy*Dz*Dw.
    // Return value is in the range [0, numGridCells-1] for valid cells.
    return hash;
}

// __device__ inline helper function to get linear cell index from cell coordinates.
__device__ inline unsigned int Grid_GetLinearCellIndex(int4 cell_coords, const GridData* grid_data) {
     // Same as hash calculation for this simple mapping
     return Grid_GetHashFromCell(cell_coords, grid_data);
}

// __device__ inline helper function to get grid coordinates from a linear cell index.
__device__ inline int4 Grid_GetGridCoordsFromLinearIndex(unsigned int linear_index, const GridData* grid_data) {
    int4 coords;
    unsigned int Dx = static_cast<unsigned int>(grid_data->gridDimensions.x);
    unsigned int Dy = static_cast<unsigned int>(grid_data->gridDimensions.y);
    unsigned int Dz = static_cast<unsigned int>(grid_data->gridDimensions.z);
    //unsigned int Dw = static_cast<unsigned int>(grid_data->gridDimensions.w); // Not needed for calculation

    coords.w = linear_index / (Dx * Dy * Dz);
    unsigned int remainder_w = linear_index % (Dx * Dy * Dz);
    coords.z = remainder_w / (Dx * Dy);
    unsigned int remainder_z = remainder_w % (Dx * Dy);
    coords.y = remainder_z / Dx;
    coords.x = remainder_z % Dx;

    return coords;
}


// --- Grid Building Kernel Functions ---

// Kernel to calculate a spatial hash for each particle based on its position.
__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem* ps,        // For particle positions (device pointer)
    // Removed SimulationParams* params as domainMin is now in GridData
    GridData* grid_data              // For writing particle_hashes and particle_indices (device pointer)
);


// Kernel to determine the start and end indices in the sorted list for each grid cell.
// This populates cell_starts and cell_ends in GridData.
// Assumes particle_hashes and particle_indices are already sorted.
__global__ void Grid_FindCellBoundsKernel(
    const unsigned int* d_sorted_particle_hashes, // Input: sorted particle hashes (device pointer)
    GridData* grid_data,              // For writing cell_starts and cell_ends (device pointer)
    int numParticles                  // Total number of particles
);

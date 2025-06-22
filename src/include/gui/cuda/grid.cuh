#pragma once

#include "params.cuh"    // For SimulationParams and domain bounds
#include "particles.cuh" // For ParticleSystem (mainly numParticles)
#include "util.cuh"      // For CHECK_CUDA_ERROR, M_PI (indirectly)
#include <vector_types.h> // For float3, int3

struct GridData {
    // Device pointers for grid structure
    unsigned int* particle_hashes;   // Hash value for each particle
    unsigned int* particle_indices;  // Original particle index, sorted according to hash
    unsigned int* cell_starts;       // Start index (inclusive) in sorted_particle_indices for each cell
    unsigned int* cell_ends;         // End index (exclusive) in sorted_particle_indices for each cell

    // Grid parameters - typically calculated on host, copied to device
    float gridCellSize;             // Typically >= smoothingRadius
    float invGridCellSize;          // 1.0f / gridCellSize
    float3 domainMin;               // Minimum corner of the simulation domain (from SimulationParams)
    int3 gridDimensions;            // Number of grid cells in x, y, z
    unsigned int numGridCells;      // Product of gridDimensions
};

// --- Host Functions ---

// Calculates grid parameters (cell size, dimensions) based on simulation domain and smoothing radius.
// Should be called *before* GridData_Host_Init or GridData_Device_Init if params change.
// Populates grid_data->gridCellSize, etc.
__host__ void Grid_CalculateParams(GridData* grid_data, const SimulationParams* params);

// Initializes host-side representation of GridData struct.
// Note: The pointers (particle_hashes, etc.) are expected to be device pointers managed separately.
// This function primarily allocates the GridData struct itself and calculates params.
__host__ GridData* GridData_Host_Init(int numParticles, const SimulationParams* params);

// Frees host-side GridData struct. Does NOT free device memory.
__host__ void GridData_Host_Free(GridData* grid_data);

// --- Device Functions ---

// Initializes device-side GridData struct and allocates device arrays.
__host__ GridData* GridData_Device_Init(int numParticles, const SimulationParams* params);

// Frees device-side GridData struct and allocated device arrays.
__host__ void GridData_Device_Free(GridData* grid_data);

// Utility function to copy host-calculated grid parameters to the device GridData struct.
__host__ void GridData_CopyParamsToDevice(GridData* host_grid_data, GridData* device_grid_data);


// --- Utility Device Functions ---
// These functions are typically called from inside kernels.

// __device__ inline helper function to get grid cell coordinates from a position.
// Returns int3(-1,-1,-1) or clamped coordinates if outside domain.
__device__ inline int3 Grid_GetCellCoords(float3 pos, const GridData* grid_data, const SimulationParams* params) {
    // Shift position by domain minimum
    float3 shifted_pos = make_float3(pos.x - params->min_x, pos.y - params->min_y, pos.z - params->min_z);

    // Calculate floating point cell coordinates
    float3 cell_f = make_float3(shifted_pos.x * grid_data->invGridCellSize,
                                shifted_pos.y * grid_data->invGridCellSize,
                                shifted_pos.z * grid_data->invGridCellSize);

    // Floor to get integer cell coordinates
    int3 cell_i = make_int3(static_cast<int>(floorf(cell_f.x)),
                            static_cast<int>(floorf(cell_f.y)),
                            static_cast<int>(floorf(cell_f.z)));

    // Clamp cell coordinates to grid dimensions
    // This handles particles slightly outside the defined domain due to floating point inaccuracies,
    // or particles that might be pushed temporarily outside boundary conditions.
    cell_i.x = max(0, min(grid_data->gridDimensions.x - 1, cell_i.x));
    cell_i.y = max(0, min(grid_data->gridDimensions.y - 1, cell_i.y));
    cell_i.z = max(0, min(grid_data->gridDimensions.z - 1, cell_i.z));

    return cell_i;
}

// __device__ inline helper function to calculate a spatial hash from cell coordinates.
// Simple 3D to 1D mapping, possibly using modulo for limited grid size if needed (though
// linear mapping is simpler if grid dimensions are within hash range).
__device__ inline unsigned int Grid_GetHashFromCell(int3 cell_coords, const GridData* grid_data) {
    // Ensure coordinates are non-negative and within bounds (should be handled by GetCellCoords, but defensive)
    if (cell_coords.x < 0 || cell_coords.y < 0 || cell_coords.z < 0 ||
        cell_coords.x >= grid_data->gridDimensions.x ||
        cell_coords.y >= grid_data->gridDimensions.y ||
        cell_coords.z >= grid_data->gridDimensions.z)
    {
         return grid_data->numGridCells; // Return an 'invalid' hash or a hash outside the valid range [0, numGridCells-1]
                                         // Using numGridCells as an invalid hash value means it will sort last.
    }

    // Simple spatial hash: cell_z * Dx * Dy + cell_y * Dx + cell_x
    // Assuming linear indexing matches gridDimensions order (Z then Y then X changing fastest)
    unsigned int hash = static_cast<unsigned int>(
        cell_coords.z * grid_data->gridDimensions.x * grid_data->gridDimensions.y +
        cell_coords.y * grid_data->gridDimensions.x +
        cell_coords.x
    );

    // The hash *must* be less than numGridCells if the cell is valid.
    // If numGridCells is the product Dx*Dy*Dz, this will be the case for valid cells.
    // Return value is in the range [0, numGridCells-1] for valid cells.
    return hash;
}

// __device__ inline helper function to get linear cell index from cell coordinates.
__device__ inline unsigned int Grid_GetLinearCellIndex(int3 cell_coords, const GridData* grid_data) {
     // Same as hash calculation for this simple mapping
     return Grid_GetHashFromCell(cell_coords, grid_data);
}

// __device__ inline helper function to get grid coordinates from a linear cell index.
__device__ inline int3 Grid_GetGridCoordsFromLinearIndex(unsigned int linear_index, const GridData* grid_data) {
    int3 coords;
    unsigned int Dx = grid_data->gridDimensions.x;
    unsigned int Dy = grid_data->gridDimensions.y;
    //unsigned int Dz = grid_data->gridDimensions.z; // Not needed for calculation

    coords.z = linear_index / (Dx * Dy);
    unsigned int remainder = linear_index % (Dx * Dy);
    coords.y = remainder / Dx;
    coords.x = remainder % Dx;

    return coords;
}


// --- Grid Building Kernel Functions ---

// Kernel to calculate a spatial hash for each particle based on its position.
__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem* ps,        // For particle positions (device pointer)
    const SimulationParams* params,  // For simulation domain (device pointer)
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


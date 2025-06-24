#pragma once

#include "types/simulationTypes.h" // For GridData, ParticleSystem, SimulationParams
#include "cuda/util.cuh"      // For CHECK_CUDA_ERROR

// --- Host Functions ---

__host__ void Grid_CalculateParams(GridData* grid_data, const SimulationParams* params);
__host__ GridData* GridData_CreateOnHost(int numParticles, const SimulationParams* params);
__host__ void GridData_FreeOnHost(GridData* grid_data); // Frees host struct only

// --- Device Functions ---

__host__ GridData* GridData_CreateOnDevice(int numParticles, const SimulationParams* params); // Allocates device memory and copies parameters
__host__ void GridData_Device_Free(GridData* grid_data); // Frees device memory
__host__ void GridData_CopyParamsToDevice(GridData* host_grid_data, GridData* device_grid_data); // Copies host params to device struct

// --- Utility Device Functions ---

__device__ inline int4 Grid_GetCellCoords(float4 pos, const GridData* grid_data) {
    float4 shifted_pos = make_float4(
        pos.x - grid_data->domainMin.x,
        pos.y - grid_data->domainMin.y,
        pos.z - grid_data->domainMin.z,
        pos.w - grid_data->domainMin.w
    );

    float4 cell_f = make_float4(
        shifted_pos.x * grid_data->invGridCellSize,
        shifted_pos.y * grid_data->invGridCellSize,
        shifted_pos.z * grid_data->invGridCellSize,
        shifted_pos.w * grid_data->invGridCellSize
    );

    int4 cell_i = make_int4(static_cast<int>(floorf(cell_f.x)),
                            static_cast<int>(floorf(cell_f.y)),
                            static_cast<int>(floorf(cell_f.z)),
                            static_cast<int>(floorf(cell_f.w)));

    cell_i.x = max(0, min(grid_data->gridDimensions.x - 1, cell_i.x));
    cell_i.y = max(0, min(grid_data->gridDimensions.y - 1, cell_i.y));
    cell_i.z = max(0, min(grid_data->gridDimensions.z - 1, cell_i.z));
    cell_i.w = max(0, min(grid_data->gridDimensions.w - 1, cell_i.w));

    return cell_i;
}

__device__ inline unsigned int Grid_GetHashFromCell(int4 cell_coords, const GridData* grid_data) {
    if (cell_coords.x < 0 || cell_coords.y < 0 || cell_coords.z < 0 || cell_coords.w < 0 ||
        cell_coords.x >= grid_data->gridDimensions.x ||
        cell_coords.y >= grid_data->gridDimensions.y ||
        cell_coords.z >= grid_data->gridDimensions.z ||
        cell_coords.w >= grid_data->gridDimensions.w)
    {
         return grid_data->numGridCells; // Sentinel for invalid cells
    }

    // Linear index (hash) based on X then Y then Z then W changing slowest
    unsigned int Dx = static_cast<unsigned int>(grid_data->gridDimensions.x);
    unsigned int Dy = static_cast<unsigned int>(grid_data->gridDimensions.y);
    unsigned int Dz = static_cast<unsigned int>(grid_data->gridDimensions.z);

    unsigned int hash = static_cast<unsigned int>(
        static_cast<unsigned int>(cell_coords.w) * Dx * Dy * Dz +
        static_cast<unsigned int>(cell_coords.z) * Dx * Dy +
        static_cast<unsigned int>(cell_coords.y) * Dx +
        static_cast<unsigned int>(cell_coords.x)
    );

    return hash; // In range [0, numGridCells-1] for valid cells
}

__device__ inline unsigned int Grid_GetLinearCellIndex(int4 cell_coords, const GridData* grid_data) {
     return Grid_GetHashFromCell(cell_coords, grid_data); // Same as hash for this mapping
}

__device__ inline int4 Grid_GetGridCoordsFromLinearIndex(unsigned int linear_index, const GridData* grid_data) {
    int4 coords;
    unsigned int Dx = static_cast<unsigned int>(grid_data->gridDimensions.x);
    unsigned int Dy = static_cast<unsigned int>(grid_data->gridDimensions.y);
    unsigned int Dz = static_cast<unsigned int>(grid_data->gridDimensions.z);

    coords.w = linear_index / (Dx * Dy * Dz);
    unsigned int remainder_w = linear_index % (Dx * Dy * Dz);
    coords.z = remainder_w / (Dx * Dy);
    unsigned int remainder_z = remainder_w % (Dx * Dy);
    coords.y = remainder_z / Dx;
    coords.x = remainder_z % Dx;

    return coords;
}

// --- Grid Building Kernel Prototypes ---

__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem* ps,
    GridData* grid_data
);

__global__ void Grid_FindCellBoundsKernel(
    const unsigned int* d_sorted_particle_hashes,
    GridData* grid_data,
    int numParticles
);

// --- Initialization Kernels (Called from CreateOnDevice) ---

// Initializes cell_starts array (typically to numParticles or UINT_MAX)
__global__ void Grid_InitCellStartsKernel(
    unsigned int* d_cell_starts,
    unsigned int numGridCells,
    unsigned int init_value
);

// Initializes cell_ends array (typically to 0)
__global__ void Grid_InitCellEndsKernel(
    unsigned int* d_cell_ends,
    unsigned int numGridCells,
    unsigned int init_value // Should be 0
);
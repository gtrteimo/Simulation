#include "cuda/grid.cuh"
#include "cuda/params.cuh"
#include "cuda/particles.cuh"
#include "cuda/util.cuh"
#include <cmath>
#include <algorithm> // For std::max
#include <cstdio>    // For fprintf, stderr
#include <limits>    // For std::numeric_limits

// --- Host Functions ---

__host__ void Grid_CalculateParams(GridData* grid_data, const SimulationParams* params) {
    if (!grid_data || !params) {
        fprintf(stderr, "Error: Null pointer passed to Grid_CalculateParams.\n");
        return;
    }

    grid_data->gridCellSize = params->smoothingRadius;
    if (grid_data->gridCellSize <= 1e-6f) {
         grid_data->gridCellSize = 1e-6f;
    }
    grid_data->invGridCellSize = 1.0f / grid_data->gridCellSize;

    grid_data->domainMin = params->min;

    float4 domainExtent = make_float4(
        params->max.x - params->min.x,
        params->max.y - params->min.y,
        params->max.z - params->min.z,
        params->max.w - params->min.w
    );

    grid_data->gridDimensions.x = std::max(1, static_cast<int>(ceilf(domainExtent.x * grid_data->invGridCellSize)));
    grid_data->gridDimensions.y = std::max(1, static_cast<int>(ceilf(domainExtent.y * grid_data->invGridCellSize)));
    grid_data->gridDimensions.z = std::max(1, static_cast<int>(ceilf(domainExtent.z * grid_data->invGridCellSize)));
    grid_data->gridDimensions.w = std::max(1, static_cast<int>(ceilf(domainExtent.w * grid_data->invGridCellSize)));

    unsigned long long totalCells_ull =
        static_cast<unsigned long long>(grid_data->gridDimensions.x) *
        static_cast<unsigned long long>(grid_data->gridDimensions.y) *
        static_cast<unsigned long long>(grid_data->gridDimensions.z) *
        static_cast<unsigned long long>(grid_data->gridDimensions.w);

    if (totalCells_ull > std::numeric_limits<unsigned int>::max()) {
         fprintf(stderr, "Warning: Calculated numGridCells (%llu) exceeds unsigned int maximum (%u). Potential overflow in indices/hashes!\n",
                 totalCells_ull, std::numeric_limits<unsigned int>::max());
         grid_data->numGridCells = std::numeric_limits<unsigned int>::max(); // Assign clipped value
    } else {
        grid_data->numGridCells = static_cast<unsigned int>(totalCells_ull);
    }

     if (grid_data->numGridCells == 0) { // Should not happen if dimensions >= 1
         fprintf(stderr, "Warning: numGridCells calculated as 0! Forcing to 1.\n");
         grid_data->numGridCells = 1;
     }
}

__host__ GridData* GridData_CreateOnHost(int numParticles, const SimulationParams* params) {
    GridData* h_grid_data = new GridData();
    if (!h_grid_data) {
        fprintf(stderr, "Failed to allocate host GridData struct.\n");
        return nullptr;
    }

    Grid_CalculateParams(h_grid_data, params);

    h_grid_data->particle_hashes = nullptr;
    h_grid_data->particle_indices = nullptr;
    h_grid_data->cell_starts = nullptr;
    h_grid_data->cell_ends = nullptr;

    return h_grid_data;
}

__host__ void GridData_FreeOnHost(GridData* grid_data) {
    if (grid_data) {
        // Only free the host struct itself. Device memory must be freed by GridData_Device_Free.
        delete grid_data; // Using delete since it was allocated with new
    }
}

__host__ GridData* GridData_CreateOnDevice(int numParticles, const SimulationParams* params) {
    GridData* d_grid_data = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grid_data, sizeof(GridData)));

    GridData h_grid_params_temp;
    Grid_CalculateParams(&h_grid_params_temp, params);

    // Allocate device arrays
    unsigned int* d_particle_hashes;
    unsigned int* d_particle_indices;
    unsigned int* d_cell_starts;
    unsigned int* d_cell_ends;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_particle_hashes, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_particle_indices, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cell_starts, h_grid_params_temp.numGridCells * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cell_ends, h_grid_params_temp.numGridCells * sizeof(unsigned int)));

    // Copy calculated parameters to the device struct
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->gridCellSize, &h_grid_params_temp.gridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->invGridCellSize, &h_grid_params_temp.invGridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->domainMin, &h_grid_params_temp.domainMin, sizeof(float4), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->gridDimensions, &h_grid_params_temp.gridDimensions, sizeof(int4), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&d_grid_data->numGridCells, &h_grid_params_temp.numGridCells, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Copy the device array pointers to the device struct
    // NOTE: Need a separate cudaMemcpy or kernel launch to set these pointers *on the device*
    // This requires launching a kernel or a host-to-device memcpy of the *pointers*.
    // The easiest is a host-to-device memcpy of the d_grid_data struct *after* setting pointers on the host copy.
    // Let's allocate h_grid_data first, set its pointers, then copy the whole struct.

    // Revised approach: Allocate host struct, calculate params, allocate device arrays, set pointers on host struct, memcpy host struct to device struct.
    GridData* h_grid_data_temp = new GridData();
    if (!h_grid_data_temp) {
        cudaFree(d_particle_hashes); cudaFree(d_particle_indices); cudaFree(d_cell_starts); cudaFree(d_cell_ends);
        cudaFree(d_grid_data);
        fprintf(stderr, "Failed to allocate host temporary GridData struct.\n");
        return nullptr;
    }
    Grid_CalculateParams(h_grid_data_temp, params);

    // Allocate device arrays
    CHECK_CUDA_ERROR(cudaMalloc((void**)&h_grid_data_temp->particle_hashes, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&h_grid_data_temp->particle_indices, numParticles * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&h_grid_data_temp->cell_starts, h_grid_data_temp->numGridCells * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&h_grid_data_temp->cell_ends, h_grid_data_temp->numGridCells * sizeof(unsigned int)));

    // Copy the *entire* host struct (containing calculated params and device pointers) to the device
    CHECK_CUDA_ERROR(cudaMemcpy(d_grid_data, h_grid_data_temp, sizeof(GridData), cudaMemcpyHostToDevice));

    // Initialize cell_starts (to numParticles) and cell_ends (to 0) on the device
    // Required before Grid_FindCellBoundsKernel
    unsigned int numGridCells = h_grid_data_temp->numGridCells;
    dim3 init_blocks( (numGridCells + 255) / 256 ); // Simple block calculation
    dim3 init_threads(256);

    Grid_InitCellStartsKernel<<<init_blocks, init_threads>>>(h_grid_data_temp->cell_starts, numGridCells, numParticles); // Use numParticles as sentinel
    Grid_InitCellEndsKernel<<<init_blocks, init_threads>>>(h_grid_data_temp->cell_ends, numGridCells, 0);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors launching kernels

    // The host struct h_grid_data_temp and its pointers are temporary.
    // The device pointers are stored in d_grid_data now.
    // We don't free the pointers in h_grid_data_temp as they point to device memory.
    delete h_grid_data_temp; // Free the host struct itself

    return d_grid_data;
}

__host__ void GridData_Device_Free(GridData* grid_data) {
    if (grid_data) {
        // Need to copy the struct back to host to get the device pointers
        GridData h_grid_data_ptrs;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_grid_data_ptrs, grid_data, sizeof(GridData), cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.particle_hashes));
        CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.particle_indices));
        CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.cell_starts));
        CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.cell_ends));

        CHECK_CUDA_ERROR(cudaFree(grid_data)); // Free the struct itself on device
    }
}

__host__ void GridData_CopyParamsToDevice(GridData* host_grid_data, GridData* device_grid_data) {
    if (!host_grid_data || !device_grid_data) {
        fprintf(stderr, "Error: Null pointer passed to GridData_CopyParamsToDevice.\n");
        return;
    }
    // Copies parameters *only*. Does not update the device array pointers if they changed.
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridCellSize, &host_grid_data->gridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->invGridCellSize, &host_grid_data->invGridCellSize, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->domainMin, &host_grid_data->domainMin, sizeof(float4), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridDimensions, &host_grid_data->gridDimensions, sizeof(int4), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->numGridCells, &host_grid_data->numGridCells, sizeof(unsigned int), cudaMemcpyHostToDevice));
}


// --- Grid Building Kernel Functions ---

__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem* ps,
    GridData* grid_data
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps->numParticles) return;

    float4 pos = ps->pos[i];

    int4 cell_coords = Grid_GetCellCoords(pos, grid_data);
    unsigned int hash = Grid_GetHashFromCell(cell_coords, grid_data);

    grid_data->particle_hashes[i] = hash;
    grid_data->particle_indices[i] = i;
}

__global__ void Grid_FindCellBoundsKernel(
    const unsigned int* d_sorted_particle_hashes,
    GridData* grid_data,
    int numParticles
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    unsigned int current_hash = d_sorted_particle_hashes[i];

    if (current_hash < grid_data->numGridCells) {
        // Use atomic operations to update cell_starts and cell_ends
        // cell_starts initialized to numParticles, cell_ends initialized to 0
        atomicMin(&grid_data->cell_starts[current_hash], i);
        atomicMax(&grid_data->cell_ends[current_hash], i + 1); // i+1 for exclusive end
    }
}

// --- Initialization Kernels ---

__global__ void Grid_InitCellStartsKernel(
    unsigned int* d_cell_starts,
    unsigned int numGridCells,
    unsigned int init_value
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numGridCells) {
        d_cell_starts[i] = init_value;
    }
}

__global__ void Grid_InitCellEndsKernel(
    unsigned int* d_cell_ends,
    unsigned int numGridCells,
    unsigned int init_value
) {
     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numGridCells) {
        d_cell_ends[i] = init_value;
    }
}
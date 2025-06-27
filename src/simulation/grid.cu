#include "simulation/grid.cuh"
#include "simulation/params.cuh"
#include "simulation/particles.cuh"
#include "simulation/util.cuh"
#include "types/simulationTypes.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// --- Host Functions ---

__host__ void Grid_CalculateParams(GridData *grid_data, const SimulationParams *params) {
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
	    params->max.w - params->min.w);

	grid_data->gridDimensions.x = std::max(1, static_cast<int>(ceilf(domainExtent.x * grid_data->invGridCellSize)));
	grid_data->gridDimensions.y = std::max(1, static_cast<int>(ceilf(domainExtent.y * grid_data->invGridCellSize)));
	grid_data->gridDimensions.z = std::max(1, static_cast<int>(ceilf(domainExtent.z * grid_data->invGridCellSize)));
	grid_data->gridDimensions.w = std::max(1, static_cast<int>(ceilf(domainExtent.w * grid_data->invGridCellSize)));

	grid_data->numGridCells =
	    static_cast<unsigned long long>(grid_data->gridDimensions.x) *
	    static_cast<unsigned long long>(grid_data->gridDimensions.y) *
	    static_cast<unsigned long long>(grid_data->gridDimensions.z) *
	    static_cast<unsigned long long>(grid_data->gridDimensions.w);

	if (grid_data->numGridCells == 0) { // Should not happen if dimensions >= 1
		fprintf(stderr, "Warning: numGridCells calculated as 0! Forcing to 1.\n");
		grid_data->numGridCells = 1;
	}
}

__host__ GridData *GridData_CreateOnHost(int numParticles, const SimulationParams *params) {
	GridData *h_grid_data = (GridData *)malloc(sizeof(GridData));
	if (!h_grid_data) {
		fprintf(stderr, "Failed to allocate host GridData struct.\n");
		exit(-1);
	}

	Grid_CalculateParams(h_grid_data, params);

	h_grid_data->particle_hashes = (unsigned long long *)malloc(numParticles * sizeof(unsigned long long));
	h_grid_data->particle_indices = (unsigned int *)malloc(numParticles * sizeof(unsigned int));
	h_grid_data->cell_starts = (unsigned int *)malloc(h_grid_data->numGridCells * sizeof(unsigned int));
	h_grid_data->cell_ends = (unsigned int *)malloc(h_grid_data->numGridCells * sizeof(unsigned int));

	return h_grid_data;
}

__host__ void GridData_FreeOnHost(GridData *grid_data) {
	if (grid_data) {
		if (grid_data->particle_hashes) {
			free(grid_data->particle_hashes);
		}
		if (grid_data->particle_indices) {
			free(grid_data->particle_indices);
		}
		if (grid_data->cell_starts) {
			free(grid_data->cell_starts);
		}
		if (grid_data->cell_ends) {
			free(grid_data->cell_ends);
		}
		free(grid_data);
	}
}

__host__ GridData *GridData_CreateOnDevice(int numParticles, const SimulationParams *params) {

	GridData *d_grid_data = nullptr;
	GridData h_grid_data;

	// Step 1: Allocate space for the main struct on the device
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_grid_data, sizeof(GridData)));

	// Step 2 & 3: Calculate parameters into the host-side struct
	Grid_CalculateParams(&h_grid_data, params);

	// Step 4: Allocate all device arrays, storing pointers in the host struct
	CHECK_CUDA_ERROR(cudaMalloc((void **)&h_grid_data.particle_hashes, numParticles * sizeof(unsigned long long)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&h_grid_data.particle_indices, numParticles * sizeof(unsigned int)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&h_grid_data.cell_starts, h_grid_data.numGridCells * sizeof(unsigned int)));
	CHECK_CUDA_ERROR(cudaMalloc((void **)&h_grid_data.cell_ends, h_grid_data.numGridCells * sizeof(unsigned int)));

	// Step 5: Initialize the device arrays directly, avoiding custom kernels.
	// Initialize cell_ends to 0 (the starting value for atomicMax).
	// Initialize cell_starts to `numParticles` (the sentinel value for atomicMin).
	if (h_grid_data.numGridCells > 0) {
		unsigned long long numGridCells = h_grid_data.numGridCells;
		int threads_per_block = 256;
		dim3 blocks((numGridCells + threads_per_block - 1) / threads_per_block);
		dim3 threads(threads_per_block);

		Grid_InitArrayKernel<<<blocks, threads>>>(h_grid_data.cell_starts, numGridCells, numParticles);
		CHECK_CUDA_ERROR(cudaGetLastError());

		CHECK_CUDA_ERROR(cudaMemset(h_grid_data.cell_ends, 0, h_grid_data.numGridCells * sizeof(unsigned int)));
	}

	// Step 6: Copy the entire, fully-configured host struct to the device
	CHECK_CUDA_ERROR(cudaMemcpy(d_grid_data, &h_grid_data, sizeof(GridData), cudaMemcpyHostToDevice));

	return d_grid_data;
}

__host__ void GridData_FreeOnDevice(GridData *grid_data) {
	if (grid_data) {
		GridData h_grid_data_ptrs;
		CHECK_CUDA_ERROR(cudaMemcpy(&h_grid_data_ptrs, grid_data, sizeof(GridData), cudaMemcpyDeviceToHost));

		CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.particle_hashes));
		CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.particle_indices));
		CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.cell_starts));
		CHECK_CUDA_ERROR(cudaFree(h_grid_data_ptrs.cell_ends));

		CHECK_CUDA_ERROR(cudaFree(grid_data));
	}
}

__host__ void GridData_CopyParamsToDevice(GridData *host_grid_data, GridData *device_grid_data) {
	if (!host_grid_data || !device_grid_data) {
		fprintf(stderr, "Error: Null pointer passed to GridData_CopyParamsToDevice.\n");
		return;
	}
	// Copies parameters *only*. Does not update the device array pointers if they changed.
	CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridCellSize, &host_grid_data->gridCellSize, sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->invGridCellSize, &host_grid_data->invGridCellSize, sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->domainMin, &host_grid_data->domainMin, sizeof(float4), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->gridDimensions, &host_grid_data->gridDimensions, sizeof(int4), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(&device_grid_data->numGridCells, &host_grid_data->numGridCells, sizeof(unsigned long long), cudaMemcpyHostToDevice));
}

// --- Grid Building Kernel Functions ---

__global__ void Grid_CalculateHashesKernel(
    const ParticleSystem *ps,
    GridData *grid_data) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;

	float4 pos = ps->pos[i];

	int4 cell_coords = Grid_GetCellCoords(pos, grid_data);
	unsigned long long hash = Grid_GetHashFromCell(cell_coords, grid_data);

	grid_data->particle_hashes[i] = hash;
	grid_data->particle_indices[i] = i;
}

__global__ void Grid_FindCellBoundsKernel(
    const unsigned long long *d_sorted_particle_hashes,
    GridData *grid_data,
    int numParticles) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numParticles) return;

	unsigned long long current_hash = d_sorted_particle_hashes[i];

	if (current_hash < grid_data->numGridCells) {
		// Use atomic operations to update cell_starts and cell_ends
		// cell_starts initialized to numParticles, cell_ends initialized to 0
		atomicMin(&grid_data->cell_starts[current_hash], i);
		atomicMax(&grid_data->cell_ends[current_hash], i + 1); // i+1 for exclusive end
	}
}

// --- Initialization Kernel ---

__global__ void Grid_InitArrayKernel(
    unsigned int *array,
    unsigned long long num_elements,
    unsigned int value) {
	unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements) {
		array[i] = value;
	}
}

// --- Build Grid Function ---

__host__ void Grid_Build(GridData *grid_data, const ParticleSystem *ps, int numParticles) {
	if (numParticles == 0) return;

	dim3 threads(256);
	dim3 blocks((numParticles + threads.x - 1) / threads.x);

	// 1. Calculate a hash for each particle based on its grid cell location.
	Grid_CalculateHashesKernel<<<blocks, threads>>>(ps, grid_data);
	CHECK_CUDA_ERROR(cudaGetLastError());

	// --- Sorting ---
	// We need the device pointers on the host to configure the Thrust sort.
	// The standard pattern is to copy the GridData struct (which contains the pointers)
	// from device to a temporary host struct.
	GridData h_grid_ptrs;
	CHECK_CUDA_ERROR(cudaMemcpy(&h_grid_ptrs, grid_data, sizeof(GridData), cudaMemcpyDeviceToHost));

	// 2. Sort particles by hash using Thrust.
	// This reorders particle_indices so that particles in the same cell are adjacent.
	thrust::device_ptr<unsigned long long> hashes_ptr(h_grid_ptrs.particle_hashes);
	thrust::device_ptr<unsigned int> indices_ptr(h_grid_ptrs.particle_indices);

	try {
		thrust::sort_by_key(thrust::device, hashes_ptr, hashes_ptr + numParticles, indices_ptr);
	} catch (const thrust::system_error &e) {
		fprintf(stderr, "Thrust sort_by_key failed in Grid_Build: %s\n", e.what());
		exit(-200);
	}
	CHECK_CUDA_ERROR(cudaGetLastError());

	// --- Finding Cell Bounds ---
	unsigned long long numGridCells = h_grid_ptrs.numGridCells;
	if (numGridCells > 0) {
		// 3. Reset cell start/end arrays for this frame.
		dim3 grid_blocks((numGridCells + threads.x - 1) / threads.x);

		// Use the initialization kernel from our previous fix.
		Grid_InitArrayKernel<<<grid_blocks, threads>>>(h_grid_ptrs.cell_starts, numGridCells, numParticles);

		// Standard cudaMemset is efficient for zeroing memory.
		CHECK_CUDA_ERROR(cudaMemset(h_grid_ptrs.cell_ends, 0, numGridCells * sizeof(unsigned int)));
		CHECK_CUDA_ERROR(cudaGetLastError());

		// 4. Find the start and end index for each cell in the sorted particle array.
		// NOTE: The first argument to this kernel MUST be the sorted hashes pointer.
		Grid_FindCellBoundsKernel<<<blocks, threads>>>(h_grid_ptrs.particle_hashes, grid_data, numParticles);
		CHECK_CUDA_ERROR(cudaGetLastError());
	}
}
#include "simulation.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cmath> // For M_PI, powf, etc.
#include <vector_types.h>
#include <vector_functions.h>

// --- SPH Kernel Helper Functions (Device-Only) ---
// These functions compute the values of the SPH smoothing kernels and their derivatives.
// They use the pre-computed coefficients from the SimulationParams struct for efficiency.

// Poly6 Kernel for density calculation
__device__ inline float poly6_kernel(float r_sq, const SimulationParams *params) {
	if (r_sq >= params->smoothingRadiusSq) return 0.0f;
	float term = params->smoothingRadiusSq - r_sq;
	return params->poly6KernelCoeff * term * term * term;
}

// Spiky Kernel Gradient for pressure force
__device__ inline float4 spiky_kernel_gradient(float4 r, float dist, const SimulationParams *params) {
	if (dist >= params->smoothingRadius || dist < 1e-6f) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float h_minus_r = params->smoothingRadius - dist;
	float coeff = params->spikyKernelGradientCoeff * h_minus_r * h_minus_r;
	return r * (coeff / dist);
}

// Viscosity Kernel Laplacian for viscosity force
__device__ inline float viscosity_kernel_laplacian(float dist, const SimulationParams *params) {
	if (dist >= params->smoothingRadius) return 0.0f;
	return params->viscosityKernelLaplacianCoeff * (params->smoothingRadius - dist);
}

// Poly6 Kernel Gradient for surface tension normal (color field gradient)
__device__ inline float4 poly6_kernel_gradient(float4 r, float r_sq, const SimulationParams *params) {
	if (r_sq >= params->smoothingRadiusSq) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float term = params->smoothingRadiusSq - r_sq;
	// Coefficient for gradient: -945 / (32 * pi * h^9)
	float coeff = -945.0f / (32.0f * M_PI * powf(params->smoothingRadius, 9.0f)) * term * term;
	return r * coeff;
}

// Poly6 Kernel Laplacian for surface tension force (color field laplacian)
__device__ inline float poly6_kernel_laplacian(float r_sq, const SimulationParams *params) {
	if (r_sq >= params->smoothingRadiusSq) return 0.0f;
	// Coefficient for laplacian: -945 / (32 * pi * h^9)
	float coeff = -945.0f / (32.0f * M_PI * powf(params->smoothingRadius, 9.0f));
	return coeff * (params->smoothingRadiusSq - r_sq) * (3.0f * params->smoothingRadiusSq - 7.0f * r_sq);
}

// --- Simulation Lifecycle Functions ---

Simulation *Simulation_Create(int numParticles) {
	Simulation *sim = (Simulation *)malloc(sizeof(Simulation));
	if (!sim) {
		fprintf(stderr, "Failed to allocate Simulation struct on host.\n");
		return nullptr;
	}

	// Allocate host structures
	sim->host_ps = ParticleSystem_CreateOnHost(numParticles);
	sim->host_params = SimulationParams_CreateOnHost();
	sim->host_grid = GridData_CreateOnHost(numParticles, sim->host_params);

	// Allocate device structures
	sim->device_ps = ParticleSystem_CreateOnDevice(numParticles);
	sim->device_params = SimulationParams_CreateOnDevice();
	sim->device_grid = GridData_CreateOnDevice(numParticles, sim->host_params);

	sim->host_sim = sim;
	CHECK_CUDA_ERROR(cudaMalloc((void **)&sim->device_sim, sizeof(Simulation)));

	CHECK_CUDA_ERROR(cudaMemcpy(sim->device_sim, sim->host_sim, sizeof(Simulation), cudaMemcpyHostToDevice));

	return sim;
}

void Simulation_Free(Simulation *sim) {
	if (sim) {
		// Free device memory
		ParticleSystem_FreeOnDevice(sim->device_ps);
		SimulationParams_FreeOnDevice(sim->device_params);
		GridData_FreeOnDevice(sim->device_grid);

		// Free host memory
		ParticleSystem_FreeOnHost(sim->host_ps);
		SimulationParams_FreeOnHost(sim->host_params);
		GridData_FreeOnHost(sim->host_grid);
		CHECK_CUDA_ERROR(cudaFree(sim->device_sim));
		free(sim);
	}
}

// --- Data Synchronization Functions ---

void Simulation_CopyAll_HostToDevice(Simulation *sim) {
	Simulation_CopyParticles_HostToDevice(sim);
	Simulation_CopyParameters_HostToDevice(sim);
	Simulation_CopyGrid_HostToDevice(sim);
}

void Simulation_CopyParticles_HostToDevice(Simulation *sim) {
	ParticleSystem_CopyAll_HostToDevice(sim->host_ps, sim->device_ps);
}

void Simulation_CopyParameters_HostToDevice(Simulation *sim) {
	SimulationParams_Copy_HostToDevice(sim->host_params, sim->device_params);
}

void Simulation_CopyGrid_HostToDevice(Simulation *sim) {
	// Note: Grid copy is mostly about parameters. The main arrays are device-only.
	GridData_CopyParamsToDevice(sim->host_grid, sim->device_grid);
}

void Simulation_CopyAll_DeviceToHost(Simulation *sim) {
	Simulation_CopyParticles_DeviceToHost(sim);
	Simulation_CopyParameters_DeviceToHost(sim);
	Simulation_CopyGrid_DeviceToHost(sim);
}

void Simulation_CopyParticles_DeviceToHost(Simulation *sim) {
	ParticleSystem_CopyAll_DeviceToHost(sim->host_ps, sim->device_ps);
}

void Simulation_CopyParameters_DeviceToHost(Simulation *sim) {
	SimulationParams_Copy_DeviceToHost(sim->device_params, sim->host_params);
}

void Simulation_CopyGrid_DeviceToHost(Simulation *sim) {
	// This is not typically needed, as the grid is an intermediate structure.
	// A full implementation would copy back the arrays too.
}

// --- Simulation Control Functions ---

void Simulation_Step(Simulation *h_sim, float dt) {

	unsigned int numParticles = h_sim->host_ps->numParticles;
	dim3 threads(256);
	dim3 blocks((numParticles + threads.x - 1) / threads.x);

	// 1. Update spatial grid for neighbor search
	Grid_Build(h_sim->device_grid, h_sim->device_ps, h_sim->host_ps->numParticles);

	Simulation *sim = h_sim->device_sim; // Use device pointer for kernels

	// 2. Reset forces from previous step
	Simulation_Kernel_ResetForces<<<blocks, threads>>>(sim);

	// 3. Compute density and pressure
	Simulation_Kernel_ComputeDensityPressure<<<blocks, threads>>>(sim);

	// 4. Compute internal forces (pressure + viscosity)
	Simulation_Kernel_ComputeInternalForces<<<blocks, threads>>>(sim);

	// 5. Compute surface tension force
	Simulation_Kernel_ComputeSurfaceTension<<<blocks, threads>>>(sim);

	// 6. Apply external forces (gravity) and boundary conditions
	Simulation_Kernel_ApplyExternalAndBoundaryForces<<<blocks, threads>>>(sim);

	// 7. Integrate particle positions and velocities
	Simulation_Kernel_IntegrateStepKernel<<<blocks, threads>>>(sim, dt);

	// Check for errors after kernel launches
	CHECK_CUDA_ERROR(cudaGetLastError());
}

// --- Simulation Kernels ---

__global__ void Simulation_Kernel_ResetForces(Simulation *sim) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= sim->device_ps->numParticles) return;

	sim->device_ps->force[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void Simulation_Kernel_ComputeDensityPressure(Simulation *sim) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	ParticleSystem *ps = sim->device_ps;
	if (i >= ps->numParticles) return;

	const SimulationParams *params = sim->device_params;
	const GridData *grid = sim->device_grid;

	float4 pos_i = ps->pos[i];
	float density = 0.0f;

	// Get the grid cell coordinates for particle i
	int4 cell_coords_i = Grid_GetCellCoords(pos_i, grid);

	for (int dw = -1; dw <= 1; ++dw) {
		for (int dz = -1; dz <= 1; ++dz) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int4 neighbor_cell_coords = cell_coords_i + make_int4(dx, dy, dz, dw);
					unsigned int hash = Grid_GetHashFromCell(neighbor_cell_coords, grid);

					if (hash >= grid->numGridCells) continue;

					unsigned int start_idx = grid->cell_starts[hash];
					if (start_idx >= ps->numParticles) continue; // Skip empty cells

					unsigned int end_idx = grid->cell_ends[hash];

					// Iterate over particles in the neighboring cell
					for (unsigned int j_idx = start_idx; j_idx < end_idx; ++j_idx) {
						unsigned int j = grid->particle_indices[j_idx];

						float4 r = pos_i - ps->pos[j];
						float r_sq = dot(r, r);

						if (r_sq < params->smoothingRadiusSq) {
							// Accumulate density from this neighbor
							density += ps->mass[j] * poly6_kernel(r_sq, params);
						}
					}
				}
			}
		}
	}

	ps->density[i] = density;
	// Simplified Tait's equation of state to compute pressure
	ps->pressure[i] = params->gasConstantK * (density - params->restDensity);
	if (ps->pressure[i] < 0.0f) {
		ps->pressure[i] = 0.0f;
	}
}

__global__ void Simulation_Kernel_ComputeInternalForces(Simulation *sim) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	ParticleSystem *ps = sim->device_ps;
	if (i >= ps->numParticles) return;

	const SimulationParams *params = sim->device_params;
	const GridData *grid = sim->device_grid;

	float4 pos_i = ps->pos[i];
	float4 vel_i = ps->vel[i];
	float pressure_i = ps->pressure[i];
	float density_i = ps->density[i];

	float4 pressure_force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 viscosity_force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	int4 cell_coords_i = Grid_GetCellCoords(pos_i, grid);

	// Iterate through neighboring cells
	for (int dw = -1; dw <= 1; ++dw) {
		for (int dz = -1; dz <= 1; ++dz) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int4 neighbor_cell_coords = cell_coords_i + make_int4(dx, dy, dz, dw);
					unsigned int hash = Grid_GetHashFromCell(neighbor_cell_coords, grid);
					if (hash >= grid->numGridCells) continue;

					unsigned int start_idx = grid->cell_starts[hash];
					if (start_idx >= ps->numParticles) continue;

					unsigned int end_idx = grid->cell_ends[hash];

					for (unsigned int j_idx = start_idx; j_idx < end_idx; ++j_idx) {
						unsigned int j = grid->particle_indices[j_idx];
						if (i == j) continue;

						float4 r = pos_i - ps->pos[j];
						float r_sq = dot(r, r);

						if (r_sq < params->smoothingRadiusSq) {
							float dist = sqrtf(max(1.e-12f, r_sq));

							float density_j = ps->density[j];
							if (density_j < 1e-6f) continue;

							// Pressure force (symmetrized)
							float pressure_term = (pressure_i + ps->pressure[j]) / (2.0f * density_j);
							pressure_force -= ps->mass[j] * pressure_term * spiky_kernel_gradient(r, dist, params);

							// Viscosity force
							viscosity_force += params->viscosityCoefficient * ps->mass[j] *
							                   ((ps->vel[j] - vel_i) / density_j) *
							                   viscosity_kernel_laplacian(dist, params);
						}
					}
				}
			}
		}
	}

	ps->force[i] += pressure_force + viscosity_force;
}

__global__ void Simulation_Kernel_ComputeSurfaceTension(Simulation *sim) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	ParticleSystem *ps = sim->device_ps;
	if (i >= ps->numParticles) return;

	const SimulationParams *params = sim->device_params;
	const GridData *grid = sim->device_grid;

	float4 pos_i = ps->pos[i];
	float density_i = ps->density[i];
	if (density_i < 1e-6f) return;

	float4 color_gradient = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // This is the surface normal vector
	float color_laplacian = 0.0f;

	int4 cell_coords_i = Grid_GetCellCoords(pos_i, grid);

	// Iterate through neighboring cells
	for (int dw = -1; dw <= 1; ++dw) {
		for (int dz = -1; dz <= 1; ++dz) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					int4 neighbor_cell_coords = cell_coords_i + make_int4(dx, dy, dz, dw);
					unsigned int hash = Grid_GetHashFromCell(neighbor_cell_coords, grid);
					if (hash >= grid->numGridCells) continue;

					unsigned int start_idx = grid->cell_starts[hash];
					if (start_idx >= ps->numParticles) continue;

					unsigned int end_idx = grid->cell_ends[hash];

					for (unsigned int j_idx = start_idx; j_idx < end_idx; ++j_idx) {
						unsigned int j = grid->particle_indices[j_idx];

						float4 r = pos_i - ps->pos[j];
						float r_sq = dot(r, r);

						if (r_sq < params->smoothingRadiusSq) {
							float density_j = ps->density[j];
							if (density_j < 1e-6f) continue;

							float vol_j = ps->mass[j] / density_j;

							// Accumulate color field gradient (surface normal)
							color_gradient += vol_j * poly6_kernel_gradient(r, r_sq, params);

							// Accumulate color field laplacian
							color_laplacian += vol_j * poly6_kernel_laplacian(r_sq, params);
						}
					}
				}
			}
		}
	}

	ps->normal[i] = color_gradient;
	ps->color_laplacian[i] = color_laplacian;

	float normal_mag = length(color_gradient);
	if (normal_mag > params->surfaceTensionThreshold) {
		// Apply surface tension force (acts to minimize surface area)
		// Force is proportional to curvature (laplacian) and in the direction of the normal
		float4 st_force = -params->surfaceTensionCoefficient * color_laplacian * (color_gradient / normal_mag);
		ps->force[i] += st_force;
	}
}

__global__ void Simulation_Kernel_ApplyExternalAndBoundaryForces(Simulation *sim) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	ParticleSystem *ps = sim->device_ps;
	if (i >= ps->numParticles) return;

	const SimulationParams *params = sim->device_params;

	// Apply gravity
	ps->force[i] += ps->mass[i] * params->gravity;

	// Boundary conditions (penalty method)
	float4 pos = ps->pos[i];
	float4 vel = ps->vel[i];
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// Check against min/max bounds
	if (pos.x < params->min.x) {
		force.x += params->wallStiffness * (params->min.x - pos.x);
		if (vel.x < 0) force.x += params->boundaryDamping * vel.x;
	}
	if (pos.x > params->max.x) {
		force.x += params->wallStiffness * (params->max.x - pos.x);
		if (vel.x > 0) force.x += params->boundaryDamping * vel.x;
	}
	if (pos.y < params->min.y) {
		force.y += params->wallStiffness * (params->min.y - pos.y);
		if (vel.y < 0) force.y += params->boundaryDamping * vel.y;
	}
	if (pos.y > params->max.y) {
		force.y += params->wallStiffness * (params->max.y - pos.y);
		if (vel.y > 0) force.y += params->boundaryDamping * vel.y;
	}
	if (pos.z < params->min.z) {
		force.z += params->wallStiffness * (params->min.z - pos.z);
		if (vel.z < 0) force.z += params->boundaryDamping * vel.z;
	}
	if (pos.z > params->max.z) {
		force.z += params->wallStiffness * (params->max.z - pos.z);
		if (vel.z > 0) force.z += params->boundaryDamping * vel.z;
	}
	if (pos.w < params->min.w) {
		force.w += params->wallStiffness * (params->min.w - pos.w);
		if (vel.w < 0) force.w += params->boundaryDamping * vel.w;
	}
	if (pos.w > params->max.w) {
		force.w += params->wallStiffness * (params->max.w - pos.w);
		if (vel.w > 0) force.w += params->boundaryDamping * vel.w;
	}

	ps->force[i] += force;
}

__global__ void Simulation_Kernel_IntegrateStepKernel(Simulation *sim, float dt) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	ParticleSystem *ps = sim->device_ps;
	if (i >= ps->numParticles) return;

	float mass_i = ps->mass[i];
	if (mass_i < 1e-6f) return;

	// Symplectic Euler integration
	// a = F / m
	float4 acceleration = ps->force[i] / mass_i;

	// v(t+dt) = v(t) + a(t) * dt
	ps->vel[i] += acceleration * dt;

	// x(t+dt) = x(t) + v(t+dt) * dt
	ps->pos[i] += ps->vel[i] * dt;
}
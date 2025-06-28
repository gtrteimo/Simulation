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
	// The coefficient is now positive, and we apply the negative sign in the force calculation.
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
	float coeff = params->poly6KernelGradientCoeff * term * term;
	return r * coeff;
}

// Poly6 Kernel Laplacian for surface tension force (color field laplacian)
__device__ inline float poly6_kernel_laplacian(float r_sq, const SimulationParams *params) {
	if (r_sq >= params->smoothingRadiusSq) return 0.0f;
	float coeff = params->poly6KernelLaplacianCoeff;
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
	ParticleSystem_CopyAll_HostToDevice(sim->host_ps, sim->device_ps, sim->host_ps->maxParticles);
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
	ParticleSystem_CopyAll_DeviceToHost(sim->host_ps, sim->device_ps, sim->host_ps->maxParticles);
}

void Simulation_CopyParameters_DeviceToHost(Simulation *sim) {
	SimulationParams_Copy_DeviceToHost(sim->device_params, sim->host_params);
}

void Simulation_CopyGrid_DeviceToHost(Simulation *sim) {
	// This is not typically needed, as the grid is an intermediate structure.
	// A full implementation would copy back the arrays too.
}

// --- Simulation Control Functions ---

void Simulation_SetActiveParticles(Simulation *sim, unsigned int numParticles) {
	sim->host_ps->numParticles = numParticles;
	ParticleSystem_SetNumParticlesOnDevice(sim->device_ps, numParticles);
}

void Simulation_Step(Simulation *h_sim, float dt) {

	unsigned int numParticles = h_sim->host_ps->numParticles;
	if (numParticles == 0) return;
	dim3 threads(256);
	dim3 blocks((numParticles + threads.x - 1) / threads.x);

	// Get pointers to device data to pass to kernels
	ParticleSystem *d_ps = h_sim->device_ps;
	SimulationParams *d_params = h_sim->device_params;
	GridData *d_grid = h_sim->device_grid;

	// 1. Update spatial grid for neighbor search
	Grid_Build(d_grid, d_ps, numParticles);

	// 2. Reset forces from previous step
	Simulation_Kernel_ResetForces<<<blocks, threads>>>(d_ps);

	// 3. Compute density and pressure
	Simulation_Kernel_ComputeDensityPressure<<<blocks, threads>>>(d_ps, d_params, d_grid);

	// 4. Compute internal forces (pressure + viscosity)
	Simulation_Kernel_ComputeInternalForces<<<blocks, threads>>>(d_ps, d_params, d_grid);

	// 5. Compute surface tension force
	Simulation_Kernel_ComputeSurfaceTension<<<blocks, threads>>>(d_ps, d_params, d_grid);

	// 6. Apply external forces (gravity) and boundary conditions
	Simulation_Kernel_ApplyExternalForces<<<blocks, threads>>>(d_ps, d_params);

	// 7. Integrate particle positions and velocities
	Simulation_Kernel_IntegrateStepAndBoundary<<<blocks, threads>>>(d_ps, d_params, dt);

	// Check for errors after kernel launches
	CHECK_CUDA_ERROR(cudaGetLastError());
}

// --- Simulation Kernels ---

__global__ void Simulation_Kernel_ResetForces(ParticleSystem *ps) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;
	ps->force[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void Simulation_Kernel_ComputeDensityPressure(ParticleSystem *ps, const SimulationParams *params, const GridData *grid) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;

	float4 pos_i = ps->pos[i];
	float density = 0.0f;

	// Add self-density first to avoid issues with zero-neighbor particles
	density += ps->mass[i] * poly6_kernel(0.0f, params);

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
						if (i == j) continue; // Skip self-interaction

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

__global__ void Simulation_Kernel_ComputeInternalForces(ParticleSystem *ps, const SimulationParams *params, const GridData *grid) {
	// ############### CORRECTED KERNEL ###############
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;

	float4 pos_i = ps->pos[i];
	float4 vel_i = ps->vel[i];
	float pressure_i = ps->pressure[i];
	float density_i = ps->density[i];

	if (density_i < 1e-6f) return;

	// Use temporary variables to sum the force contributions.
	float4 pressure_force_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 viscosity_force_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

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

							// --- CORRECTED PRESSURE FORCE ---
							// Symmetrical pressure term: (p_i/rho_i^2 + p_j/rho_j^2)
							float pressure_val = (pressure_i / (density_i * density_i)) + (ps->pressure[j] / (density_j * density_j));
							// Add the contribution from neighbor j. Note that mass_i is NOT included here.
							pressure_force_sum += ps->mass[j] * pressure_val * spiky_kernel_gradient(r, dist, params);

							// --- CORRECTED VISCOSITY FORCE ---
							// Add the contribution from neighbor j. Note that mass_i and viscosity coeff are NOT included here.
							viscosity_force_sum += ps->mass[j] *
							                       ((ps->vel[j] - vel_i) / density_j) *
							                       viscosity_kernel_laplacian(dist, params);
						}
					}
				}
			}
		}
	}

	// --- FINAL FORCE CALCULATION (PHYSICALLY CORRECT) ---
	// The total force on particle 'i' is its own mass times the sum of neighbor contributions.
	// This is done *after* the loop to prevent multiplying by mass_i for every neighbor.
	float mass_i = ps->mass[i];

	// The pressure force is repulsive (acts opposite to the gradient).
	// Our spiky kernel coefficient is now positive, so we add the negative sign here.
	float4 final_pressure_force = -mass_i * pressure_force_sum;

	// The viscosity force is scaled by the viscosity coefficient.
	float4 final_viscosity_force = mass_i * params->viscosityCoefficient * viscosity_force_sum;

	// Add the computed forces to the total force accumulator for this particle.
	ps->force[i] += final_pressure_force + final_viscosity_force;
}

__global__ void Simulation_Kernel_ComputeSurfaceTension(ParticleSystem *ps, const SimulationParams *params, const GridData *grid) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;

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

	float normal_mag_sq = dot(color_gradient, color_gradient);
	if (normal_mag_sq > (params->surfaceTensionThreshold * params->surfaceTensionThreshold)) {
		// Apply surface tension force (acts to minimize surface area)
		// Force is proportional to curvature (laplacian) and in the direction of the normal
		float normal_mag = sqrtf(normal_mag_sq);
		float4 st_force = -params->surfaceTensionCoefficient * color_laplacian * (color_gradient / normal_mag);
		ps->force[i] += st_force;
	}
}

__global__ void Simulation_Kernel_ApplyExternalForces(ParticleSystem *ps, const SimulationParams *params) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;

	ps->force[i] += ps->mass[i] * params->gravity;
}

__global__ void Simulation_Kernel_IntegrateStepAndBoundary(ParticleSystem *ps, const SimulationParams *params, float dt) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ps->numParticles) return;

	float mass_i = ps->mass[i];
	if (mass_i < 1e-6f) return;

	// --- 1. Symplectic Euler Integration ---
	float4 vel = ps->vel[i];
	float4 acceleration = ps->force[i] / mass_i;

	// v(t+dt) = v(t) + a(t) * dt
	vel += acceleration * dt;

	// x(t+dt) = x(t) + v(t+dt) * dt
	float4 pos = ps->pos[i] + vel * dt;

	// --- 2. Hard Wall Boundary Condition ---
	const float damping = params->boundaryDamping;

	// X-axis
	if (pos.x < params->min.x) {
		pos.x = params->min.x; // Clamp position to the wall boundary
		vel.x *= -damping;     // Reflect and dampen velocity
	}
	if (pos.x > params->max.x) {
		pos.x = params->max.x;
		vel.x *= -damping;
	}
	// Y-axis
	if (pos.y < params->min.y) {
		pos.y = params->min.y;
		vel.y *= -damping;
	}
	if (pos.y > params->max.y) {
		pos.y = params->max.y;
		vel.y *= -damping;
	}
	// Z-axis
	if (pos.z < params->min.z) {
		pos.z = params->min.z;
		vel.z *= -damping;
	}
	if (pos.z > params->max.z) {
		pos.z = params->max.z;
		vel.z *= -damping;
	}
	// W-axis
	
	if (pos.w < params->min.w) {
		pos.w = params->min.w;
		vel.w *= -damping;
	}
	if (pos.w > params->max.w) {
		pos.w = params->max.w;
		vel.w *= -damping;
	}

	// --- 3. Update Global Memory ---
	ps->pos[i] = pos;
	ps->vel[i] = vel;
}
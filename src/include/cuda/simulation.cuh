#pragma once

#include "cuda/util.cuh"
#include "cuda/params.cuh"
#include "cuda/particles.cuh"
#include "cuda/grid.cuh"
#include "cuda/util.cuh"

struct Simulation {
	ParticleSystem *device_ps = nullptr;
	SimulationParams *device_params = nullptr;
	GridData *device_grid = nullptr;

	ParticleSystem *host_ps = nullptr;
	SimulationParams *host_params = nullptr;
	GridData *host_grid = nullptr;
};

Simulation *Simulation_Create(int numParticles);

void Simulation_Free(Simulation *sim);

// Function to synchronize

void Simulation_Host_Sync_AllToDevice(Simulation *sim);
void Simulation_Host_Sync_ParticlesToDevice(Simulation *sim);
void Simulation_Host_Sync_ParametersToDevice(Simulation *sim);
void Simulation_Host_Sync_GridToDevice(Simulation *sim);

void Simulation_Device_Sync_AllToHost(Simulation *sim);
void Simulation_Device_Sync_ParticlesToHost(Simulation *sim);
void Simulation_Device_Sync_ParametersToHost(Simulation *sim);
void Simulation_Device_Sync_GridToHost(Simulation *sim);

// --- Simulation Device control functions ---

// Main function to advance the SPH simulation by one time step (dt).
// This function orchestrates calls to various CUDA kernels.
// - sim: Contains device particle data (ps_device), simulation parameters, and grid data.
// - dt: The time step for this simulation update.
void Simulation_Device_Step(
    Simulation *sim,
    float dt);

// Function to build or update the spatial grid for neighbor search.
// This involves:
// 1. Calculating hashes (Grid_CalculateHashesKernel).
// 2. Sorting particles by hash (host-side library call like thrust::sort_by_key).
// 3. Finding cell bounds (Grid_FindCellBoundsKernel).
// Modifies grid data within sim->grid_data_device.
void Simulation_Device_BuildGrid(Simulation *sim);

// --- Simulation Kernel ---

// Kernel to reset forces for all particles to zero.
__global__ void Simulation_Kernel_ResetForces(
    Simulation *sim);

// Kernel to compute density and pressure for each particle.
// Uses the spatial grid for efficient neighbor lookup.
__global__ void Simulation_Kernel_ComputeDensityPressure(
    Simulation *sim);

// Kernel to compute internal SPH forces (pressure, viscosity).
__global__ void Simulation_Kernel_ComputeInternalForces(
    Simulation *sim);
// Kernel to compute surface tension forces (e.g., using color field method).
// This involves calculating color field gradients (normals) and Laplacians.
__global__ void Simulation_Kernel_ComputeSurfaceTension(
    Simulation *sim);

// Kernel to apply external forces (e.g., gravity) and handle boundary interactions.
__global__ void Simulation_Kernel_ApplyExternalAndBoundaryForces(
    Simulation *sim);

// Kernel to integrate particle positions and velocities using current forces.
// (e.g., Leapfrog, Verlet, or Symplectic Euler integration)
__global__ void Simulation_Kernel_IntegrateStepKernel(
    Simulation *sim,
    float dt);
#pragma once

#include "cuda/util.cuh"
#include "cuda/params.cuh"
#include "cuda/particles.cuh"
#include "cuda/grid.cuh"
#include "cuda/util.cuh"
#include "types/simulationTypes.h"

struct Simulation {
	ParticleSystem *device_ps = nullptr;
	SimulationParams *device_params = nullptr;
	GridData *device_grid = nullptr;

	ParticleSystem *host_ps = nullptr;
	SimulationParams *host_params = nullptr;
	GridData *host_grid = nullptr;

	Simulation *host_sim;
	Simulation *device_sim;
};

Simulation *Simulation_Create(int numParticles);

void Simulation_Free(Simulation *sim);

// Function to synchronize

void Simulation_CopyAll_HostToDevice(Simulation *sim);
void Simulation_CopyParticles_HostToDevice(Simulation *sim);
void Simulation_CopyParameters_HostToDevice(Simulation *sim);
void Simulation_CopyGrid_HostToDevice(Simulation *sim);

void Simulation_CopyAll_DeviceToHost(Simulation *sim);
void Simulation_CopyParticles_DeviceToHost(Simulation *sim);
void Simulation_CopyParameters_DeviceToHost(Simulation *sim);
void Simulation_CopyGrid_DeviceToHost(Simulation *sim);

// --- Simulation control functions ---

// Main function to advance the SPH simulation by one time step (dt).
// This function orchestrates calls to various CUDA kernels.
// - sim: Contains all necessary data, but only device data is used.
// - dt: The time step for this simulation update.
void Simulation_Step(Simulation *sim, float dt);

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
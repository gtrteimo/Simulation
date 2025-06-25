#pragma once

// No logic or functions in here. just structs. The logic goes into other files so someone can 
// write a C++ alternative to my CUDA code and retain the same data structures.
// This file should contain only simulation-related data structures. GUI structs go somewhere else.

struct GridData {
    unsigned long long* particle_hashes;
    unsigned int* particle_indices;
    unsigned int* cell_starts;
    unsigned int* cell_ends;

    float gridCellSize;
    float invGridCellSize;
    float4 domainMin;
    int4 gridDimensions;
    unsigned long long numGridCells; // WARNING: Can overflow for large 4D grids
};

struct SimulationParams {
    // Core SPH parameters
    float smoothingRadius;      // h: Radius of influence for kernel computations
    float gasConstantK;         // k_gas: Stiffness for the equation of state (e.g., P = k * (rho - rho_0))
    float restDensity;          // rho_0: Target density of the fluid
    float viscosityCoefficient; // mu: Dynamic viscosity coefficient
    float surfaceTensionCoefficient; // sigma: Coefficient for surface tension force
    float surfaceTensionThreshold;   // Threshold for color field laplacian to apply surface tension

    // Simulation control & environment
    float4 gravity;

    // Domain boundaries (Axis-Aligned Bounding Box - AABB)
    float4 min, max;
    float boundaryDamping;      // Damping coefficient for collisions with boundaries (e.g., 0.6 means 60% velocity retained perpendicular to wall)
    float wallStiffness;        // Stiffness for boundary repulsion penalty force

    // Kernel precomputed values (derived from smoothingRadius, crucial for SPH calculations)
    float smoothingRadiusSq;
    float poly6KernelCoeff;
    float spikyKernelGradientCoeff;
    float viscosityKernelLaplacianCoeff;
};

struct ParticleSystem {
	float4 *pos;
	float4 *vel;
	float4 *force;
	float *mass;
	float *density;
	float *pressure;
	float4 *normal;
	float *color_laplacian;
	unsigned int numParticles;
};


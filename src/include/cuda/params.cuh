#pragma once

#include "cuda/util.cuh"

struct SimulationParams {
    // Core SPH parameters
    float smoothingRadius;      // h: Radius of influence for kernel computations
    float smoothingRadiusSq;    // h^2: Precomputed for efficiency
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
    // These are for specific kernel formulations (e.g., Poly6 for density, Spiky for pressure gradient)
    float poly6KernelCoeff;
    float spikyKernelGradientCoeff;
    float viscosityKernelLaplacianCoeff;

    /*
    // Constructor to initialize with some default values (optional, can be set from main)
    SimulationParams() :
        smoothingRadius(0.04f), // Example: 4cm if units are meters
        gasConstantK(20.0f),
        restDensity(1000.0f),   // kg/m^3 (like water)
        viscosityCoefficient(0.05f), // Tunable
        surfaceTensionCoefficient(0.0728f), // N/m (water-air)
        surfaceTensionThreshold(7.0f),    // Tunable
        gravity_x(0.0f), gravity_y(-9.81f), gravity_z(0.0f),
        min_x(-0.5f), max_x(0.5f), min_y(-0.5f), max_y(0.5f), min_z(-0.5f), max_z(0.5f),
        boundaryDamping(-0.5f), // Negative for reflection with damping
        wallStiffness(3000.0f),
        numParticles(0)
    {
        PrecomputeKernelCoefficients(); // Initialize coefficients based on smoothingRadius
    }*/

    
};

// --- Device Functions ---

SimulationParams* SimulationParams_Device_Create();
void SimulationParams_Device_Free(SimulationParams* ps);
void SimulationParams_Device_CopyToHost(SimulationParams* ps_device, SimulationParams* ps_host);

// --- Host Functions ---

SimulationParams* SimulationParams_Host_Create();
void SimulationParams_Host_Free(SimulationParams* ps);
void SimulationParams_Host_CopyToDevice(SimulationParams* ps_host, SimulationParams* ps_device);

// --- Utility Functions ---

// Call this if smoothingRadius is changed after construction
void PrecomputeKernelCoefficients(SimulationParams& params) {
    if (params.smoothingRadius > 1e-6f) { // Avoid division by zero
        const float PI_F = static_cast<float>(M_PI);
        float h = params.smoothingRadius;
        float h2 = h * h;
        float h3 = h2 * h;
        float h6 = h3 * h3;
        float h9 = h3 * h6;

        params.smoothingRadiusSq = h2;
        params.poly6KernelCoeff = 315.0f / (64.0f * PI_F * h9);
        params.spikyKernelGradientCoeff = -45.0f / (PI_F * h6); // Negative sign often incorporated directly in force summation
        params.viscosityKernelLaplacianCoeff = 45.0f / (PI_F * h6);
    } else {
        params.smoothingRadiusSq = 0.0f;
        params.poly6KernelCoeff = 0.0f;
        params.spikyKernelGradientCoeff = 0.0f;
        params.viscosityKernelLaplacianCoeff = 0.0f;
    }
}
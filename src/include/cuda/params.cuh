#pragma once

#include "cuda/util.cuh"
#include <math.h>

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

const SimulationParams SimulationParams_Default = {
    0.04f, // smoothingRadius (4cm)
    20.0f, // gasConstantK
    1000.0f, // restDensity (kg/m^3, like water)
    0.05f, // viscosityCoefficient
    0.0728f, // surfaceTensionCoefficient (N/m, like water-air)
    7.0f, // surfaceTensionThreshold
    {0.0f, -9.81f, 0.0f, 0.0f}, // gravity (m/s^2)
    {-0.5f, -0.5f, -0.5f, -0.5f}, // min AABB
    {0.5f, 0.5f, 0.5f, 0.5f}, // max AABB
    -0.5f, // boundaryDamping (negative for reflection with damping)
    3000.0f, // wallStiffness
    0.0016f, // smoothingRadiusSq (h^2)
    315.5767f * M_PI * pow(0.04f, 9.0f), // poly6KernelCoeff
    -45.0f * M_PI * pow(0.04f, 6.0f), // spikyKernelGradientCoeff
    45.0f * M_PI * pow(0.04f, 6.0f) // viscosityKernelLaplacianCoeff
};

// --- Host Memory Functions ---

__host__ SimulationParams* SimulationParams_CreateOnHost();
__host__ void SimulationParams_FreeOnHost(SimulationParams* ps);

// --- Copy Host to Device Functions ---

__host__ void SimulationParams_Copy_HostToDevice(SimulationParams* ps_host, SimulationParams* ps_device);

// --- Device Memory Functions ---

__host__ SimulationParams* SimulationParams_CreateOnDevice();
__host__ void SimulationParams_FreeOnDevice(SimulationParams* ps);

// --- Copy Device to Host Functions ---

__host__ void SimulationParams_Copy_DeviceToHost(SimulationParams* ps_device, SimulationParams* ps_host);

// --- Utility Functions ---

__host__ void SimulationParams_PrecomputeKernelCoefficients(SimulationParams& params);
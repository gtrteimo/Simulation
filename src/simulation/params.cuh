#pragma once

#include "simulation/util.cuh"
#include "types/simulationTypes.h"
#include <math.h>

const SimulationParams SimulationParams_Default = {
    0.02f,                              // smoothingRadius (2cm)
    20.0f,                              // gasConstantK
    1000.0f,                            // restDensity (kg/m^3, like water)
    0.05f,                              // viscosityCoefficient
    0.0728f,                            // surfaceTensionCoefficient (N/m, like water-air)
    7.0f,                               // surfaceTensionThreshold
    {0.0f, -9.81f, 0.0f, 0.0f},         // gravity (m/s^2)
    {-1.0f, -1.0f, -1.0f, -1.0f},       // min AABB
    {1.0f, 1.0f, 1.0f, 1.0f},           // max AABB
    0.5f,                              // boundaryDamping (how much velocity is retained when getting reflected by the boundary)
    0.0016f,                            // smoothingRadiusSq (h^2)
    315.5767f / M_PI *pow(0.02f, 9.0f), // poly6KernelCoeff
    -45.0f / M_PI *pow(0.02f, 6.0f),    // spikyKernelGradientCoeff
    45.0f / M_PI *pow(0.02f, 6.0f)      // viscosityKernelLaplacianCoeff
};

// --- Host Memory Functions ---

__host__ SimulationParams *SimulationParams_CreateOnHost();
__host__ void SimulationParams_FreeOnHost(SimulationParams *ps);

// --- Copy Host to Device Functions ---

__host__ void SimulationParams_Copy_HostToDevice(SimulationParams *ps_host, SimulationParams *ps_device);

// --- Device Memory Functions ---

__host__ SimulationParams *SimulationParams_CreateOnDevice();
__host__ void SimulationParams_FreeOnDevice(SimulationParams *ps);

// --- Copy Device to Host Functions ---

__host__ void SimulationParams_Copy_DeviceToHost(SimulationParams *ps_device, SimulationParams *ps_host);

// --- Utility Functions ---

__host__ void SimulationParams_PrecomputeKernelCoefficients(SimulationParams &params);
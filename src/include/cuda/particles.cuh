#pragma once

#include "cuda/util.cuh"

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

// --- Host Memory Functions ---

__host__ ParticleSystem *ParticleSystem_CreateOnHost(int numParticles);
__host__ void ParticleSystem_FreeOnHost(ParticleSystem *ps);

__host__ void ParticleSystem_CopyAll_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyPos_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyVel_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyForce_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyMass_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyDensity_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyPressure_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyNormal_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyColorLaplacian_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device);

// --- Device Memory Functions ---

__host__ ParticleSystem *ParticleSystem_CreateOnDevice(int numParticles);
__host__ void ParticleSystem_FreeOnDevice(ParticleSystem *ps);
__host__ void ParticleSystem_SetNumParticlesOnDevice(ParticleSystem *ps, int numParticles);

__host__ void ParticleSystem_CopyAll_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyPos_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyVel_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyForce_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyMass_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyDensity_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyPressure_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyNormal_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);
__host__ void ParticleSystem_CopyColorLaplacian_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device);

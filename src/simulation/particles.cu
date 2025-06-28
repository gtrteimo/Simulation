#include "simulation/particles.cuh"
#include "simulation/util.cuh"
#include <stdlib.h>
#include <cstddef>

// --- Utility Functions ---

void checkIfCopyPossible(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	if (!ps_host || !ps_device) {
		fprintf(stderr, "Error: Invalid ParticleSystem pointers for copy operation.\n");
		exit(-100);
	}
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));

	if (numParticles > ps_host->maxParticles || numParticles > ps_d_copy.maxParticles) {
		fprintf(stderr, "Error: Particle count (%u) must be smaller or equal to maxParticles.\n", numParticles);
		exit(-101);
	}
}

// --- Host Memory Functions ---

__host__ ParticleSystem *ParticleSystem_CreateOnHost(int maxParticles) {
	ParticleSystem *ps = (ParticleSystem *)malloc(sizeof(ParticleSystem));
	if (!ps) {
		fprintf(stderr, "Host memory allocation failed for ParticleSystem struct.\n");
		exit(-1);
	}

	ps->maxParticles = maxParticles;
	ps->numParticles = 0;

	if (ps->maxParticles > 0) {
		size_t numBytes_float4 = ps->maxParticles * sizeof(float4);
		size_t numBytes_float = ps->maxParticles * sizeof(float);

		ps->pos = (float4 *)malloc(numBytes_float4);
		ps->vel = (float4 *)malloc(numBytes_float4);
		ps->force = (float4 *)malloc(numBytes_float4);
		ps->mass = (float *)malloc(numBytes_float);
		ps->density = (float *)malloc(numBytes_float);
		ps->pressure = (float *)malloc(numBytes_float);
		ps->normal = (float4 *)malloc(numBytes_float4);
		ps->color_laplacian = (float *)malloc(numBytes_float);

		if (!ps->pos || !ps->vel || !ps->force || !ps->mass ||
		    !ps->density || !ps->pressure || !ps->normal || !ps->color_laplacian) {
			fprintf(stderr, "Host memory allocation failed for particle arrays.\n");
			exit(EXIT_FAILURE);
		}
	} else {
		ps->pos = nullptr;
		ps->vel = nullptr;
		ps->force = nullptr;
		ps->mass = nullptr;
		ps->density = nullptr;
		ps->pressure = nullptr;
		ps->normal = nullptr;
		ps->color_laplacian = nullptr;
	}
	return ps;
}

__host__ void ParticleSystem_FreeOnHost(ParticleSystem *ps) {
	if (ps) {
		if (ps->pos) free(ps->pos);
		if (ps->vel) free(ps->vel);
		if (ps->force) free(ps->force);
		if (ps->mass) free(ps->mass);
		if (ps->density) free(ps->density);
		if (ps->pressure) free(ps->pressure);
		if (ps->normal) free(ps->normal);
		if (ps->color_laplacian) free(ps->color_laplacian);
		free(ps);
	}
}

// --- Copy Host to Device Functions ---

__host__ void ParticleSystem_CopyAll_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);

	if (numParticles == 0) return;

	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));

	size_t numBytes_float4 = numParticles * sizeof(float4);
	size_t numBytes_float = numParticles * sizeof(float);

	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.pos, ps_host->pos, numBytes_float4, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.vel, ps_host->vel, numBytes_float4, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.force, ps_host->force, numBytes_float4, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.mass, ps_host->mass, numBytes_float, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.density, ps_host->density, numBytes_float, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.pressure, ps_host->pressure, numBytes_float, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.normal, ps_host->normal, numBytes_float4, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.color_laplacian, ps_host->color_laplacian, numBytes_float, cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyPos_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.pos, ps_host->pos, numParticles * sizeof(float4), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyVel_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.vel, ps_host->vel, numParticles * sizeof(float4), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyForce_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.force, ps_host->force, numParticles * sizeof(float4), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyMass_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.mass, ps_host->mass, numParticles * sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyDensity_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.density, ps_host->density, numParticles * sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyPressure_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.pressure, ps_host->pressure, numParticles * sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyNormal_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.normal, ps_host->normal, numParticles * sizeof(float4), cudaMemcpyHostToDevice));
}

__host__ void ParticleSystem_CopyColorLaplacian_HostToDevice(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_d_copy.color_laplacian, ps_host->color_laplacian, numParticles * sizeof(float), cudaMemcpyHostToDevice));
}

// --- Device Memory Functions ---

__host__ ParticleSystem *ParticleSystem_CreateOnDevice(int maxParticles) {
	ParticleSystem *d_ps;
	CHECK_CUDA_ERROR(cudaMalloc((void **)&d_ps, sizeof(ParticleSystem)));

	// Create a temporary host-side struct to configure before copying to device
	ParticleSystem h_ps;

	h_ps.maxParticles = maxParticles;
	h_ps.numParticles = 0;

	if (maxParticles > 0) {
		size_t numBytes_float4 = maxParticles * sizeof(float4);
		size_t numBytes_float = maxParticles * sizeof(float);
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.pos, numBytes_float4));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.vel, numBytes_float4));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.force, numBytes_float4));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.mass, numBytes_float));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.density, numBytes_float));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.pressure, numBytes_float));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.normal, numBytes_float4));
		CHECK_CUDA_ERROR(cudaMalloc((void **)&h_ps.color_laplacian, numBytes_float));
	}
	CHECK_CUDA_ERROR(cudaMemcpy(d_ps, &h_ps, sizeof(ParticleSystem), cudaMemcpyHostToDevice));
	return d_ps;
}

__host__ void ParticleSystem_FreeOnDevice(ParticleSystem *ps_device) {
	if (ps_device) {
		ParticleSystem ps;
		CHECK_CUDA_ERROR(cudaMemcpy(&ps, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
		if (ps.maxParticles > 0) {
			CHECK_CUDA_ERROR(cudaFree(ps.pos));
			CHECK_CUDA_ERROR(cudaFree(ps.vel));
			CHECK_CUDA_ERROR(cudaFree(ps.force));
			CHECK_CUDA_ERROR(cudaFree(ps.mass));
			CHECK_CUDA_ERROR(cudaFree(ps.density));
			CHECK_CUDA_ERROR(cudaFree(ps.pressure));
			CHECK_CUDA_ERROR(cudaFree(ps.normal));
			CHECK_CUDA_ERROR(cudaFree(ps.color_laplacian));
		}
		CHECK_CUDA_ERROR(cudaFree(ps_device));
	}
}

// --- Device Accessor Functions ---

__host__ void ParticleSystem_SetNumParticlesOnDevice(ParticleSystem *ps_device, int numParticles) {
	CHECK_CUDA_ERROR(cudaMemcpy((char *)ps_device + offsetof(ParticleSystem, numParticles), &numParticles, sizeof(int), cudaMemcpyHostToDevice));
}

__host__ unsigned int ParticleSystem_GetNumParticlesOnDevice(ParticleSystem *ps_device) {
	unsigned int numParticles;
	CHECK_CUDA_ERROR(cudaMemcpy(&numParticles, (char *)ps_device + offsetof(ParticleSystem, numParticles), sizeof(unsigned int), cudaMemcpyDeviceToHost));
	return numParticles;
}

// --- Copy Device to Host Functions ---

__host__ void ParticleSystem_CopyAll_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);

	if (numParticles == 0) return;

	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));

	size_t numBytes_float4 = numParticles * sizeof(float4);
	size_t numBytes_float = numParticles * sizeof(float);

	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->pos, ps_d_copy.pos, numBytes_float4, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->vel, ps_d_copy.vel, numBytes_float4, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->force, ps_d_copy.force, numBytes_float4, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->mass, ps_d_copy.mass, numBytes_float, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->density, ps_d_copy.density, numBytes_float, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->pressure, ps_d_copy.pressure, numBytes_float, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->normal, ps_d_copy.normal, numBytes_float4, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->color_laplacian, ps_d_copy.color_laplacian, numBytes_float, cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyPos_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->pos, ps_d_copy.pos, numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyVel_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->vel, ps_d_copy.vel, numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyForce_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->force, ps_d_copy.force, numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyMass_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->mass, ps_d_copy.mass, numParticles * sizeof(float), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyDensity_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->density, ps_d_copy.density, numParticles * sizeof(float), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyPressure_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->pressure, ps_d_copy.pressure, numParticles * sizeof(float), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyNormal_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->normal, ps_d_copy.normal, numParticles * sizeof(float4), cudaMemcpyDeviceToHost));
}

__host__ void ParticleSystem_CopyColorLaplacian_DeviceToHost(ParticleSystem *ps_host, ParticleSystem *ps_device, unsigned int numParticles) {
	checkIfCopyPossible(ps_host, ps_device, numParticles);
	if (numParticles == 0) return;
	ParticleSystem ps_d_copy;
	CHECK_CUDA_ERROR(cudaMemcpy(&ps_d_copy, ps_device, sizeof(ParticleSystem), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host->color_laplacian, ps_d_copy.color_laplacian, numParticles * sizeof(float), cudaMemcpyDeviceToHost));
}
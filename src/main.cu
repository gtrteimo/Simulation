#include "simulation/simulation.cuh"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

void write_to_csv(const std::string &filename, float4 *positions, size_t num_positions) {
	std::cout << "Writing " << num_positions << " points to " << filename << "..." << std::endl;

	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
		return;
	}

	file << "x;y;z;w\n";

	// Write every 10th particle to keep file size manageable
	for (size_t i = 0; i < num_positions; i+= 1) {
		file << positions[i].x << ";" << positions[i].y << ";" << positions[i].z << ";" << positions[i].w << "\n";
	}

	file.close();
}

int main(void) {
	// Initialize CUDA
	cudaError_t err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
		return -1;
	}

	const float particleMass = 1.0f;
	const int particlesX = 50;
	const int particlesY = 50;
	const int particlesZ = 50;
	const int maxParticles = particlesX * particlesY * particlesZ;

	Simulation *sim = Simulation_Create(maxParticles);

	const float particleSpacing = 0.02f;

	const float cubeWidth = static_cast<float>(particlesX -1) * particleSpacing;
	const float cubeHeight = static_cast<float>(particlesY - 1) * particleSpacing;
	const float cubeDepth = static_cast<float>(particlesZ - 1) * particleSpacing;

	const float startX = -cubeWidth/2;
	const float startY = -cubeHeight/2;
	const float startZ = -cubeDepth/2;

	int p_idx = 0;
	for (int i = 0; i < particlesX; ++i) {
		for (int j = 0; j < particlesY; ++j) {
			for (int k = 0; k < particlesZ; ++k) {
				if (p_idx >= maxParticles) break;
				/*
				float x = startX + static_cast<float>(i) * particleSpacing;
				float y = startY + static_cast<float>(j) * particleSpacing;
				float z = startZ + static_cast<float>(k) * particleSpacing;
				*/
				float x = startX + std::rand()%particlesX * particleSpacing;
				float y = startY + std::rand()%particlesY * particleSpacing;
				float z = startZ + std::rand()%particlesZ * particleSpacing;

				sim->host_ps->pos[p_idx] = make_float4(x, y, z, 0.0f);
				sim->host_ps->vel[p_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				sim->host_ps->mass[p_idx] = particleMass;

				p_idx++;
			}
		}
	}
    // Set the initial number of particles to all the ones we just created.
    sim->host_ps->numParticles = p_idx;

	sim->host_params->smoothingRadius = particleSpacing * 2.0f;
	sim->host_params->restDensity = 4.0f * (particleMass * maxParticles) / (cubeWidth * cubeHeight * cubeDepth);
	sim->host_params->gasConstantK = 20.0f;
	sim->host_params->viscosityCoefficient = 0.05f;
	sim->host_params->boundaryDamping = 0.60f;
    sim->host_params->gravity = make_float4(0.0f, 0.0f, -9.81f, 0.0f);

	SimulationParams_PrecomputeKernelCoefficients(*sim->host_params);

    Grid_CalculateParams(sim->host_grid, sim->host_params);

	Simulation_CopyAll_HostToDevice(sim);

	Simulation_CopyParticles_DeviceToHost(sim);

	printf("Starting Simulation with %d particles.\n", sim->host_ps->numParticles);

	auto start_time = std::chrono::high_resolution_clock::now();

	const int num_steps = 15000;
	const float dt = 0.0004f;

	Simulation_SetActiveParticles(sim, 1);

	int step = 1;
	printf("--- Starting Simulation ---\n");
	while (step < num_steps) {
		Simulation_Step(sim, dt);

        if (step % 250 == 0) {
			Simulation_SetActiveParticles(sim, step);
		    Simulation_CopyParticles_DeviceToHost(sim);
		    write_to_csv("output_" + std::to_string(step) + ".csv", sim->host_ps->pos, sim->host_ps->numParticles);
        }
		printf("Step %d / %d completed. (Active Particles: %u)\r", step, num_steps, sim->host_ps->numParticles);
        fflush(stdout);
		step++;
	}

	printf("\nSimulation steps completed.\n");
	cudaDeviceSynchronize();

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
	printf("\n--- Simulation Finished ---\n");
	printf("Total execution time: %.2f ms (%.2f ms/step)\n", duration_ms.count(), duration_ms.count()/num_steps);

	Simulation_Free(sim);
	cudaDeviceReset();
	return 0;
}
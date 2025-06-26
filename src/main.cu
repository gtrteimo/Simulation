#include "cuda/simulation.cuh"
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

	// Header for CSV
	file << "x;y;z;w\n";

	for (size_t i = 0; i < num_positions; ++i) {
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

	// --- SETUP PARTICLE INITIAL POSITIONS ---

	const int particlesX = 10;
	const int particlesY = 10;
	const int particlesZ = 10;
	const int numParticles = particlesX * particlesY * particlesZ;

	Simulation *sim = Simulation_Create(numParticles);

	const float particleSpacing = 0.05f;

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
				if (p_idx >= numParticles) break;

				float x = startX + static_cast<float>(i) * particleSpacing;
				float y = startY + static_cast<float>(j) * particleSpacing;
				float z = startZ + static_cast<float>(k) * particleSpacing;

				sim->host_ps->pos[p_idx] = make_float4(x, y, z, 0.0f);
				sim->host_ps->vel[p_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

				sim->host_ps->mass[p_idx] = 1000.0f;

				p_idx++;
			}
		}
	}
    // Make sure the host particle count is correct
    sim->host_ps->numParticles = p_idx;

	// --- APPLY STABLE SIMULATION PARAMETERS ---
	sim->host_params->smoothingRadius = particleSpacing * 2.0f; // A larger radius helps prevent particle pairing and improves stability
	sim->host_params->restDensity = 1000.0f / (cubeDepth* cubeWidth * cubeHeight / numParticles);
	sim->host_params->gasConstantK = 200.0f;          // Stiffness constant. Higher = less compressible.
	sim->host_params->viscosityCoefficient = 0.05f;   // Damps oscillations.
	sim->host_params->wallStiffness = 10000.0f;       // Penalty force stiffness for boundaries
	sim->host_params->boundaryDamping = -0.9f * 2.0f * sqrt(sim->host_ps->mass[0] * sim->host_params->wallStiffness); // Critical damping
    sim->host_params->gravity = make_float4(0.0f, -9.81f, 0.0f, 0.0f);

	SimulationParams_PrecomputeKernelCoefficients(*sim->host_params);
    Grid_CalculateParams(sim->host_grid, sim->host_params); // Recalculate grid params on host

	Simulation_CopyAll_HostToDevice(sim);

	// Test writing initial state
	Simulation_CopyParticles_DeviceToHost(sim);
	write_to_csv("output_0.csv", sim->host_ps->pos, sim->host_ps->numParticles);

	printf("Starting Simulation with %d particles.\n", p_idx);

	auto start_time = std::chrono::high_resolution_clock::now();

	const int num_steps = 5000;
	// CRITICAL: A very small fixed time step is required for stability without an adaptive scheme.
	// This is a common source of explosions.
	const float dt = 0.001f;

	for (int step = 1; step <= num_steps; ++step) {
		Simulation_Step(sim, dt);

        if (step % 50 == 0) {
		    Simulation_CopyParticles_DeviceToHost(sim);
		    write_to_csv("output_" + std::to_string(step) + ".csv", sim->host_ps->pos, sim->host_ps->numParticles);
        }
		printf("Step %d / %d completed. (dt=%.5f)\r", step, num_steps, dt);
        fflush(stdout);
	}

	printf("\nSimulation steps completed.\n");
	cudaDeviceSynchronize(); // Wait for all GPU work to finish before stopping the timer

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
	printf("\n--- Simulation Finished ---\n");
	printf("Total execution time: %.2f ms (%.2f ms/step)\n", duration_ms.count(), duration_ms.count()/num_steps);

	Simulation_Free(sim);
	cudaDeviceReset();
	return 0;
}
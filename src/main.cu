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

	for (size_t i = 0; i < num_positions; ++i) {
		file << positions[i].x << ";" << positions[i].y << ";" << positions[i].z << ";" << positions[i].w << "\n";
	}

	file.close();
	// Commented out to reduce console spam during simulation
	// std::cout << "Successfully wrote CSV file." << std::endl;
}

int main(void) {
	// Initialize CUDA
	cudaError_t err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
		return -1;
	}

	// --- SETUP PARTICLE INITIAL POSITIONS ---
	// Create a 0.4m x 0.4m x 0.4m cube of particles in a corner of the domain
	// This is a classic "dam break" scenario which is much more stable.
	const int particlesX = 45;
	const int particlesY = 45;
	const int particlesZ = 45;
    const int numParticles = particlesX * particlesY * particlesZ;

	Simulation *sim = Simulation_Create(numParticles);

	const float cubeDim = 0.4f;
    const float particleSpacing = cubeDim / (particlesX - 1);
	
    // The domain is [-0.5, 0.5]. Let's place the cube on the "floor" at y=-0.5
    // and in the corner at x=-0.5, z=-0.5.
    const float startX = -0.5f;
    const float startY = -0.5f;
    const float startZ = -0.5f;


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

                // Calculate mass to match rest density
                // V_particle = spacing^3; mass = rho * V_particle
                float particleVolume = particleSpacing * particleSpacing * particleSpacing;
				sim->host_ps->mass[p_idx] = 1000.0f * particleVolume;

				p_idx++;
			}
		}
	}
    // Make sure the host particle count is correct
    sim->host_ps->numParticles = p_idx;


	// --- APPLY STABLE SIMULATION PARAMETERS ---
    // These values are much less "explosive" than the previous set.
	sim->host_params->smoothingRadius = particleSpacing * 1.5f; // A good rule of thumb
	sim->host_params->restDensity = 1000.0f;
	sim->host_params->gasConstantK = 50.0f;          // SIGNIFICANTLY REDUCED: Lower stiffness reduces pressure forces.
	sim->host_params->viscosityCoefficient = 0.1f;   // SLIGHTLY INCREASED: More viscosity helps damp oscillations.
	sim->host_params->wallStiffness = 3000.0f;       // REDUCED: Less violent wall collisions.
    sim->host_params->gravity = make_float4(0.0f, -9.81f, 0.0f, 0.0f); // Standard gravity

	SimulationParams_PrecomputeKernelCoefficients(*sim->host_params);

	Simulation_CopyAll_HostToDevice(sim);

	printf("Start Simulation with %d particles.\n", p_idx);

	auto start_time = std::chrono::high_resolution_clock::now();

	const int num_steps = 10000;
	const float dt = 0.0001f;

	for (int step = 0; step <= num_steps; ++step) {
		Simulation_Step(sim, dt);

        if (step % 1000 == 0) {
		    Simulation_CopyParticles_DeviceToHost(sim);
		    write_to_csv("output_" + std::to_string(step) + ".csv", sim->host_ps->pos, sim->host_ps->numParticles);
        }
		printf("Step %d / %d completed. (dt=%.5f)\r", step, num_steps, dt);
        fflush(stdout);
	}

	printf("\nSimulation steps completed.\n");

	cudaDeviceSynchronize();

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
	printf("\n--- Simulation Finished ---\n");
	printf("Total execution time: %.2f ms\n", duration_ms.count());

	Simulation_Free(sim);

	cudaDeviceReset();
	return 0;
}
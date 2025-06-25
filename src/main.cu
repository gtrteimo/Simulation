#include "cuda/simulation.cuh"
#include <chrono>

int main(void) {
	// Initialize CUDA
	cudaError_t err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
		return -1;
	}
	printf("CUDA device set successfully.\n");
	Simulation *sim = Simulation_Create(1000000);
	printf("Simulation created with %d particles.\n", sim->host_ps->numParticles);

	int p_idx = 0;
	for (int i = 0; i < 100; ++i) {
		for (int j = 0; j < 100; ++j) {
			for (int k = 0; k < 100; ++k) {
				float x = (static_cast<float>(i) - 50.0f);
				float y = (static_cast<float>(j) - 50.0f);
				float z = (static_cast<float>(k) - 50.0f);

				sim->host_ps->pos[p_idx] = make_float4(x, y, z, 0.0f);
				sim->host_ps->vel[p_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				sim->host_ps->mass[p_idx] = 0.01f;

				p_idx++; // I hate this little bitch of variable: I accidentally incremented it before using and that caused heap metadata corruption that didn't cause a crash until the free, which took hours to find
			}
		}
	}
	sim->host_params->restDensity = 80000.0f; // Set a reasonable rest density

	printf("Initialized %d particles with positions and velocities.\n", p_idx);

	Simulation_CopyAll_HostToDevice(sim);

	printf("Copied all simulation data from host to device.\n");

	auto start_time = std::chrono::high_resolution_clock::now();

	printf("Starting simulation steps...\n");
	printf("host sim = %p\n", sim->host_sim);
	printf("device sim = %p\n", sim->device_sim);

	for (int step = 0; step < 10; ++step) {
		Simulation_Step(sim, 0.01f);

		printf("Step %d / %d completed.\n", step + 1, 10);
	}

	printf("Simulation steps completed.\n");

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	printf("All simulation steps synchronized.\n");

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
	printf("\n--- Simulation Finished ---\n");
	printf("Total execution time: %.2f ms\n", duration_ms.count());

	Simulation_CopyParticles_DeviceToHost(sim);

	printf("Copied particle data from device to host.\n");

	Simulation_Free(sim);

	printf("Simulation resources freed.\n");

	cudaDeviceReset();
	return 0;
}
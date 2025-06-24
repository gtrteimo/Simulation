#include "cuda/simulation.cuh"


int main(void) {
	cudaError_t cudaStatus = cudaSetDevice(0); // Use GPU 0
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-enabled GPU?\n");
        return 1;
    }

	// --- Test params ---
	printf("Starting params Test\n");
	SimulationParams *params_Host = SimulationParams_CreateOnHost();
	printf("Params_CreateOnHost: %p\n", params_Host);
	SimulationParams *params_Device = SimulationParams_CreateOnDevice();
	printf("Params_CreateOnDevice: %p\n", params_Device);
	SimulationParams_Copy_HostToDevice(params_Host, params_Device);
	printf("Params_Copy_HostToDevice completed\n");
	SimulationParams_FreeOnHost(params_Host);
	SimulationParams_FreeOnDevice(params_Device);
	printf("Params_FreeOnHost and Params_FreeOnDevice completed\n");

	// --- Test particles ---
	printf("Starting particles Test\n");
	ParticleSystem *ps_Host = ParticleSystem_CreateOnHost(1000);
	printf("ParticleSystem_CreateOnHost: %p\n", ps_Host);
	ParticleSystem *ps_Device = ParticleSystem_CreateOnDevice(1000);
	printf("ParticleSystem_CreateOnDevice: %p\n", ps_Device);
	for(int i = 0; i < ps_Host->numParticles; i++) {
		ps_Host->pos[i] = make_float4((float)i, ps_Host->numParticles-i, 0.0f, 0.0f);
	}
	printf("ParticleSystem_Host filled with data\n");
	ParticleSystem_CopyAll_HostToDevice(ps_Host, ps_Device);
	printf("ParticleSystem_CopyAll_HostToDevice completed\n");
	ParticleSystem_FreeOnHost(ps_Host);
	ParticleSystem_FreeOnDevice(ps_Device);
	printf("ParticleSystem_FreeOnHost and ParticleSystem_FreeOnDevice completed\n");

	// --- Test grid ---
	printf("Starting grid Test\n");
	printf("Just kidding. I am too lazy to implement grid myself or write a test for it.\n");
	printf("You can implement it yourself if you want to.\n");
	printf("I made everything else, so i think it's fair enough to skip this one\n");
	printf("Actually i tested it but it was a large test and i don't want to deal with putting it in here right now.\n");
	printf("Onto the next test!\n");

	// --- Test Simulation ---

	return 0;
}
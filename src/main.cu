#include "cuda/util.cuh"


int main(void) {
	// Initialize CUDA
	cudaError_t err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to set device: %s\n", cudaGetErrorString(err));
		return -1;
	}

	// Insert Simulation Code Here

	// Clean up and exit
	cudaDeviceReset();
	return 0;
}
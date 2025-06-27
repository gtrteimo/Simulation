#include "simulation/params.cuh"
#include "simulation/util.cuh"

// --- Host Memory Functions ---

__host__ SimulationParams *SimulationParams_CreateOnHost() {
	SimulationParams *ps = (SimulationParams *)malloc(sizeof(SimulationParams));
	if (!ps) {
		fprintf(stderr, "Host memory allocation failed for SimulationParams struct.\n");
		exit(-1);
	}
	ps->smoothingRadius = SimulationParams_Default.smoothingRadius;
	ps->gasConstantK = SimulationParams_Default.gasConstantK;
	ps->restDensity = SimulationParams_Default.restDensity;
	ps->viscosityCoefficient = SimulationParams_Default.viscosityCoefficient;
	ps->surfaceTensionCoefficient = SimulationParams_Default.surfaceTensionCoefficient;
	ps->surfaceTensionThreshold = SimulationParams_Default.surfaceTensionThreshold;
	ps->gravity = SimulationParams_Default.gravity;
	ps->min = SimulationParams_Default.min;
	ps->max = SimulationParams_Default.max;
	ps->boundaryDamping = SimulationParams_Default.boundaryDamping;
	ps->wallStiffness = SimulationParams_Default.wallStiffness;
	SimulationParams_PrecomputeKernelCoefficients(*ps);
	return ps;
}

__host__ void SimulationParams_FreeOnHost(SimulationParams *ps) {
	if (ps) {
		free(ps);
	}
}

// --- Copy Host to Device Functions ---

__host__ void SimulationParams_Copy_HostToDevice(SimulationParams *ps_host, SimulationParams *ps_device) {
	CHECK_CUDA_ERROR(cudaMemcpy(ps_device, ps_host, sizeof(SimulationParams), cudaMemcpyHostToDevice));
}

// --- Device Memory Functions ---

__host__ SimulationParams *SimulationParams_CreateOnDevice() {
	SimulationParams *ps;
	CHECK_CUDA_ERROR(cudaMalloc((void **)&ps, sizeof(SimulationParams)));
	CHECK_CUDA_ERROR(cudaMemcpy(ps, &SimulationParams_Default, sizeof(SimulationParams), cudaMemcpyHostToDevice));
	return ps;
}

__host__ void SimulationParams_FreeOnDevice(SimulationParams *ps) {
	if (ps) {
		CHECK_CUDA_ERROR(cudaFree(ps));
	}
}

// --- Copy Device to Host Functions ---

__host__ void SimulationParams_Copy_DeviceToHost(SimulationParams *ps_device, SimulationParams *ps_host) {
	CHECK_CUDA_ERROR(cudaMemcpy(ps_host, ps_device, sizeof(SimulationParams), cudaMemcpyDeviceToHost));
}

// --- Utility Functions ---

// Call this if smoothingRadius is changed after construction
__host__ void SimulationParams_PrecomputeKernelCoefficients(SimulationParams &params) {
	if (params.smoothingRadius > 1e-6f) { // Avoid division by zero
		const float PI_F = static_cast<float>(M_PI);
		float h = params.smoothingRadius;
		float h2 = h * h;
		float h3 = h2 * h;
		float h6 = h3 * h3;
		float h9 = h3 * h6;

		params.smoothingRadiusSq = h2;
		params.poly6KernelCoeff = 315.0f / (64.0f * PI_F * h9);
		params.spikyKernelGradientCoeff = -45.0f / (PI_F * h6);
		params.viscosityKernelLaplacianCoeff = 45.0f / (PI_F * h6);
		
		// --- ADDED FOR SURFACE TENSION ---
		// Gradient of Poly6 kernel
		params.poly6KernelGradientCoeff = -945.0f / (32.0f * PI_F * h9);
		// Laplacian of Poly6 kernel
		params.poly6KernelLaplacianCoeff = 945.0f / (32.0f * PI_F * h9);
	} else {
		// ... (zeroing out coefficients)
		params.poly6KernelGradientCoeff = 0.0f;
		params.poly6KernelLaplacianCoeff = 0.0f;
	}
}
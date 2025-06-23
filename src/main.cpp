#include "cuda/particles.cuh"

int main(void) {
	printf("Starting Particle System Example\n");
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
	return 0;
}
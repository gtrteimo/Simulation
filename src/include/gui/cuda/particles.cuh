#pragma once

struct ParticleSystem {
    float *pos_x, *pos_y, *pos_z;
    float *vel_x, *vel_y, *vel_z;
    float *force_x, *force_y, *force_z;
    float *mass;
    float *density;
    float *pressure;
    float *normal_x, *normal_y, *normal_z;
    float *color_laplacian;
    int numParticles;
};

// --- Host Functions ---

ParticleSystem* ParticleSystem_Host_Init(int numParticles);
void ParticleSystem_Host_Free(ParticleSystem* ps);

void ParticleSystem_Host_CopyAllToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Host_CopyPosToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyVelToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyForceToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyMassToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyDensityToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyPressureToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyNormalToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);
void ParticleSystem_Host_CopyColorLaplacianToDevice(ParticleSystem* ps_device, ParticleSystem* ps_host);

// --- Device Functions ---

ParticleSystem* ParticleSystem_Device_Init(int numParticles);
void ParticleSystem_Device_Free(ParticleSystem* ps);

void ParticleSystem_Device_CopyAllToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyPosToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyVelToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyForceToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyMassToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyDensityToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyPressureToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyNormalToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);
void ParticleSystem_Device_CopyColorLaplacianToDevice(ParticleSystem* ps_host, ParticleSystem* ps_device);

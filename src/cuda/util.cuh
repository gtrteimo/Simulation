#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define M_PI 3.14159265358979323846f

// --- CUDA Error Checking Macro ---
// This macro simplifies checking for CUDA errors after API calls
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      fflush(stderr);
      if (abort) exit(code);
   }
}

// --- float4 Operators ---
__device__ __host__ __forceinline__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __host__ __forceinline__ float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __host__ __forceinline__ void operator+=(float4 &a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __host__ __forceinline__ void operator-=(float4 &a, float4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; }
__device__ __host__ __forceinline__ float4 operator*(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __host__ __forceinline__ float4 operator*(float s, float4 a) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __host__ __forceinline__ float4 operator/(float4 a, float s) { float inv_s = 1.0f / s; return make_float4(a.x * inv_s, a.y * inv_s, a.z * inv_s, a.w * inv_s); }
__device__ __host__ __forceinline__ float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __host__ __forceinline__ float length(float4 v) { return sqrtf(dot(v, v)); }

// --- int4 Operators ---

__device__ __host__ __forceinline__ int4 operator+(int4 a, int4 b) { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __host__ __forceinline__ int4 operator-(int4 a, int4 b) { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __host__ __forceinline__ void operator+=(int4 &a, int4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __host__ __forceinline__ void operator-=(int4 &a, int4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; }
__device__ __host__ __forceinline__ int4 operator*(int4 a, int s) { return make_int4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __host__ __forceinline__ int4 operator*(int s, int4 a) { return make_int4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __host__ __forceinline__ int4 operator/(int4 a, int s) { return make_int4(a.x / s, a.y / s, a.z / s, a.w / s); }
__device__ __host__ __forceinline__ int dot(int4 a, int4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __host__ __forceinline__ float length(int4 v) { return sqrtf(dot(v, v)); }
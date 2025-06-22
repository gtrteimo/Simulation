#pragma once

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
      if (abort) exit(code);
   }
}
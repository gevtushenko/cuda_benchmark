#include <iostream>
#include "cuda_benchmark.h"

int main ()
{
  cudaSetDevice (1);

  float *in_f {};
  cudaMalloc (&in_f, 2 * sizeof (float));

  float *in_d {};
  cudaMalloc (&in_d, 2 * sizeof (float));

  benchmark ([=] __device__ (cuda_benchmark::state &state)
  {
    float a = in_f[threadIdx.x];
    float b = in_f[threadIdx.x + 1];

    for (auto _ : state)
      a = a + b;

    in_f[0] = (a + b);
  });

  benchmark ([=] __device__ (cuda_benchmark::state &state)
  {
    double a = in_d[threadIdx.x];
    double b = in_d[threadIdx.x + 1];

    for (auto _ : state)
      a = a + b;

    in_d[0] = (a + b);
  });

  cudaFree (in_f);
  cudaFree (in_d);

  return 0;
}

#include <iostream>
#include "include/cuda_benchmark.h"

#define REPEAT2(x)  x x
#define REPEAT4(x)  REPEAT2(x) REPEAT2(x)
#define REPEAT8(x)  REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)

int main ()
{
  cuda_benchmark::controller controller (1024, 1);

  float *in_f {};
  cudaMalloc (&in_f, 2 * sizeof (float));

  float *in_d {};
  cudaMalloc (&in_d, 2 * sizeof (float));

  controller.benchmark ("float add", [=] __device__ (cuda_benchmark::state &state)
  {
    float a = in_f[threadIdx.x];
    float b = in_f[threadIdx.x + 1];

    for (auto _ : state)
      {
        REPEAT32(a = a + b;);
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in_f[0] = (a + b);
  });

  controller.benchmark ("float div", [=] __device__ (cuda_benchmark::state &state)
  {
    float a = in_f[threadIdx.x];
    float b = in_f[threadIdx.x + 1];

    for (auto _ : state)
      {
        REPEAT32(a = a / b;);
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in_f[0] = (a + b);
  });

  controller.benchmark ("double add", [=] __device__ (cuda_benchmark::state &state)
  {
    double a = in_d[threadIdx.x];
    double b = in_d[threadIdx.x + 1];

    for (auto _ : state)
      {
        REPEAT32(a = a + b;);
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in_d[0] = (a + b);
  });

  controller.benchmark ("double div", [=] __device__ (cuda_benchmark::state &state)
  {
    double a = in_d[threadIdx.x];
    double b = in_d[threadIdx.x + 1];

    for (auto _ : state)
      {
        REPEAT32(a = a / b;);
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in_d[0] = (a + b);
  });

  cudaFree (in_f);
  cudaFree (in_d);

  return 0;
}

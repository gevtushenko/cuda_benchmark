#include "cuda_benchmark.h"

#include <memory>
#include <numeric>

int main ()
{
  int n = 1024;

  int *device_a {};
  int *device_b {};
  int *device_permutation {};

  cudaMalloc (&device_a, n * sizeof (int));
  cudaMalloc (&device_b, n * sizeof (int));
  cudaMalloc (&device_permutation, n * sizeof (int));

  cudaMemset (device_a, 0, n * sizeof (int));
  cudaMemset (device_b, 0, n * sizeof (int));

  std::unique_ptr<int> host_permutation (new int[n]);
  std::iota (host_permutation.get (), host_permutation.get () + n, 0);
  cudaMemcpy (device_permutation, host_permutation.get (), n * sizeof (int), cudaMemcpyHostToDevice);


  cuda_benchmark::controller controller;

  controller.benchmark ("coalesced", [=] __device__ (cuda_benchmark::state &state) {
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (auto _ : state)
      device_b[i] = device_a[device_permutation[i]];
  }, 0, 1);


  cudaFree (device_a);
  cudaFree (device_b);
  cudaFree (device_permutation);
}
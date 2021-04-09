#include "cuda_benchmark.h"

#include <memory>
#include <numeric>
#include <fstream>

template <typename data_type>
void reminder_test ()
{
  const unsigned int block_size = 128;
  const unsigned int grid_size = 20;
  int n = block_size * grid_size + block_size / 2;

  data_type *device_a {};
  data_type *device_b {};
  int *device_permutation {};

  cudaMalloc (&device_a, n * sizeof (data_type));
  cudaMalloc (&device_b, n * sizeof (data_type));
  cudaMalloc (&device_permutation, n * sizeof (int));

  cudaMemset (device_a, 0, n * sizeof (data_type));
  cudaMemset (device_b, 0, n * sizeof (data_type));

  std::unique_ptr<int> host_permutation (new int[n]);
  std::iota (host_permutation.get (), host_permutation.get () + n, 0);
  cudaMemcpy (device_permutation, host_permutation.get (), n * sizeof (int), cudaMemcpyHostToDevice);


  cuda_benchmark::controller controller (block_size, grid_size);

  cuda_benchmark::warp_intervals intervals (controller.benchmark (
      "coalesced",
      [=] __device__ (cuda_benchmark::state & state) {
        const unsigned int stride = gridDim.x * blockDim.x;

        for (auto _ : state)
          for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += stride)
            device_b[i] = device_a[device_permutation[i]] + 2.0;
      },
      0, 1));

  std::ofstream csv_file ("/tmp/wl");
  intervals.store (csv_file);
  csv_file.close ();

  cudaFree (device_a);
  cudaFree (device_b);
  cudaFree (device_permutation);
}

int main ()
{
  reminder_test<double> ();
}
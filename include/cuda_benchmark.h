//
// Created by egi on 1/3/20.
//

#ifndef CUDA_BENCHMARK_H
#define CUDA_BENCHMARK_H

#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>

/*
__device__ static void do_not_optimize (const float& value)
{
  asm volatile ("" : "+f" (value) :: "memory");
}

__device__ static void do_not_optimize (const double& value)
{
  asm volatile ("" : "+d" (value) :: "memory");
}
*/

namespace cuda_benchmark
{
  class state_iterator;
  class state
  {
    unsigned long long iterations = 100;
    unsigned long long operations = iterations;
    unsigned long long clock_begin {};
    unsigned long long clock_end {};

  public:
    __device__ state_iterator begin ();
    __device__ state_iterator end ();
    __device__ unsigned long long int max_iterations () { return iterations; }
    __device__ unsigned long long int operations_processed () { return operations; }
    __device__ void set_operations_processed (unsigned long long int ops) { operations = ops; }

    __device__ void run () { clock_begin = clock64 (); }
    __device__ void complete_run () { clock_end = clock64 (); }

    __device__ unsigned long long get_clk_begin () const { return clock_begin; }
    __device__ unsigned long long get_clk_end () const { return clock_end; }

    __device__ unsigned long long count () { return (clock_end - clock_begin) / operations_processed (); }
  };

  class state_iterator
  {
    state *s {};
    unsigned long long count;

  public:
    state_iterator () = default;
    explicit __device__ state_iterator (state *s_arg) : s (s_arg), count (s->max_iterations ()) {}

    __device__ bool operator != (const state_iterator &rhs)
    {
      if (count > 0)
        return true;
      s->complete_run ();
      return false;
    }

    __device__ state_iterator &operator++ ()
    {
      count--;
      return *this;
    }

    __device__ bool operator* () const { return false; }
  };

  inline __device__ state_iterator state::begin ()
  {
    return state_iterator (this);
  }

  inline __device__ state_iterator state::end ()
  {
    __syncthreads ();
    run ();
    return state_iterator ();
  }

  template <typename lambda_type>
  __global__ void benchmark_kernel (
      unsigned long long *clk_begin,
      unsigned long long *clk_end,
      unsigned long long *iterations,
      const lambda_type action)
  {
    cuda_benchmark::state state;
    action (state);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    clk_begin[tid] = state.get_clk_begin ();
    clk_end[tid] = state.get_clk_end ();
    iterations[tid] = state.operations_processed ();
  }

  class controller
  {
    class result
    {
    public:
      result (
        const std::string benchmark_name_arg,
        const unsigned long long latency_arg,
        const float throughput_arg,
        const long iterations_arg)
        : benchmark_name (benchmark_name_arg)
        , latency (latency_arg)
        , throughput (throughput_arg)
        , iterations (iterations_arg)
      { }

      const std::string benchmark_name;
      const unsigned long long latency {};
      const float throughput {};
      const unsigned long long iterations {};
    };

    const int gpu_id {};
    const int default_block_size {};
    std::vector<result> results;
    unsigned long long int *gpu_array {};
    std::unique_ptr<unsigned long long int[]> cpu_array {};

  public:
    explicit controller (int block_size = 1024, int gpu_id_arg = 0)
      : gpu_id (gpu_id_arg)
      , default_block_size (block_size)
    {
      cudaSetDevice (gpu_id);
      cudaMalloc (&gpu_array, block_size * 3 * sizeof (unsigned long long));
      cpu_array = std::make_unique<unsigned long long[]> (block_size * 3);
    }

    ~controller ();

    template <typename lambda_type>
    void benchmark (const std::string &name, const lambda_type &action)
    {
      benchmark_kernel<<<1, 1>>> (gpu_array, gpu_array + 1, gpu_array + 2, action);
      cudaMemcpy (cpu_array.get (), gpu_array, 3 * sizeof (unsigned long long), cudaMemcpyDeviceToHost);

      const unsigned long long int clk_begin = cpu_array[0];
      const unsigned long long int clk_end = cpu_array[1];
      const unsigned long long int operations = cpu_array[2];

      benchmark_kernel<<<1, default_block_size>>> (gpu_array, gpu_array + default_block_size, gpu_array + 2 * default_block_size, action);
      cudaMemcpy (cpu_array.get (), gpu_array, default_block_size * 2 * sizeof (unsigned long long), cudaMemcpyDeviceToHost);

      const unsigned long long int min_clk_begin = *std::min_element (cpu_array.get (), cpu_array.get () + default_block_size);
      const unsigned long long int max_clk_end = *std::max_element (cpu_array.get () + default_block_size, cpu_array.get () + 2 * default_block_size);

      results.emplace_back (
        name,
        (clk_end - clk_begin) / operations,
        static_cast<float>(operations * default_block_size) / (max_clk_end - min_clk_begin),
        operations);
    }
  };
}

#endif // CUDA_BENCHMARK_H

//
// Created by egi on 1/3/20.
//

#ifndef CUDA_BENCHMARK_H
#define CUDA_BENCHMARK_H

#include <cuda_runtime.h>

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
    unsigned long long clock_begin {};
    unsigned long long clock_end {};

  public:
    __device__ state_iterator begin ();
    __device__ state_iterator end ();
    __device__ unsigned long long int max_iterations () { return 100; }

    __device__ void run () { clock_begin = clock64 (); }
    __device__ void complete_run () { clock_end = clock64 (); }

    __device__ unsigned long long count () { return (clock_end - clock_begin) / max_iterations (); }
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
    run ();
    return state_iterator ();
  }

  template <typename lambda_type>
  __global__ void benchmark_kernel (unsigned long long *elapsed, unsigned long long *iterations, const lambda_type action)
  {
    cuda_benchmark::state state;
    action (state);
    elapsed[0] = state.count ();
    iterations[0] = state.max_iterations ();
  }

  class controller
  {
    class result
    {
    public:
      result (
        const std::string benchmark_name_arg,
        const unsigned long long elapsed_arg,
        const unsigned long long iterations_arg)
        : benchmark_name (benchmark_name_arg)
        , elapsed (elapsed_arg)
        , iterations (iterations_arg)
      { }

      const std::string benchmark_name;
      const unsigned long long elapsed {};
      const unsigned long long iterations {};
    };

    const int gpu_id {};
    std::vector<result> results;
    unsigned long long int *gpu_array {};
    std::unique_ptr<unsigned long long int[]> cpu_array {};

  public:
    explicit controller (int gpu_id_arg = 0)
      : gpu_id (gpu_id_arg)
    {
      cudaSetDevice (gpu_id);
      cudaMalloc (&gpu_array, 2 * sizeof (unsigned long long));
      cpu_array = std::make_unique<unsigned long long[]> (2);
    }

    ~controller ();

    template <typename lambda_type>
    void benchmark (const std::string &name, const lambda_type &action)
    {
      benchmark_kernel<<<1, 1>>> (gpu_array, gpu_array + 1, action);
      cudaMemcpy (cpu_array.get (), gpu_array, 2 * sizeof (unsigned long long), cudaMemcpyDeviceToHost);
      results.emplace_back (name, cpu_array[0], cpu_array[1]);
    }
  };
}

#endif // CUDA_BENCHMARK_H

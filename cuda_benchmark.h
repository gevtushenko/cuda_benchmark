//
// Created by egi on 1/3/20.
//

#ifndef CUDA_BENCHMARK_H
#define CUDA_BENCHMARK_H

#include <cuda_runtime.h>

__device__ void do_not_optimize (const float& value)
{
  asm volatile ("" : "+f" (value) :: "memory");
}

__device__ void do_not_optimize (const double& value)
{
  asm volatile ("" : "+d" (value) :: "memory");
}

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

    __device__ void run () { clock_begin = clock64 (); }
    __device__ void complete_run () { clock_end = clock64 (); }

    __device__ unsigned long long count () { return (clock_end - clock_begin) / 100; }
  };

  class state_iterator
  {
    state *s {};
    unsigned long long count = 100;

  public:
    state_iterator () = default;
    explicit __device__ state_iterator (state *s_arg) : s (s_arg) {}

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

  __device__ state_iterator state::begin ()
  {
    return state_iterator (this);
  }

  __device__ state_iterator state::end ()
  {
    run ();
    return state_iterator ();
  }
}

template <typename lambda_type>
__global__ void benchmark_kernel (unsigned long long *elapsed, const lambda_type action)
{
  cuda_benchmark::state state;
  action (state);
  elapsed[0] = state.count ();
}

template <typename lambda_type>
void benchmark (const lambda_type &action)
{
  unsigned long long int h_elapsed {};
  unsigned long long int *d_elapsed {};
  cudaMalloc (&d_elapsed, sizeof (unsigned long long));

  benchmark_kernel<<<1, 1>>> (d_elapsed, action);
  cudaMemcpy (&h_elapsed, d_elapsed, sizeof (unsigned long long), cudaMemcpyDeviceToHost);

  std::cout << "Clock: " << h_elapsed << std::endl;
}

#endif // CUDA_BENCHMARK_H

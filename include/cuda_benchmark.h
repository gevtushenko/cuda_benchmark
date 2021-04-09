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
  using interval_type = std::pair<unsigned long long int, unsigned long long int>; // clk_begin, clk_end

  __device__ inline unsigned int get_smid ()
  {
    unsigned int ret;
    asm ("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
  }

  class temporal_threads_intervals
  {
  public:
    temporal_threads_intervals (
        const unsigned int threads_count_arg,
        const unsigned long long int *const thread_clk_begin_arg,
        const unsigned long long int *const thread_clk_end_arg,
        const unsigned int *const thread_sm_ids_arg)
        : threads_count (threads_count_arg)
        , thread_clk_begin (thread_clk_begin_arg)
        , thread_clk_end (thread_clk_end_arg)
        , thread_sm_ids (thread_sm_ids_arg)
    {
    }

    const unsigned int threads_count {};
    const unsigned long long int * const thread_clk_begin {};
    const unsigned long long int * const thread_clk_end {};
    const unsigned int * const thread_sm_ids {};
  };

  class warp_intervals
  {
    const unsigned int warp_size = 32;
    const unsigned int warps_count {};

    std::vector<interval_type> intervals;
    std::vector<unsigned int> sm_ids;

  public:
    explicit warp_intervals (temporal_threads_intervals thread_intervals);

    void store (std::ostream &os);
  };

  class state_iterator;
  class state
  {
    unsigned long long iterations = 100;
    unsigned long long operations = iterations;
    unsigned long long clock_begin {};
    unsigned long long clock_end {};

  public:
    __device__ explicit state (unsigned long long int iterations_arg)
        : iterations (iterations_arg)
        , operations (iterations)
    {}

    __device__ state_iterator begin ();
    __device__ state_iterator end ();
    [[nodiscard]] __device__ unsigned long long int max_iterations () const { return iterations; }
    [[nodiscard]] __device__ unsigned long long int operations_processed () const { return operations; }
    __device__ void set_operations_processed (unsigned long long int ops) { operations = ops; }

    __device__ void run () { clock_begin = clock64 (); }
    __device__ void complete_run () { clock_end = clock64 (); }

    [[nodiscard]] __device__ unsigned long long get_clk_begin () const { return clock_begin; }
    [[nodiscard]] __device__ unsigned long long get_clk_end () const { return clock_end; }

    [[nodiscard]] __device__ unsigned long long count () const {
      return (clock_end - clock_begin) / operations_processed ();
    }
  };

  class state_iterator
  {
    state *s {};
    unsigned long long count {};

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
    return { };
  }


  template <typename lambda_type>
  __global__ void benchmark_kernel (
      unsigned long long iterations,
      unsigned long long *clk_begin,
      unsigned long long *clk_end,
      unsigned long long *operations,
      unsigned int *sm_ids,
      const lambda_type action)
  {
    cuda_benchmark::state state (iterations);
    action (state);

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    clk_begin[tid] = state.get_clk_begin ();
    clk_end[tid] = state.get_clk_end ();
    operations[tid] = state.operations_processed ();
    sm_ids[tid] = get_smid ();
  }

  class controller
  {
    class result
    {
    public:
      result (
        std::string &&benchmark_name_arg,
        const unsigned long long latency_arg,
        const float throughput_arg,
        const unsigned long long int operations_arg)
        : benchmark_name (benchmark_name_arg)
        , latency (latency_arg)
        , throughput (throughput_arg)
        , operations (operations_arg)
      { }

      const std::string benchmark_name;
      const unsigned long long latency {};
      const float throughput {};
      const unsigned long long operations {};
    };

    const int gpu_id {};
    const int default_block_size {};
    const int default_grid_size {};
    unsigned long long int *device_clk_begin {};
    unsigned long long int *device_clk_end {};
    unsigned long long int *device_iterations {};
    unsigned int *device_sm_ids {};

    std::unique_ptr<unsigned long long int[]> host_clk_begin {};
    std::unique_ptr<unsigned long long int[]> host_clk_end {};
    std::unique_ptr<unsigned long long int[]> host_iterations {};
    std::unique_ptr<unsigned int[]> host_sm_ids {};

    std::vector<result> results;

    void receive_results (size_t elements) const;

    [[nodiscard]] interval_type get_min_begin_max_end(size_t elements) const;

    void process_measurements (
            std::string &&name,
            interval_type latency_interval,
            interval_type throughput_interval);

    template <typename lambda_type>
    [[nodiscard]] interval_type measure (
            const int iterations,
            const lambda_type &action,
            unsigned int grid_size,
            unsigned int thread_block_size,
            int shared_memory_size)
    {
      benchmark_kernel<<<grid_size, thread_block_size, shared_memory_size>>> (
              iterations,
              device_clk_begin,
              device_clk_end,
              device_iterations,
              device_sm_ids,
              action);

      return get_min_begin_max_end (grid_size * thread_block_size);
    }

  public:
    explicit controller (int block_size = 1024, int grid_size = 1, int gpu_id_arg = 0);

    ~controller ();

    [[nodiscard]] int get_block_size () const { return default_block_size; }

    template <typename lambda_type>
    temporal_threads_intervals benchmark (
        std::string &&name, const lambda_type &action, int shared_memory_size=0, int iterations=100)
    {
      const unsigned int latency_grid_size = 1;
      const unsigned int latency_thread_block_size = 1;

      const unsigned int throughput_grid_size = default_grid_size;
      const unsigned int throughput_thread_block_size = default_block_size;

      const auto latency_interval = measure (iterations, action, latency_grid_size, latency_thread_block_size, shared_memory_size);
      const auto throughput_interval = measure (iterations, action, throughput_grid_size, throughput_thread_block_size, shared_memory_size);

      process_measurements (std::move (name), latency_interval, throughput_interval);
      return {
          throughput_grid_size * throughput_thread_block_size,
          host_clk_begin.get (),
          host_clk_end.get (),
          host_sm_ids.get () };
    }
  };
}

#endif // CUDA_BENCHMARK_H

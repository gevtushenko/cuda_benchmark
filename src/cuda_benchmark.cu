//
// Created by egi on 1/3/20.
//

#include "cuda_benchmark.h"

#include "fmt/format.h"
#include "fmt/color.h"
#include "fmt/core.h"
#include "../external/fmt/include/fmt/color.h"
#include "../include/cuda_benchmark.h"

namespace cuda_benchmark
{

warp_intervals::warp_intervals (temporal_threads_intervals thread_intervals)
  : warps_count (thread_intervals.threads_count / warp_size)
{
  intervals.resize (warps_count);
  sm_ids.resize (warps_count);

  const unsigned long long int *thread_clk_begin = thread_intervals.thread_clk_begin;
  const unsigned long long int *thread_clk_end = thread_intervals.thread_clk_end;
  const unsigned int *thread_sm_id = thread_intervals.thread_sm_ids;

  for (unsigned int wid = 0; wid < warps_count; wid++)
    {
      const unsigned long long int min_clk_begin = *std::min_element (
          thread_clk_begin, thread_clk_begin + warp_size);
      const unsigned long long int max_clk_end = *std::max_element (
          thread_clk_end, thread_clk_end + warp_size);

      intervals[wid] = { min_clk_begin, max_clk_end };
      sm_ids[wid] = thread_sm_id[0];

      thread_clk_begin += warp_size;
      thread_clk_end += warp_size;
      thread_sm_id += warp_size;
    }
}

void warp_intervals::store (std::ostream &os)
{
  for (unsigned int wid = 0; wid < warps_count; wid++)
    os << wid << ", "
       << sm_ids[wid] << ", "
       << intervals[wid].first << ", "
       << intervals[wid].second << "\n";
}

/**
 *
 * @param clk Elapsed clocks
 * @param peak_clk in KHz
 * @return time in ms
 */
static float clk_to_t (unsigned long long int clk, int peak_clk)
{
  return (static_cast<float> (clk) / static_cast<float> (peak_clk)) * 1000000.0f;
}

controller::controller (int block_size, int grid_size, int gpu_id_arg)
    : gpu_id (gpu_id_arg)
    , default_block_size (block_size)
    , default_grid_size (grid_size)
{
  cudaSetDevice (gpu_id);

  const unsigned int arrays_size = block_size * grid_size;

  cudaMalloc (&device_clk_begin, arrays_size * sizeof (unsigned long long));
  cudaMalloc (&device_clk_end, arrays_size * sizeof (unsigned long long));
  cudaMalloc (&device_iterations, arrays_size * sizeof (unsigned long long));
  cudaMalloc (&device_sm_ids, arrays_size * sizeof (unsigned int));

  host_clk_begin = std::make_unique<unsigned long long[]> (arrays_size);
  host_clk_end = std::make_unique<unsigned long long[]> (arrays_size);
  host_iterations = std::make_unique<unsigned long long[]> (arrays_size);
  host_sm_ids = std::make_unique<unsigned int[]> (arrays_size);
}

controller::~controller ()
{
  cudaFree (device_clk_begin);
  cudaFree (device_clk_end);
  cudaFree (device_iterations);

  if (results.empty ())
    return;

  cudaDeviceProp prop {};
  cudaGetDeviceProperties (&prop, gpu_id);

  int peak_clk {};
  cudaDeviceGetAttribute (&peak_clk, cudaDevAttrClockRate, gpu_id);

  fmt::print ("Run on ");
  fmt::print (fmt::fg (fmt::color::yellow_green), "{0}\n", prop.name);

  const auto longest_name_size = std::max (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return a.benchmark_name.size () < b.benchmark_name.size ();
  })->benchmark_name.size (), 20ul);
  const auto longest_clock_size = std::max (std::to_string (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return std::to_string (a.latency).size () < std::to_string (b.latency).size ();
  })->latency).size (), std::string("Latency (clk)").size ());
  const auto longest_time_size = std::max (fmt::format ("{:.2f}", clk_to_t (std::max_element (results.begin (), results.end (), [peak_clk] (const result &a, const result &b) {
    return fmt::format ("{:.2f}", clk_to_t (a.latency, peak_clk)).size () < fmt::format ("{:.2f}", clk_to_t (b.latency, peak_clk)).size ();
  })->latency, peak_clk)).size (), std::string ("Latency (ns)").size ());
  const auto longest_throughtput_size = std::max (std::to_string (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return fmt::format ("{:.6f}", a.throughput).size () < fmt::format ("{:.6f}", b.throughput).size ();
  })->throughput).size (), std::string ("Throughput (ops/clk)").size ());

  fmt::print ("{0:<{1}} {2:<{3}}    {4:<{5}}    {6:<{7}}    {8}\n",
    "Benchmark", longest_name_size,
    "Latency (ns)", longest_time_size,
    "Latency (clk)", longest_clock_size,
    "Throughput (ops/clk)", longest_throughtput_size, "Operations");
  for (const auto &result: results)
    {
      fmt::print (fmt::fg (fmt::color::green),  "{0:<{1}} ", result.benchmark_name, longest_name_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0:>{1}.2f}    ", clk_to_t (result.latency, peak_clk), longest_time_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0:>{1}}    ", result.latency, longest_clock_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0:>{1}.6f}    ", result.throughput, longest_throughtput_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0} ({1})\n", result.operations, result.operations * default_block_size);
    }
}

void controller::receive_results (size_t elements) const
{
  cudaMemcpy (host_clk_begin.get (), device_clk_begin, elements * sizeof (unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy (host_clk_end.get (), device_clk_end, elements * sizeof (unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy (host_iterations.get (), device_iterations, elements * sizeof (unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy (host_sm_ids.get (), device_sm_ids, elements * sizeof (unsigned int), cudaMemcpyDeviceToHost);
}

unsigned long long int controller::get_min_latency (size_t elements) const
{
  receive_results (elements);

  unsigned long long int min_latency = std::numeric_limits<unsigned long long int>::max ();

  for (size_t i = 0; i < elements; i += default_block_size)
    {
      const size_t first_element = i;
      const size_t last_element = std::min (first_element + default_block_size, elements);
      const unsigned long long int tb_min_clk_begin = *std::min_element (
          host_clk_begin.get () + first_element, host_clk_begin.get () + last_element);

      const unsigned long long int tb_max_clk_end = *std::max_element (
          host_clk_end.get () + first_element, host_clk_end.get () + last_element);

      min_latency = std::min (min_latency, tb_max_clk_end - tb_min_clk_begin);
    }

  return min_latency;
}

void controller::process_measurements (
    std::string &&name,
    unsigned long long int latency_interval,
    unsigned long long int throughput_interval)
{
  const auto operations = host_iterations[0];

  const auto mean_latency = latency_interval / operations;
  const auto mean_throughput =
      static_cast<float>(operations * default_block_size) / static_cast<float> (throughput_interval);

  results.emplace_back (std::move (name), mean_latency, mean_throughput, operations);
}

}

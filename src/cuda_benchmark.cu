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

/**
 *
 * @param clk Elapsed clocks
 * @param peak_clk in KHz
 * @return time in ms
 */
static float clk_to_t (unsigned long long int clk, int peak_clk)
{
  return (static_cast<double> (clk) / peak_clk) * 1000000.0;
}

controller::~controller ()
{
  cudaFree (gpu_array);

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

}

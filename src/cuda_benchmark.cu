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

controller::~controller ()
{
  cudaFree (gpu_array);

  if (results.empty ())
    return;

  cudaDeviceProp prop {};
  cudaGetDeviceProperties (&prop, gpu_id);
  fmt::print ("Run on {0}\n", prop.name);

  const auto longest_name_size = std::max (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return a.benchmark_name.size () < b.benchmark_name.size ();
  })->benchmark_name.size (), 20ul);
  const auto longest_clock_size = std::max (std::to_string (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return std::to_string (a.latency).size () < std::to_string (b.latency).size ();
  })->latency).size (), 18ul);
  const auto longest_throughtput_size = std::max (std::to_string (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return std::to_string (a.throughput).size () < std::to_string (b.throughput).size ();
  })->throughput).size (), 22ul);

  fmt::print ("{0:<{1}} {2:<{3}} {4:<{5}} {6}\n", "Benchmark", longest_name_size, "Latency (clk)", longest_clock_size, "Throughput (ops/clk)", longest_throughtput_size, "Operations");
  for (const auto &result: results)
    {
      fmt::print (fmt::fg (fmt::color::green),  "{0:<{1}} ", result.benchmark_name, longest_name_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0:<{1}} ", result.latency, longest_clock_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0:<{1}} ", result.throughput, longest_throughtput_size);
      fmt::print (fmt::fg (fmt::color::orange), "{0} ({1})\n", result.operations, result.operations * default_block_size);
    }
}

}

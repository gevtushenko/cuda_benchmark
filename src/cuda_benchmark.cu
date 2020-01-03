//
// Created by egi on 1/3/20.
//

#include "cuda_benchmark.h"

#include "fmt/format.h"
#include "fmt/color.h"
#include "fmt/core.h"
#include "../external/fmt/include/fmt/color.h"

#include <algorithm>

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

  const auto longest_name_size = std::min (std::max_element (results.begin (), results.end (), [] (const result &a, const result &b) {
    return a.benchmark_name.size () < b.benchmark_name.size ();
  })->benchmark_name.size (), 20ul);

  fmt::print ("{0:<{1}}  {2}\n", "Benchmark", longest_name_size, "Clocks");
  for (const auto &result: results)
    {
      fmt::print (fmt::fg (fmt::color::green), "{0:<{1}}  ", result.benchmark_name, longest_name_size);
      fmt::print (fmt::fg (fmt::color::orange), "{2}\n", result.benchmark_name, longest_name_size, result.elapsed);
    }
}

}

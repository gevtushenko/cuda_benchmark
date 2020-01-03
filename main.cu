#include <iostream>
#include "include/cuda_benchmark.h"

#define REPEAT2(x)  x x
#define REPEAT4(x)  REPEAT2(x) REPEAT2(x)
#define REPEAT8(x)  REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)

template <typename data_type>
class add_op
{
public:
  static std::string get_name () { return "add"; }

  __device__ data_type operator() (const data_type &a, const data_type &b) const { return a + b; }
};

template <typename data_type>
class div_op
{
public:
  static std::string get_name () { return "div"; }

  __device__ data_type operator() (const data_type &a, const data_type &b) const { return a / b; }
};

template <typename data_type>
std::string get_type ();

template <> std::string get_type<int> () { return "int"; }
template <> std::string get_type<float> () { return "float"; }
template <> std::string get_type<double> () { return "double"; }

template <typename data_type, typename operation_type>
void operation_benchmark (cuda_benchmark::controller &controller)
{
  data_type *in {};
  cudaMalloc (&in, 2 * sizeof (data_type));

  operation_type op;

  controller.benchmark (get_type<data_type> () + " " + operation_type::get_name (), [=] __device__ (cuda_benchmark::state &state)
  {
    data_type a = in[threadIdx.x];
    data_type b = in[threadIdx.x + 1];

    for (auto _ : state)
      {
        REPEAT32(a = op (a, b););
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in[0] = (a + b);
  });

  cudaFree (in);
}

template <template <typename> typename op_type>
void operation_benchmark (cuda_benchmark::controller &controller)
{
  operation_benchmark<int, op_type<int>> (controller);
  operation_benchmark<float, op_type<float>> (controller);
  operation_benchmark<double, op_type<double>> (controller);
}

int main ()
{
  cuda_benchmark::controller controller (1024, 1);

  operation_benchmark<add_op> (controller);
  operation_benchmark<div_op> (controller);

  return 0;
}

#include "include/cuda_benchmark.h"

#define REPEAT2(x)  x x
#define REPEAT4(x)  REPEAT2(x) REPEAT2(x)
#define REPEAT8(x)  REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT64(x) REPEAT32(x) REPEAT32(x)
#define REPEAT128(x) REPEAT64(x) REPEAT64(x)
#define REPEAT256(x) REPEAT128(x) REPEAT128(x)

template <typename data_type>
class add_op
{
public:
  static std::string get_name () { return "add"; }
  __device__ data_type operator() (const data_type &a, const data_type &b) const { return a + b; }
};

template<>
struct add_op<int>
{
  static std::string get_name () { return "add"; }
  __device__ int operator() (const int& a, const int& b) const { int tmp; asm volatile ("add.s32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct add_op<long long int>
{
  static std::string get_name () { return "add"; }
  __device__ long long int operator()(const long long int& a, const long long int& b) const { long long int tmp; asm volatile ("add.s64 %0, %1, %2;": "=l"(tmp):"l"(a), "l"(b)); return tmp; }
};

template<>
struct add_op<unsigned int>
{
  static std::string get_name () { return "add"; }
  __device__ unsigned int operator()(const unsigned int& a, const unsigned int& b) const { unsigned int tmp; asm volatile ("add.u32 %0, %1, %2;": "=r"(tmp):"r"(a), "r"(b)); return tmp; }
};

template<>
struct add_op<float>
{
  static std::string get_name () { return "add"; }
  __device__ float operator()(const float& a, const float& b) const { float tmp; asm volatile ("add.f32 %0, %1, %2;": "=f"(tmp):"f"(a), "f"(b)); return tmp; }
};

template<>
struct add_op<double>
{
  static std::string get_name () { return "add"; }
  __device__ double operator()(const double& a, const double& b) const { double tmp; asm volatile ("add.f64 %0, %1, %2;": "=d"(tmp):"d"(a), "d"(b)); return tmp; }
};

template <typename data_type>
class div_op
{
public:
  static std::string get_name () { return "div"; }
  __device__ data_type operator() (const data_type &a, const data_type &b) const { return a / b; }
};

template <typename data_type>
class mul_op
{
public:
  static std::string get_name () { return "mul"; }
  __device__ data_type operator() (const data_type &a, const data_type &b) const { return a * b; }
};

template <>
class mul_op<int>
{
public:
  static std::string get_name () { return "mul"; }
  __device__ int operator() (const int &a, const int &b) const { int tmp; asm volatile ("add.s32 %0, %1, %2;" : "=r"(tmp) : "r"(a), "r"(b)); return tmp; }
};

template <typename data_type>
class mad_op
{
public:
  static std::string get_name () { return "mad"; }
  __device__ data_type operator() (const data_type &a, const data_type &b) const { data_type tmp = a; tmp += a * b; return tmp; }
};

template <typename data_type>
class exp_op
{
public:
  static std::string get_name () { return "exp"; }
  __device__ data_type operator() (const data_type &a) const { return std::exp (a); }
};

template <typename data_type>
class fast_exp_op
{
public:
  static std::string get_name () { return "fast exp"; }
  __device__ data_type operator() (const data_type &a) const { return __expf (a); }
};

template <typename data_type>
class sin_op
{
public:
  static std::string get_name () { return "sin"; }
  __device__ data_type operator() (const data_type &a) const { return std::sin (a); }
};

template <typename data_type>
class fast_sin_op
{
public:
  static std::string get_name () { return "fast sin"; }
  __device__ data_type operator() (const data_type &a) const { return __sinf (a); }
};

template <typename data_type>
std::string get_type ();

template <> std::string get_type<int> () { return "int"; }
template <> std::string get_type<float> () { return "float"; }
template <> std::string get_type<double> () { return "double"; }

template <typename data_type, typename operation_type>
void operation_benchmark_1 (cuda_benchmark::controller &controller)
{
  data_type *in {};
  const int block_size = controller.get_block_size ();
  cudaMalloc (&in, block_size * sizeof (data_type));
  cudaMemset (in, block_size * sizeof (data_type), 0);

  operation_type op;

  controller.benchmark (get_type<data_type> () + " " + operation_type::get_name (), [=] __device__ (cuda_benchmark::state &state)
  {
    data_type a = in[threadIdx.x];

    for (auto _ : state)
      {
        REPEAT32(a = op (a););
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in[0] = a;
  });

  cudaFree (in);
}

template <typename data_type, typename operation_type>
void operation_benchmark_2 (cuda_benchmark::controller &controller)
{
  data_type *in {};
  const int block_size = controller.get_block_size ();
  cudaMalloc (&in, (block_size + 1) * sizeof (data_type));
  cudaMemset (in, (block_size + 1) * sizeof (data_type), 0);

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
  operation_benchmark_2<int, op_type<int>> (controller);
  operation_benchmark_2<float, op_type<float>> (controller);
  operation_benchmark_2<double, op_type<double>> (controller);
}

template <template <typename> typename op_type>
void operation_benchmark_float (cuda_benchmark::controller &controller)
{
  operation_benchmark_1<float, op_type<float>> (controller);
  operation_benchmark_1<double, op_type<double>> (controller);
}

struct node
{
public:
  node *next_node;
};

void global_access_benchmark (cuda_benchmark::controller &controller, int n, int stride)
{
  std::unique_ptr<node[]> cpu_in (new node[n]);

  node *in {};
  cudaMalloc (&in, n * sizeof (node));

  for (int i = 0; i < n; i++)
    cpu_in[i].next_node = in + (i + stride) % n;
  cudaMemcpy (in, cpu_in.get (), n * sizeof (node), cudaMemcpyHostToDevice);

  controller.benchmark (
    "global access (stride=" + std::to_string (stride) + "; n=" + std::to_string (n) + ")",
    [=] __device__ (cuda_benchmark::state &state)
  {
    node *a = in + threadIdx.x;

    for (auto _ : state)
      {
        REPEAT32(a = a->next_node;);
      }
    state.set_operations_processed (state.max_iterations () * 32);

    __syncthreads ();
    in[0].next_node = a->next_node;
  });

  cudaFree (in);
}

void divergence_benchmark (cuda_benchmark::controller &controller, int group_size)
{
  int n = 1024;

  int *in {};
  cudaMalloc (&in, (n + 1) * sizeof (int));
  cudaMemset (in, (n + 1) * sizeof (int), 0);

  controller.benchmark ("without divergence (group_size=" + std::to_string (group_size) + ")", [=] __device__ (cuda_benchmark::state &state) {
    int lane_id = threadIdx.x % 32;
    int group_id = lane_id / group_size;

    int a = in[threadIdx.x];
    int b = in[threadIdx.x + 1];

    for (auto _ : state)
      {
        switch (group_id)
          {
            case 0: a += b; break;
            case 1: a -= b; break;
            case 2: a ^= b; break;
            case 3: a &= b; break;
          }
      }

    in[threadIdx.x] = a;
  });

  cudaFree (in);
}

void separated_pipelines_benchmark (cuda_benchmark::controller &controller)
{
  int n = 1024;

  int *in_i {};
  cudaMalloc (&in_i, (n + 1) * sizeof (int));
  cudaMemset (in_i, (n + 1) * sizeof (int), 0);

  float *in_f {};
  cudaMalloc (&in_f, (n + 1) * sizeof (float));
  cudaMemset (in_f, (n + 1) * sizeof (float), 0);

  add_op<int> op_i;
  add_op<float> op_f;

  controller.benchmark ("separated pipelines", [=] __device__ (cuda_benchmark::state &state) {
    int ai = in_i[threadIdx.x];
    int bi = in_i[threadIdx.x + 1];

    float af = in_f[threadIdx.x];
    float bf = in_f[threadIdx.x + 1];

    for (auto _ : state)
      {
        REPEAT32(ai = op_i (ai, bi); af = op_f (af, bf); );
      }
    state.set_operations_processed (state.max_iterations () * 32 * 2);

    in_i[threadIdx.x] = ai;
    in_f[threadIdx.x] = af;
  });

  cudaFree (in_i);
  cudaFree (in_f);
}

int main ()
{
  cuda_benchmark::controller controller;

  operation_benchmark<add_op> (controller);
  operation_benchmark<div_op> (controller);
  operation_benchmark<mul_op> (controller);
  operation_benchmark<mad_op> (controller);

  operation_benchmark_float<exp_op> (controller);
  operation_benchmark_1<float, fast_exp_op<float>> (controller);

  operation_benchmark_float<sin_op> (controller);
  operation_benchmark_1<float, fast_sin_op<float>> (controller);

  global_access_benchmark (controller, 1024, 1);
  global_access_benchmark (controller, 16 * 1024 * 1024, 4);
  global_access_benchmark (controller, 16 * 1024 * 1024, 8);

  divergence_benchmark (controller, 32);
  divergence_benchmark (controller, 16);
  divergence_benchmark (controller, 8);

  separated_pipelines_benchmark (controller);

  return 0;
}

# CUDA Benchmark

A library to benchmark CUDA code, similar to google benchmark. To create a benchmark, define a
device lambda with cuda_benchmark::state& argument. It's possible to prepare measurement
before for loop and postprocess the results to prevent optimization of the results after for loop.

```cuda
void example_benchmark (cuda_benchmark::controller &controller)
{
  float *in {};
  const int block_size = controller.get_block_size ();
  cudaMalloc (&in, block_size * sizeof (float));
  cudaMemset (in, block_size * sizeof (float), 0);

  controller.benchmark ("float __sin ", [=] __device__ (cuda_benchmark::state &state)
  {
    float a = in[threadIdx.x];

    for (auto _ : state)
      {
        REPEAT32(a = __sinf (a););
      }
    state.set_operations_processed (state.max_iterations () * 32);

    in[0] = a;
  });

  cudaFree (in);
}

int main ()
{
  cuda_benchmark::controller controller;
  example_benchmark (controller);
}
```

See an [example](https://github.com/senior-zero/cuda_benchmark/blob/master/example.cu) for more details.

## Installation

As pre-requisites, you'll need git, cmake and nvcc installed.

```bash
$ git clone https://github.com/senior-zero/cuda_benchmark.git
$ cd cuda_benchmark
$ git submodule update --init --recursive
$ mkdir build && cd build
$ cmake ..
$ make
```

## Reporting
When the benchmark binary is executed, each benchmark function is run serially. 
Within each benchmark, there are two kernel calls. The first launch measures 
latency by executing a single thread on GPU. The latency is measured in clock cycles. 
The maximal clock rate of GPU is used to show execution time in nanoseconds. 
The second launch measures throughput by executing multiple threads (1024 by default).
The result is reported in operations per clock cycle.

```
Run on GeForce RTX 2080
Benchmark                            Latency (ns)    Latency (clk)    Throughput (ops/clk)    Operations
int add                                      2.34                4               97.384689    3200 (3276800)
float add                                    2.92                5               62.062958    3200 (3276800)
double add                                  28.65               49                1.683383    3200 (3276800)
int div                                     37.43               64                6.394642    3200 (3276800)
float div                                  155.56              266                2.325893    3200 (3276800)
double div                                 654.39             1119                0.092748    3200 (3276800)
int mul                                      1.75                3               97.791573    3200 (3276800)
float mul                                    2.92                5               62.062958    3200 (3276800)
double mul                                  28.65               49                1.683453    3200 (3276800)
int mad                                      2.92                5               62.157139    3200 (3276800)
float mad                                    2.92                5               62.135921    3200 (3276800)
double mad                                  31.58               54                1.998943    3200 (3276800)
float exp                                   46.20               79                5.277177    3200 (3276800)
double exp                                 495.91              848                0.093855    3200 (3276800)
float fast exp                              25.15               43               15.742342    3200 (3276800)
float sin                                  156.14              267                2.878059    3200 (3276800)
double sin                                 555.56              950                0.102293    3200 (3276800)
float fast sin                              12.87               22               15.928834    3200 (3276800)
global access (stride=1; n=1024)            28.07               48               13.475069    3200 (3276800)
global access (stride=4; n=16777216)       194.74              333                3.053668    3200 (3276800)
global access (stride=8; n=16777216)       258.48              442                2.309383    3200 (3276800)
without divergence (group_size=32)          36.84               63               12.522930    100 (102400)
without divergence (group_size=16)          32.16               55                7.698091    100 (102400)
without divergence (group_size=8)           32.16               55                3.341273    100 (102400)
```
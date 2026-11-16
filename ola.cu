// O kernel do olá mundo: dobra 32 inteiros na GPU (compilado pelo nvcc 12.8, como os .cu do transcritor).
#include <cuda_runtime.h>

__global__ void dobra(int* x) { x[threadIdx.x] *= 2; }

// 0 = ok e *soma = 992 (2 * (0 + 1 + ... + 31)); senão, o código do erro do CUDA
extern "C" int ola_cuda(int* soma) {
  int h[32];
  for (int i = 0; i < 32; i++) h[i] = i;
  int* d = nullptr;
  cudaError_t e = cudaMalloc(&d, sizeof h);
  if (e != cudaSuccess) return e;
  cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
  dobra<<<1, 32>>>(d);
  e = cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  cudaFree(d);
  if (e != cudaSuccess) return e;
  int s = 0;
  for (int i = 0; i < 32; i++) s += h[i];
  *soma = s;
  return 0;
}

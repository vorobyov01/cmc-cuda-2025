#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <omp.h>

#ifdef USE_DOUBLE
using t_float = double;
#define EPS_THRESH 1e-6
#else
using t_float = float;
#define EPS_THRESH 1e-3f
#endif

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

static const int DEFAULT_L     = 900;
static const int DEFAULT_ITERS = 20;

#define IDX(i,j,k) ((size_t)(i)*L*L + (size_t)(j)*L + (k))

void init_field(t_float *A, int L) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < L; k++) {
                size_t idx = IDX(i,j,k);
                if (i==0||i==L-1||j==0||j==L-1||k==0||k==L-1) {
                    A[idx] = (t_float)(
                        10.0 * i/(L-1) +
                        10.0 * j/(L-1) +
                        10.0 * k/(L-1)
                    );
                } else {
                    A[idx] = 0;
                }
            }
}

// CPU Adi w/ OpenMP
void adi_cpu(t_float *A, int L, int iters) {
    t_float *B = (t_float*)malloc((size_t)L*L*L*sizeof(t_float));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 1; it <= iters; ++it) {
        t_float eps = 0;

        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L-1; ++i)
            for (int j = 1; j < L-1; ++j)
                for (int k = 1; k < L-1; ++k) {
                    size_t idx = IDX(i,j,k);
                    B[idx] = 0.5f * (A[idx - (size_t)L*L] + A[idx + (size_t)L*L]);
                }
        std::swap(A,B);

        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L-1; ++i)
            for (int j = 1; j < L-1; ++j)
                for (int k = 1; k < L-1; ++k) {
                    size_t idx = IDX(i,j,k);
                    B[idx] = 0.5f * (A[idx - L] + A[idx + L]);
                }
        std::swap(A,B);

        #pragma omp parallel for collapse(3) reduction(max:eps)
        for (int i = 1; i < L-1; ++i)
            for (int j = 1; j < L-1; ++j)
                for (int k = 1; k < L-1; ++k) {
                    size_t idx = IDX(i,j,k);
                    t_float v = 0.5f * (A[idx - 1] + A[idx + 1]);
                    eps = fmax(eps, fabs(A[idx] - v));
                    B[idx] = v;
                }
        std::swap(A,B);

        printf("CPU IT = %3d EPS = %e\n", it, eps);
        if (eps < EPS_THRESH) break;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();
    printf("ADI CPU Completed. L=%d iters=%d Time=%.3f s\n",
           L, iters, cpu_time);
    free(B);
}

// GPU-kernel
__global__ void sweep_x(const t_float *A, t_float *B, int L) {
    int k = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (k < 1 || k >= L-1 || i < 1 || i >= L-1) return;
    size_t base = (size_t)i*L*L + k;
    for (int j = 1; j < L-1; ++j) {
        size_t idx = base + (size_t)j*L;
        B[idx] = 0.5f * (A[idx - (size_t)L*L] + A[idx + (size_t)L*L]);
    }
}

// GPU-kernel
__global__ void sweep_y(const t_float *A, t_float *B, int L) {
    int k = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (k < 1 || k >= L-1 || i < 1 || i >= L-1) return;
    size_t base = (size_t)i*L*L + k;
    for (int j = 1; j < L-1; ++j) {
        size_t idx = base + (size_t)j*L;
        B[idx] = 0.5f * (A[idx - L] + A[idx + L]);
    }
}

// GPU-kernel
__global__ void sweep_z(const t_float *A, t_float *B, int L) {
    int k = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if (k < 1 || k >= L-1 || i < 1 || i >= L-1) return;
    size_t base = (size_t)i*L*L + k;
    for (int j = 1; j < L-1; ++j) {
        size_t idx = base + (size_t)j*L;
        B[idx] = 0.5f * (A[idx - 1] + A[idx + 1]);
    }
}

void adi_gpu(t_float *h_A, int L, int iters) {
    size_t N = (size_t)L*L*L, bytes = N*sizeof(t_float);
    t_float *d_A=nullptr, *d_B=nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,bytes));
    CUDA_CHECK(cudaMalloc(&d_B,bytes));
    CUDA_CHECK(cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice));

    const int TILE_K = 32, TILE_I = 8;
    dim3 block(TILE_K, TILE_I);
    dim3 grid((L-2 + TILE_K-1)/TILE_K, (L-2 + TILE_I-1)/TILE_I);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < iters; ++it) {
        sweep_x<<<grid,block>>>(d_A, d_B, L);
        CUDA_CHECK(cudaGetLastError());
        sweep_y<<<grid,block>>>(d_B, d_A, L);
        CUDA_CHECK(cudaGetLastError());
        sweep_z<<<grid,block>>>(d_A, d_B, L);
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_A, d_B);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("ADI GPU Completed. L=%d iters=%d Time=%.2f ms\n",
           L, iters, ms);

    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    size_t freeMem=0, totalMem=0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem,&totalMem));
    printf("GPU: %s, Mem Free: %zu MiB / %zu MiB\n",
           prop.name, freeMem>>20, totalMem>>20);

    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
}

int main(int argc, char** argv) {
    const char* mode = (argc>1?argv[1]:"gpu");
    int L     = (argc>2?atoi(argv[2]):DEFAULT_L);
    int iters = (argc>3?atoi(argv[3]):DEFAULT_ITERS);
    size_t N  = (size_t)L*L*L;

    t_float *A0 = (t_float*)malloc(N*sizeof(t_float));
    init_field(A0, L);

    t_float *A_cpu=nullptr, *A_gpu=nullptr;
    if (strcmp(mode,"cpu")==0 || strcmp(mode,"compare")==0) {
        A_cpu = (t_float*)malloc(N*sizeof(t_float));
        memcpy(A_cpu, A0, N*sizeof(t_float));
        printf("Running CPU version...\n");
        adi_cpu(A_cpu, L, iters);
    }
    if (strcmp(mode,"gpu")==0 || strcmp(mode,"compare")==0) {
        A_gpu = (t_float*)malloc(N*sizeof(t_float));
        memcpy(A_gpu, A0, N*sizeof(t_float));
        printf("Running GPU version...\n");
        adi_gpu(A_gpu, L, iters);
    }
    if (strcmp(mode,"compare")==0) {
        t_float maxdiff = 0;
        #pragma omp parallel for reduction(max:maxdiff)
        for (size_t i = 0; i < N; ++i) {
            maxdiff = fmax(maxdiff, fabs(A_cpu[i] - A_gpu[i]));
        }
        printf("Max difference CPU vs GPU = %E\n", maxdiff);
    }

    free(A0);
    if (A_cpu) free(A_cpu);
    if (A_gpu) free(A_gpu);
    return 0;
}

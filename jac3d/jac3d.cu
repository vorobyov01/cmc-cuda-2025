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
#else
using t_float = float;
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

// CPU Jacobi w/ OpenMP
void jacobi_cpu(t_float* A, t_float* B, int L, int iters) {
    for (int it = 1; it <= iters; ++it) {
        t_float maxeps = 0;
        #pragma omp parallel for collapse(3) reduction(max:maxeps)
        for (int i = 1; i < L-1; ++i)
            for (int j = 1; j < L-1; ++j)
                for (int k = 1; k < L-1; ++k) {
                    int idx = i*L*L + j*L + k;
                    t_float diff = fabs(B[idx] - A[idx]);
                    if (diff > maxeps) maxeps = diff;
                    A[idx] = B[idx];
                }
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L-1; ++i)
            for (int j = 1; j < L-1; ++j)
                for (int k = 1; k < L-1; ++k) {
                    int idx = i*L*L + j*L + k;
                    B[idx] = (A[idx - L*L] + A[idx - L] + A[idx - 1]
                            + A[idx + 1]   + A[idx + L] + A[idx + L*L])
                            / (t_float)6;
                }
        printf("IT = %4d EPS = %14.7E\n", it, maxeps);
        if (maxeps < (t_float)0.5) break;
    }
}

// GPU-kernel
__global__ void jacobi_gpu(const t_float* A, t_float* B, int L) {
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int threadInBlock = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    size_t idx = (size_t)blockId * threadsPerBlock + threadInBlock;
    size_t N = (size_t)L * L * L;
    if (idx >= N) return;

    int i = idx / (L * L);
    int rem = idx % (L * L);
    int j = rem / L;
    int k = rem % L;
    if (i > 0 && i < L-1 && j > 0 && j < L-1 && k > 0 && k < L-1) {
        B[idx] = (A[idx - L*L] + A[idx - L] + A[idx - 1]
                + A[idx + 1]   + A[idx + L] + A[idx + L*L])
                / (t_float)6;
    } else {
        B[idx] = A[idx];
    }
}

int main(int argc, char** argv) {
    const char* mode = (argc>1?argv[1]:"gpu");
    int L     = (argc>2?atoi(argv[2]):384);
    int iters = (argc>3?atoi(argv[3]):100);
    size_t N  = (size_t)L*L*L;

    t_float *A0 = (t_float*)malloc(N*sizeof(t_float));
    t_float *B0 = (t_float*)malloc(N*sizeof(t_float));
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                int idx = i*L*L + j*L + k;
                A0[idx] = 0;
                B0[idx] = (i==0||j==0||k==0||i==L-1||j==L-1||k==L-1)
                          ? 0
                          : (t_float)(4 + i + j + k);
            }

    // CPU
    t_float *A_cpu = nullptr, *B_cpu = nullptr;
    if (strcmp(mode,"cpu")==0 || strcmp(mode,"compare")==0) {
        A_cpu = (t_float*)malloc(N*sizeof(t_float));
        B_cpu = (t_float*)malloc(N*sizeof(t_float));
        memcpy(A_cpu,A0,N*sizeof(t_float));
        memcpy(B_cpu,B0,N*sizeof(t_float));
        printf("Running CPU version...\n");
        auto t1 = std::chrono::high_resolution_clock::now();
        jacobi_cpu(A_cpu, B_cpu, L, iters);
        auto t2 = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double>(t2-t1).count();
        printf("Jacobi3D CPU Completed. L=%d iters=%d Time=%.2f s\n", L, iters, cpu_time);
    }

    // GPU
    t_float *gpu_result = nullptr;
    if (strcmp(mode,"gpu")==0 || strcmp(mode,"compare")==0) {
        t_float *A_d = nullptr, *B_d = nullptr;
        CUDA_CHECK( cudaMalloc(&A_d, N*sizeof(t_float)) );
        CUDA_CHECK( cudaMalloc(&B_d, N*sizeof(t_float)) );

        CUDA_CHECK( cudaMemcpy(A_d, B0, N*sizeof(t_float), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(B_d, A0, N*sizeof(t_float), cudaMemcpyHostToDevice) );

        cudaSetDevice(0);
        cudaDeviceProp prop;
        CUDA_CHECK( cudaGetDeviceProperties(&prop,0) );
        printf("GPU: %s, Memory: %.2f MB\n", prop.name, prop.totalGlobalMem/(1024.0*1024.0));

        dim3 block(8,8,8);
        dim3 grid((L+7)/8, (L+7)/8, (L+7)/8);

        cudaEvent_t start, stop;
        CUDA_CHECK( cudaEventCreate(&start) );
        CUDA_CHECK( cudaEventCreate(&stop) );
        CUDA_CHECK( cudaEventRecord(start) );

        for (int it = 1; it <= iters; ++it) {
            jacobi_gpu<<<grid,block>>>(A_d, B_d, L);
            CUDA_CHECK( cudaGetLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
            std::swap(A_d, B_d);
        }

        CUDA_CHECK( cudaEventRecord(stop) );
        CUDA_CHECK( cudaEventSynchronize(stop) );
        float ms;
        CUDA_CHECK( cudaEventElapsedTime(&ms, start, stop) );
        printf("Jacobi3D GPU Completed. L=%d iters=%d Time=%.2f s\n", L, iters, ms/1000.0f);

        CUDA_CHECK( cudaEventDestroy(start) );
        CUDA_CHECK( cudaEventDestroy(stop) );

        gpu_result = (t_float*)malloc(N*sizeof(t_float));
        CUDA_CHECK( cudaMemcpy(gpu_result, A_d, N*sizeof(t_float), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaFree(A_d) );
        CUDA_CHECK( cudaFree(B_d) );
    }

    if (strcmp(mode,"compare")==0) {
        t_float maxdiff = 0;
        #pragma omp parallel for reduction(max:maxdiff)
        for (size_t i = 0; i < N; ++i) {
            t_float diff = fabs(B_cpu[i] - gpu_result[i]);
            if (diff > maxdiff) maxdiff = diff;
        }
        printf("Max difference CPU vs GPU = %E\n", maxdiff);
    }

    free(A0);
    free(B0);
    if (A_cpu) free(A_cpu);
    if (B_cpu) free(B_cpu);
    if (gpu_result) free(gpu_result);
    return 0;
}

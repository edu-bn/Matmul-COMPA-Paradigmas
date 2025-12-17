#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Inicializar matrices
void initMatrix(float* M, int n) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// --------------------------------------------------------
// 1. CPU: Versión CPU multicore
// --------------------------------------------------------
void cpu_matmul(int n, const float* A, const float* B, float* C) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// --------------------------------------------------------
// 2. GPU: Versión GPU básica
// --------------------------------------------------------
__global__ void gpu_matmul_basic(int n, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// --------------------------------------------------------
// 3. GPUsm: Versión con Memoria Compartida (Tiling)
// --------------------------------------------------------
__global__ void gpu_matmul_shared(int n, const float* A, const float* B, float* C) {
    // Memoria compartida para los tiles de A y B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

  
    for (int idx = 0; idx < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++idx) {
        
        // 1. Cargar datos a Memoria Compartida
        int k = idx * BLOCK_SIZE + threadIdx.x; 
        if (row < n && k < n)
            s_A[threadIdx.y][threadIdx.x] = A[row * n + k];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        k = idx * BLOCK_SIZE + threadIdx.y; 
        if (k < n && col < n)
            s_B[threadIdx.y][threadIdx.x] = B[k * n + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Esperar a que todos los hilos del bloque terminen de cargar
        __syncthreads();

        // 2. Calcular producto punto parcial usando memoria compartida
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Esperar a que todos terminen de usar los datos antes de traer el siguiente tile
        __syncthreads();
    }

    // Escribir resultado
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    // Validación de argumentos
    if (argc < 4) {
        std::cout << "Uso: " << argv[0] << " <n> <nt> <ALG>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int nt = std::atoi(argv[2]);
    int alg = std::atoi(argv[3]);

    omp_set_num_threads(nt);

    size_t bytes = n * n * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    initMatrix(h_A, n);
    initMatrix(h_B, n);

    std::cout << "N=" << n << " ALG=" << alg << " Threads=" << nt << std::endl;

    // Solo reservamos GPU si el algoritmo es 2 o 3
    if (alg == 1) {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_matmul(n, h_A, h_B, h_C);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        std::cout << "Tiempo: " << ms.count() << " ms" << std::endl;
    } 
    else {
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        if (alg == 2) {
            gpu_matmul_basic<<<grid, block>>>(n, d_A, d_B, d_C);
        } else if (alg == 3) {
            gpu_matmul_shared<<<grid, block>>>(n, d_A, d_B, d_C);
        }
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        std::cout << "Tiempo: " << milliseconds << " ms" << std::endl;

        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    free(h_A); free(h_B); free(h_C);
    return 0;
}
/*
	*this file exercise matrix multiplication with shared memory and use 
	*the thought of dividing matrix to sub_matrix 
	*/

#include<time.h>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<cuda_profiler_api.h>

#define BLOCK_SIZE 8
#define MATRIX_SIZE 64

typedef struct {
	int width;
	int height;
	float *vals;
} Matrix;

float& GetElement(const Matrix A, int row, int col) {
	return A.vals[row * A.width + col];
}

__device__ float& GetElementKernel(const Matrix A, int row, int col) {
	return A.vals[row * A.width + col];
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
	//__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	//__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
	int range = A.width / BLOCK_SIZE;
	float c_value = 0.0f;
	for (int k = 0; k < range; ++k) {
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[threadIdx.x][threadIdx.y] = GetElementKernel(A, blockIdx.x * BLOCK_SIZE + threadIdx.x, k * BLOCK_SIZE + threadIdx.y);
		Bs[threadIdx.x][threadIdx.y] = GetElementKernel(B, k * BLOCK_SIZE + threadIdx.x, blockIdx.y * BLOCK_SIZE + threadIdx.y);
		
		__syncthreads();

		float tmp = 0.0f;
		for (int block_k = 0; block_k < BLOCK_SIZE; ++block_k) {
			tmp += As[threadIdx.x][block_k] * Bs[block_k][threadIdx.y];
		}
		
		c_value += tmp;
		__syncthreads();
	}

	GetElementKernel(C, blockIdx.x * BLOCK_SIZE + threadIdx.x, blockIdx.y * BLOCK_SIZE + threadIdx.y) = c_value;
}

void MatMulUsual(const Matrix A, const Matrix B, Matrix C) {
	for (int i = 0; i < C.height; ++i) {
		for (int j = 0; j < C.width; ++j) {
			float res = 0.0f;
			for (int k = 0; k < A.width; ++k) {
				res += GetElement(A, i, k) * GetElement(B, k, j);
			}
			GetElement(C, i, j) = res;
		}
	}
}

void checkCUDAError(const char *msg);

int main() {
	size_t memSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
	
	//initialize two matrix
	srand(time(NULL));
	float *valsA = (float*)malloc(memSize);
	float *valsB = (float*)malloc(memSize);
	for (int i = 1; i <= MATRIX_SIZE; ++i) {
		for (int j = 1; j <= MATRIX_SIZE; ++j) {
			valsA[(i - 1) * MATRIX_SIZE + (j - 1)] = (float)(rand()%100);
			valsB[(i - 1) * MATRIX_SIZE + (j - 1)] = (float)(rand()%100);
		}
	}
	Matrix matrixA = {MATRIX_SIZE, MATRIX_SIZE, valsA};
	Matrix matrixB = {MATRIX_SIZE, MATRIX_SIZE, valsB};
	
	//multiplicate with CPU
	float *valsC_CPU = (float*)malloc(memSize);
	Matrix matrixC_CPU = {MATRIX_SIZE, MATRIX_SIZE, valsC_CPU};
	MatMulUsual(matrixA, matrixB, matrixC_CPU);
	
	//multiplicate withGPU
	float *valsC_GPU = (float*)malloc(memSize);
	Matrix matrixC_GPU = {MATRIX_SIZE, MATRIX_SIZE, valsC_GPU};

	//no use
	//	int numBlocks = 8 * 8;
	//int numThreadsPerBlock = MATRIX_SIZE * MATRIX_SIZE / numBlocks;
	
	float *valsA_d, *valsB_d, *valsC_d;
	cudaMalloc(&valsA_d, memSize);
	cudaMemcpy(valsA_d, valsA, memSize, cudaMemcpyHostToDevice);
	cudaMalloc(&valsB_d, memSize);
	cudaMemcpy(valsB_d, valsB, memSize, cudaMemcpyHostToDevice);
	cudaMalloc(&valsC_d, memSize);

	Matrix A_d = {MATRIX_SIZE, MATRIX_SIZE, valsA_d};
	Matrix B_d = {MATRIX_SIZE, MATRIX_SIZE, valsB_d};
	Matrix C_d = {MATRIX_SIZE, MATRIX_SIZE, valsC_d};
	
	//launch kernel
	dim3 dimGrid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	MatMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
	
	//block until the device has completed
	cudaThreadSynchronize();
	
	//check errors
	checkCUDAError("kernel invocation");
	
	//data fetch
	cudaMemcpy(valsC_GPU, valsC_d, memSize, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy");
	
	//verify the data
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			assert(GetElement(matrixC_CPU, i, j) == GetElement(matrixC_GPU, i, j));
		}
	}

	cudaFree(valsA_d);
	cudaFree(valsB_d);
	cudaFree(valsC_d);

	free(valsA);
	free(valsB);
	free(valsC_CPU);
	free(valsC_GPU);

	printf("Correct!\n");

	cudaProfilerStop();	
	return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}


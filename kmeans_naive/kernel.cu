
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>

static constexpr int N = 1 << 16;
static constexpr int D = 512;
static constexpr int K = 128;
static constexpr int data_per_block = 16;
static constexpr int num_threads = 512;


__global__ void kmeans(float* data, float* centroids, float* result)
{
	const int current_block_x = blockIdx.x;
	const int current_block_y = blockIdx.y;
	const int num_blocks_x = gridDim.x;
	const int num_blocks_y = gridDim.y;

	const int current_thread = threadIdx.x;


	const int num_data_per_block = N / (num_blocks_x);
	const int num_centroids_per_block = 1;

	//todo : copy centroid before then do map?
	//copy work data to shared memory
	__shared__ float data_shared[data_per_block * D];

	for (int i = 0; i < D * data_per_block / num_threads; i++)
	{
		float delta = data[current_block_x * num_data_per_block * D + current_thread + i * num_threads] - centroids[current_block_y * D + (current_thread + i * num_threads) % D];
		data_shared[current_thread + i * num_threads] = delta * delta;
	}

	__syncthreads();

	int stride = D / 2;
	while (stride > 32)
	{
		for (int i = 0; i < data_per_block; i++)
		{
			int num_runs = stride > num_threads ? stride / num_threads : 1;
			for (int run = 0; run < num_runs; run++)
			{
				int idx = current_thread + run * num_threads;
				if (idx < stride)
				{
					data_shared[idx + i * D] += data_shared[idx + stride + i * D];
				}
			}
		}
		stride /= 2;
		__syncthreads();
	}
	if (current_thread < 32)
	{
		for (int i = 0; i < data_per_block; i++)
		{
			data_shared[current_thread + i * D] += data_shared[current_thread + 32 + i * D];
			data_shared[current_thread + i * D] += data_shared[current_thread + 16 + i * D];
			data_shared[current_thread + i * D] += data_shared[current_thread + 8 + i * D];
			data_shared[current_thread + i * D] += data_shared[current_thread + 4 + i * D];
			data_shared[current_thread + i * D] += data_shared[current_thread + 2 + i * D];
			data_shared[current_thread + i * D] += data_shared[current_thread + 1 + i * D];
		}
	}

	//copy results to global memory
	if (current_thread < num_data_per_block)
	{
		result[current_block_y * N + current_block_x * num_data_per_block + current_thread] = data_shared[current_thread * D];
	}
}
int main()
{
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	std::vector<float> data(N * D);
	std::vector<float> centroids(K * D);

	std::cout << "Generating data.\n";

	for (int i = 0; i < N; i++)
	{
		for (int j = 0;j < D;j++)
		{
			data[i * D + j] = 0.49f * distribution(generator);
		}
	}
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < D; j++)
		{
			centroids[i * D + j] = 0.49f * distribution(generator);
		}
	}

	std::cout << "Finished generating data.\n";

	std::cout << "Evaluating on CPU.\n";
	
	auto now = std::chrono::high_resolution_clock::now();

	std::vector<float> cpu_result(N * K);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < D; k++)
			{
				float delta = data[i * D + k] - centroids[j * D + k];
				sum += delta * delta;
			}
			cpu_result[i * K + j] = sum;
		}
	}
	auto duration = std::chrono::high_resolution_clock::now() - now;
	std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms\n";

	std::cout << "Finished evaluating on CPU.\n";
	std::cout << "Sending data to device.\n";

	float* data_d;
	float* centroids_d;
	float* result_d;
	cudaMalloc(&data_d, N * D * sizeof(float));
	cudaMalloc(&centroids_d, K * D * sizeof(float));
	cudaMalloc(&result_d, N * K * sizeof(float));
	cudaMemcpy(data_d, data.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(centroids_d, centroids.data(), K * D * sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid(N / data_per_block, K);
	dim3 block(num_threads, 1);

	std::cout << "Data sent. Starting computation.\n";
	now = std::chrono::high_resolution_clock::now();

	kmeans << <grid, block >> > (data_d, centroids_d, result_d);

	std::vector<float> result(N * K);
	cudaMemcpy(result.data(), result_d, N * K * sizeof(float), cudaMemcpyDeviceToHost);
	
	duration = std::chrono::high_resolution_clock::now() - now;
	std::cout << "Data received. Finished computation.\n";
	std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms\n";
	float max_error = 0.0f;
	//print result in matrix form
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K;j++)
		{
			max_error = std::max(max_error, std::abs(result[j * N + i] - cpu_result[i * K + j]));
		}
	}
	std::cout << "Max error: " << max_error << "\n";



}
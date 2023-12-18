
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>

static constexpr int N = 100001;
static constexpr int D = 305;
static constexpr int K = 300;

template<int tile_size_x, int tile_size_d, int tile_size_c, int x_per_thread, int c_per_thread>
__global__ void kmeans_try_seven(float* data, float* centroids, float* result)
{
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int Tx = blockDim.x;
	const int Ty = blockDim.y;
	

	int block_start_x = bx * tile_size_x;
	int block_start_c = by * tile_size_c;


	__shared__ float data_shared[tile_size_d][tile_size_x];
	__shared__ float centroid_shared[tile_size_c][tile_size_d];
	
	float sums[c_per_thread][x_per_thread] = { 0.0f };
	//float x_cache[x_per_thread] = { 0.0f };
	//float c_cache[c_per_thread] = { 0.0f };


	for (int from = 0; from < D; from += tile_size_d)
	{
		int to = from + tile_size_d;
		
		for (int i = 0;i < tile_size_x * tile_size_d / (Tx * Ty);i++)
		{
			int local_data_idx = (tx + ty * Tx + i * Tx * Ty) / tile_size_d;
			int local_data_offset = (tx + ty * Tx + i * Tx * Ty) % tile_size_d;
			if (from + local_data_offset < D && block_start_x + local_data_idx < N)
				data_shared[local_data_offset][local_data_idx] = data[(block_start_x + local_data_idx) * D + (from + local_data_offset)];
			else
				data_shared[local_data_offset][local_data_idx] = 0.0f;
		}
		for (int i = 0; i < tile_size_c * tile_size_d / (Tx * Ty); i++)
		{
			int local_centroid_idx = (tx + ty * Tx + i * Tx * Ty) / tile_size_d;
			int local_centroid_offset = (tx + ty * Tx + i * Tx * Ty) % tile_size_d;
			if (from + local_centroid_offset < D && block_start_c + local_centroid_idx < K)
				centroid_shared[local_centroid_idx][local_centroid_offset] = centroids[(block_start_c + local_centroid_idx) * D + (from + local_centroid_offset)];
			else
				centroid_shared[local_centroid_idx][local_centroid_offset] = 0.0f;
		}
		
		__syncthreads();
		for (int k = 0; k < tile_size_d; k++)
		{
			
			/*for (int j = 0; j < c_per_thread; j++)
				c_cache[j] = centroid_shared[ty + Ty * j][k];
			*/
			/*for (int i = 0; i < x_per_thread; i++)
				x_cache[i] = data_shared[k][tx + Tx * i];
				*/
			
			for (int j = 0; j < c_per_thread; j++)
			{
				float centroid_value = centroid_shared[ty + Ty * j][k];
				for (int i = 0; i < x_per_thread; i++)
				{
					float ds = data_shared[k][tx + Tx * i] - centroid_value;
					sums[j][i] += ds * ds;
				}
			}
		}
		__syncthreads();
	}


	for (int i = 0; i < x_per_thread; i++)
	{
		for (int j = 0; j < c_per_thread; j++)
		{
			if (block_start_x + i * Tx + tx < N && block_start_c + j * Ty + ty < K)
				result[(block_start_c + j * Ty + ty) + (block_start_x + i * Tx + tx) * K] = sums[j][i];
		}
	}

	
}

__global__ void kmeans_argmin(const float* input, int* output)
{
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int Tx = blockDim.x;

	const int block_start = bx * K;

	__shared__ float to_reduce[(K+1) / 2];
	__shared__ int to_reduce_idx[(K+1) / 2];

	//copy into shared memory and do one reduction
	if (tx + ((K + 1) / 2) < K)
	{
		if (input[block_start + tx] < input[block_start + tx + ((K + 1) / 2)])
		{
			to_reduce[tx] = input[block_start + tx];
			to_reduce_idx[tx] = tx;
		}
		else
		{
			to_reduce[tx] = input[block_start + tx + ((K + 1) / 2)];
			to_reduce_idx[tx] = tx + ((K + 1) / 2);
		}	
	}
	else
	{
		to_reduce[tx] = input[block_start + tx];
		to_reduce_idx[tx] = tx;
	}

	__syncthreads();

	int to_reduce_size = (K + 1) / 2;
	while (to_reduce_size > 1)
	{
		int half_reduce_size = (to_reduce_size + 1) / 2;
		if (tx + half_reduce_size < to_reduce_size)
		{
			if (to_reduce[tx] < to_reduce[tx + half_reduce_size])
			{
				to_reduce[tx] = to_reduce[tx];
				to_reduce_idx[tx] = to_reduce_idx[tx];
			}
			else
			{
				to_reduce[tx] = to_reduce[tx + half_reduce_size];
				to_reduce_idx[tx] = to_reduce_idx[tx + half_reduce_size];
			}
		}
		to_reduce_size = half_reduce_size;
		__syncthreads();
	}

	if (tx == 0)
		output[bx] = to_reduce_idx[0];
}

__global__ void kmeans_centroids_sequential(const float* data, const int* cluster_index, int* locks, int* cluster_count, float* result)
{
	const int tx = threadIdx.x;
	const int bx = blockIdx.x;

	const int my_cluster = cluster_index[bx];
	if (tx == 0)
	{
		while (atomicCAS((int*)locks + my_cluster, 0, 1) != 0);
		cluster_count[my_cluster]++;
	}
	__syncthreads();
	result[my_cluster * D + tx] += (1.0f / cluster_count[my_cluster]) * (data[bx * D + tx] - result[my_cluster * D + tx]);
	__syncthreads();
	if (tx == 0)
	{
		atomicExch((int*)locks + my_cluster, 0);
	}
}


void averaging_test()
{
	float* data_d;
	float* centroids_d;
	int* locks_d;
	int* cluster_counts_d;
	int* indices_d;

	cudaMalloc(&data_d, 10 * D * sizeof(float));
	cudaMalloc(&centroids_d, 3 * D * sizeof(float));
	cudaMalloc(&cluster_counts_d, 3 * sizeof(int));
	cudaMalloc(&locks_d, 3 * sizeof(int));
	cudaMalloc(&indices_d, 10 * sizeof(int));
	
	std::vector<float> data(10 * D);
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < D; j++)
		{
			data[i * D + j] = i*j;
		}
	}
	cudaMemcpy(data_d, data.data(), 10 * D * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(centroids_d, 0, 3 * D * sizeof(float));
	cudaMemset(cluster_counts_d, 0, 3 * sizeof(int));
	cudaMemset(locks_d, 0, 3 * sizeof(int));
	std::vector<int> indices({0,0,1,1,0,1,2,1,0,0});
	cudaMemcpy(indices_d, indices.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);

	kmeans_centroids_sequential <<< 10, D >>> (data_d, indices_d, locks_d, cluster_counts_d, centroids_d);

	std::vector<float> result(3 * D);
	cudaMemcpy(result.data(), centroids_d, 3 * D * sizeof(float), cudaMemcpyDeviceToHost);


	float expected[3] = { 0.0f };
	int bins[3] = { 0 };
	for (int i = 0; i < 10; i++)
	{
		bins[indices[i]]++;
		expected[indices[i]] += (1.0f / bins[indices[i]]) * (i - expected[indices[i]]);
	}
	
	bool pass = true;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < D; j++)
		{
			if (std::abs(result[i * D + j] - j * expected[i]) > j * 0.0001f)
			{
				pass = false;
				std::cout << i << "," << j << " : Expected: " << j * expected[i] << " Got: " << result[i * D + j] << "\n";
				break;
			}
		}
		if (!pass)
			break;
	}

	std::cout << "Result: " << (pass ? "PASS" : "FAIL") << "\n";

	cudaFree(data_d);
	cudaFree(centroids_d);
	cudaFree(cluster_counts_d);
	cudaFree(locks_d);
	cudaFree(indices_d);

	
}


int main()
{
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
	std::uniform_int_distribution<int> random_x(0, N - 1);

	std::vector<float> data(N * D);
	std::vector<float> centroids(K * D);

	std::cout << "Generating data.\n";

	for (int i = 0; i < N; i++)
	{
		data[i * D] = (i / (N / K)) + 0.49f * distribution(generator);
		for (int j = 1;j < D;j++)
		{
			data[i * D + j] = 0.49f * distribution(generator);
		}
	}

	std::cout << "Finished generating data.\n";

	float* data_d;
	float* centroids_d;
	float* result_d;
	int* argmin_d;
	int* centroid_count_d;
	int* locks_d;
	cudaMalloc(&data_d, N * D * sizeof(float));
	cudaMalloc(&centroids_d, K * D * sizeof(float));
	cudaMalloc(&result_d, N * K * sizeof(float));
	cudaMalloc(&argmin_d, N * sizeof(int));
	cudaMalloc(&centroid_count_d, K * sizeof(int));
	cudaMalloc(&locks_d, K * sizeof(int));
	
	cudaMemcpy(data_d, data.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
	for (int i = 0; i < K; i++)
	{
		int idx = random_x(generator);
		cudaMemcpy(centroids_d + i * D, data_d + idx * D, D * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	//hyperparams
	constexpr int tile_size_x = 128;
	constexpr int tile_size_d = 16;
	constexpr int tile_size_c = 128;
	dim3 grid((N + tile_size_x - 1) / tile_size_x, (K + tile_size_c - 1) / tile_size_c);
	dim3 block(tile_size_x / 8, tile_size_c / 8);

	std::cout << "Data sent. Starting computation.\n";
	auto now = std::chrono::high_resolution_clock::now();


	for (int i = 0; i < 1; i++)
	{
		kmeans_try_seven<tile_size_x, tile_size_d, tile_size_c, 8, 8> << <grid, block >> > (data_d, centroids_d, result_d);
		kmeans_argmin << < N, (K + 1) / 2 >> > (result_d, argmin_d);
		for (int i = 0; i < K; i++)
		{
			int idx = random_x(generator);
			cudaMemcpy(centroids_d + i * D, data_d + idx * D, D * sizeof(float), cudaMemcpyDeviceToDevice);
		}
		cudaMemset(locks_d, 0, K * sizeof(int));
		cudaMemset(centroid_count_d, 0, K * sizeof(int));
		kmeans_centroids_sequential << < N, D >> > (data_d, argmin_d, locks_d, centroid_count_d, centroids_d);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(centroids.data(), centroids_d, K * D * sizeof(float), cudaMemcpyDeviceToHost);
	auto duration = std::chrono::high_resolution_clock::now() - now;
	std::cout << "Data received. Finished computation.\n";
	std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms\n";
	
	std::vector<int> centroids_int(K);
	for (int i = 0; i < K; i++)
	{
		centroids_int[i] = (int)centroids[i * D];
	}
	std::sort(centroids_int.begin(), centroids_int.end());
	for (int i = 0; i < K; i++)
	{
		std::cout << centroids_int[i] << "\n";
	}


	cudaFree(data_d);
	cudaFree(centroids_d);
	cudaFree(result_d);
	cudaFree(argmin_d);
	cudaFree(centroid_count_d);
	cudaFree(locks_d);



}
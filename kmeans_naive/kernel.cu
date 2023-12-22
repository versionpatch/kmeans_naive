
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "thrust/scan.h"
#include "thrust/binary_search.h"
#include "thrust/device_ptr.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>
#include <numeric>


template<int tile_size_x, int tile_size_d, int tile_size_c, int x_per_thread, int c_per_thread>
__global__ void kmeans_try_seven(float* data, float* centroids, float* result, const int N,const int D,const int K)
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


template<int maxK>
__global__ void kmeans_argmin(const float* input, int* output, const int K)
{
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int Tx = blockDim.x;

	const int block_start = bx * K;

	__shared__ float to_reduce[(maxK + 1) / 2];
	__shared__ int to_reduce_idx[(maxK +1) / 2];

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

template<int maxK>
__global__ void kmeans_min(const float* input, float* output, const int K)
{
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	const int Tx = blockDim.x;

	const int block_start = bx * K;

	__shared__ float to_reduce[(maxK + 1) / 2];

	//copy into shared memory and do one reduction
	if (tx + ((K + 1) / 2) < K)
		to_reduce[tx] = min(input[block_start + tx], input[block_start + tx + ((K + 1) / 2)]);
	else if (tx < K)
		to_reduce[tx] = input[block_start + tx];

	__syncthreads();

	int to_reduce_size = (K + 1) / 2;
	while (to_reduce_size > 1)
	{
		int half_reduce_size = (to_reduce_size + 1) / 2;
		if (tx + half_reduce_size < to_reduce_size)
		{
			to_reduce[tx] = min(to_reduce[tx], to_reduce[tx + half_reduce_size]);
		}
		to_reduce_size = half_reduce_size;
		__syncthreads();
	}

	if (tx == 0)
		output[bx] = to_reduce[0];
}

__global__ void kmeans_centroids_sequential(const float* data, const int* cluster_index, int* locks, int* cluster_count, float* result, const int D)
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

	int D = 300;

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

	kmeans_centroids_sequential<<< 10, D >>> (data_d, indices_d, locks_d, cluster_counts_d, centroids_d, D);

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

void distance_test()
{
float* data_d;
	float* centroids_d;
	float* result_d;

	constexpr int N = 10;
	constexpr int K = 3;
	constexpr int D = 2;

	cudaMalloc(&data_d, N * D * sizeof(float));
	cudaMalloc(&centroids_d, K * D * sizeof(float));
	cudaMalloc(&result_d, N * K * sizeof(float));

	std::vector<float> data(N * D);
	for (int i = 0; i < N; i++)
	{
		data[i * D] = i;
	}
	cudaMemcpy(data_d, data.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
	std::vector<float> centroids(K * D);
	for (int i = 0; i < K; i++)
	{
		centroids[i * D] = -i;
		centroids[i * D + 1] = i;
	}
	cudaMemcpy(centroids_d, centroids.data(), K * D * sizeof(float), cudaMemcpyHostToDevice);

	constexpr int tile_size_x = 128;
	constexpr int tile_size_d = 16;
	constexpr int tile_size_c = 128;
	dim3 grid((N + tile_size_x - 1) / tile_size_x, (K + tile_size_c - 1) / tile_size_c);
	dim3 block(tile_size_x / 8, tile_size_c / 8);
	kmeans_try_seven<tile_size_x, tile_size_d, tile_size_c, 8, 8> << < grid, block >> > (data_d, centroids_d, result_d, 10, 2, 3);
	cudaDeviceSynchronize();
	std::vector<float> result(10 * 3);
	cudaMemcpy(result.data(), result_d, 10 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	bool pass = true;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			float expected = (i + j) * (i + j) + j * j;
			if (std::abs(result[i * 3 + j] - expected) > 0.0001f)
			{
				pass = false;
				std::cout << i << "," << j << " : Expected: " << expected << " Got: " << result[i * 3 + j] << "\n";
				break;
			}
		}
		if (!pass)
			break;
	}

	std::cout << "Result for distance computation test : " << (pass ? "PASS" : "FAIL") << "\n";

	if (!pass)
	{
		cudaFree(data_d);
		cudaFree(centroids_d);
		cudaFree(result_d);
		return;
	}

	//get min
	float* min_array_d;
	cudaMalloc(&min_array_d, N * sizeof(float));
	kmeans_min<K> << < N, (K + 1) / 2 >> > (result_d, min_array_d, K);
	std::vector<float> min_array(N);
	cudaMemcpy(min_array.data(), min_array_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++)
	{
		float expected = i * i;
		if (std::abs(min_array[i] - expected) > 0.0001f)
		{
			pass = false;
			std::cout << i << " : Expected: " << expected << " Got: " << min_array[i] << "\n";
			break;
		}
	}

	std::cout << "Result for min computation test : " << (pass ? "PASS" : "FAIL") << "\n";

	auto dev_ptr = thrust::device_pointer_cast(min_array_d);
	thrust::inclusive_scan(dev_ptr, dev_ptr + N, dev_ptr);
	std::vector<float> min_array_scan(N);
	cudaMemcpy(min_array_scan.data(), min_array_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	//print array
	for (int i = 0; i < N; i++)
	{
		std::cout << min_array_scan[i] / min_array_scan[N - 1] << ", ";
	}
	std::cout << '\n';


	cudaFree(data_d);
	cudaFree(centroids_d);
	cudaFree(result_d);
	cudaFree(min_array_d);

}

static constexpr int N = 100001;
static constexpr int D = 300;
static constexpr int K = 300;

template<int N, int K, int D>
void kmeans_plus_plus_init(float* data, float* centroids)
{
	std::random_device device;
	std::mt19937 generator(device());
	std::uniform_int_distribution<int> random_x(0, N - 1);
	std::uniform_real_distribution<float> random_p(0.0f, 1.0f);

	float* result_d;
	float* min_array_d;
	float* prob_number;
	int* pulled_index;
	
	cudaMalloc(&result_d, N * K * sizeof(float));
	cudaMalloc(&min_array_d, N * sizeof(float));
	cudaMalloc(&pulled_index, sizeof(int));
	cudaMalloc(&prob_number, sizeof(float));

	cudaMemcpy(result_d, data, N * K * sizeof(float), cudaMemcpyHostToDevice);

	auto min_array_dev_ptr = thrust::device_pointer_cast(min_array_d);
	auto pulled_index_dev_ptr = thrust::device_pointer_cast(pulled_index);
	auto prob_number_dev_ptr = thrust::device_pointer_cast(prob_number);


	//initiate first centroid
	int idx = random_x(generator);
	cudaMemcpy(centroids, data + idx * D, D * sizeof(float), cudaMemcpyHostToDevice);
	
	//hyperparameters (maybe optimize for small K ?)
	constexpr int tile_size_x = 128;
	constexpr int tile_size_d = 16;
	constexpr int tile_size_c = 128;
	dim3 grid((N + tile_size_x - 1) / tile_size_x, (K + tile_size_c - 1) / tile_size_c);
	dim3 block(tile_size_x / 8, tile_size_c / 8);

	for (int num_centroids = 1; num_centroids != K; num_centroids++)
	{
		std::cout << "Centroid " << num_centroids << " out of " << K << "\n";
		kmeans_try_seven<tile_size_x, tile_size_d, tile_size_c, 8, 8> << < grid, block >> > (data, centroids, result_d, N, D, num_centroids);
		kmeans_min<K> << < N, (K + 1) / 2 >> > (result_d, min_array_d, num_centroids);
		thrust::inclusive_scan(min_array_dev_ptr, min_array_dev_ptr + N, min_array_dev_ptr);
		float sum;
		cudaMemcpy(&sum, min_array_d + N - 1, sizeof(float), cudaMemcpyDeviceToHost);
		float p = random_p(generator) * sum;
		cudaMemcpy(prob_number, &p, sizeof(float), cudaMemcpyHostToDevice);
		thrust::lower_bound(min_array_dev_ptr, min_array_dev_ptr + N, prob_number_dev_ptr, prob_number_dev_ptr + 1, pulled_index_dev_ptr);
		cudaMemcpy(&idx, pulled_index, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(centroids + num_centroids * D, data + idx * D, D * sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaFree(result_d);
	cudaFree(min_array_d);
	cudaFree(pulled_index);
	cudaFree(prob_number);
}


int main()
{
	//distance_test();
	std::random_device device;
	std::mt19937 generator(device());
	std::normal_distribution<float> distribution(0.0f, 1.0f);
	std::uniform_int_distribution<int> random_x(0, N - 1);

	std::vector<float> data(N * D);
	std::vector<float> centroids(K * D);

	std::cout << "Generating data.\n";

	//generate a vector of indices from 0 to N-1 and shuffle it
	std::vector<int> indices(N);
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), generator);

	for (int i = 0; i < N; i++)
	{
		data[i * D] = 200 * (indices[i] / (N / K)) + 0.49f * distribution(generator);
		for (int j = 1; j < D; j++)
		{
			data[i * D + j] = 200 * (indices[i] / (N / K)) + 0.49f * distribution(generator);
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
	/*
	for (int i = 0; i < K; i++)
	{
		int idx = random_x(generator);
		cudaMemcpy(centroids_d + i * D, data_d + idx * D, D * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	*/
	

	//hyperparams
	constexpr int tile_size_x = 128;
	constexpr int tile_size_d = 16;
	constexpr int tile_size_c = 128;
	dim3 grid((N + tile_size_x - 1) / tile_size_x, (K + tile_size_c - 1) / tile_size_c);
	dim3 block(tile_size_x / 8, tile_size_c / 8);

	std::cout << "Data sent. Starting computation.\n";
	auto now = std::chrono::high_resolution_clock::now();

	kmeans_plus_plus_init<N, K, D>(data_d, centroids_d);

	for (int i = 0; i < 10; i++)
	{
		kmeans_try_seven<tile_size_x, tile_size_d, tile_size_c, 8, 8> << <grid, block >> > (data_d, centroids_d, result_d, N, D, K);
		kmeans_argmin<K> << < N, (K + 1) / 2 >> > (result_d, argmin_d, K);
		for (int i = 0; i < K; i++)
		{
			int idx = random_x(generator);
			cudaMemcpy(centroids_d + i * D, data_d + idx * D, D * sizeof(float), cudaMemcpyDeviceToDevice);
		}
		cudaMemset(locks_d, 0, K * sizeof(int));
		cudaMemset(centroid_count_d, 0, K * sizeof(int));
		kmeans_centroids_sequential<< < N, D >> > (data_d, argmin_d, locks_d, centroid_count_d, centroids_d, D);
	}
	cudaDeviceSynchronize();
	cudaMemcpy(centroids.data(), centroids_d, K * D * sizeof(float), cudaMemcpyDeviceToHost);
	auto duration = std::chrono::high_resolution_clock::now() - now;
	std::cout << "Data received. Finished computation.\n";
	std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms\n";

	std::vector<float> clusters_first_dim(K);
	for (int i = 0; i < K; i++)
	{
		clusters_first_dim[i] = centroids[i * D];
	}
	std::sort(clusters_first_dim.begin(), clusters_first_dim.end());

	for (int i = 0; i < K; i++)
	{
		std::cout << "Centroid " << i << ": ";
		std::cout << clusters_first_dim[i] << ", ";
		std::cout << "\n";
	}


	
	cudaFree(data_d);
	cudaFree(centroids_d);
	cudaFree(result_d);
	cudaFree(argmin_d);
	cudaFree(centroid_count_d);
	cudaFree(locks_d);



}
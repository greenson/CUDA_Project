#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <cstring>

using namespace std;

#define NUM_THREADS_PER_BLOCK 512
#define NO_OF_CHARS 256
#define NO_OF_CHUNKS 32

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
//void badCharHeuristic(char *, int[]);
void badCharHeuristic(char *, int[]);
void seq_search(char *, unsigned int, char *, unsigned int, int *, int *);
void gpu_search(char *, unsigned int, char *, unsigned int, int *, int *);
void multi_gpu_search(char *, unsigned int, char *, unsigned int, int *, int *);
__global__ void kernal_bmh(char *, unsigned int, char *, unsigned int, int *, int *, unsigned int, unsigned int, int, int);
void check_CUDA_Error(const char *);

/*****************************************************************/

/* Driver program */
int main(int argc, char *argv[])
{

	if (argc < 4)
	{
		fprintf(stderr, "usage: ./bmh target who pattern_mode pattern_path\n");
		fprintf(stderr, "target = path to the target text file\n");
		fprintf(stderr, "who = 0: sequential code on CPU, 1: Single-GPU execution, 2: Multi-GPU execution\n");
		fprintf(stderr, "pattern input mode = 0: read-in by command line, 1: read-in by file\n");
		fprintf(stderr, "pattern_path = (if choose file input) pattern file path\n");
		exit(1);
	}

	// to measure time taken by a specific part of the code
	double time_taken;
	clock_t start, end;
	int type_of_device = 0; // CPU or GPU or Multi GPU
	int type_of_pat = 0;	// command line or file

	char *tar;
	char *pat;

	// read in target text file
	FILE *file = fopen(argv[1], "r");
	if (!file)
	{
		printf("Error opening %s", argv[1]);
		exit(EXIT_FAILURE);
	}
	// get filesize
	fseek(file, 0, SEEK_END);
	int fsize = ftell(file);
	printf("File size: %d\n", fsize);
	fseek(file, 0, SEEK_SET);
	// allocate buffer
	tar = (char *)malloc(fsize);
	// read the file into buffer
	fread(tar, fsize, 1, file);
	// close the file
	fclose(file);

	type_of_pat = atoi(argv[3]);
	if (type_of_pat == 0)
	{
		int cSize = 4; //size of char is 1, but size of ascii 'a' is 4 like int
		const int pat_buffer_size = 40000000;
		pat = (char *)malloc(pat_buffer_size * cSize);

		printf("Please type a pattern/keyword you would like to search:\n");
		cin >> pat;
	}
	else
	{
		FILE *fp = fopen(argv[4], "r");
		if (!fp)
		{
			printf("Error opening %s", argv[4]);
			exit(EXIT_FAILURE);
		}
		// get filesize
		fseek(fp, 0, SEEK_END);
		fsize = ftell(fp);
		printf("Pattern size: %d\n", fsize);
		fseek(fp, 0, SEEK_SET);
		// allocate buffer
		pat = (char *)malloc(fsize);
		// read the file into buffer
		fread(pat, fsize, 1, fp);
		// close the file
		fclose(fp);
	}

	type_of_device = atoi(argv[2]);

	int n = strlen(tar);
	int m = strlen(pat);

	int *badchar;
	int *output;

	badchar = new int[NO_OF_CHARS];
	output = new int[n];

	badCharHeuristic(pat, badchar);
	for (int i = 0; i < n; i++)
	{
		output[i] = -1;
	}

	// CPU version
	if (type_of_device == 0)
	{
		start = clock();
		seq_search(tar, n, pat, m, output, badchar);
		end = clock();
	}
	// Single GPU version
	else if (type_of_device == 1)
	{
		cudaFree(0);
		start = clock();
		gpu_search(tar, n, pat, m, output, badchar);
		end = clock();
	}
	// Multi GPU version
	else if (type_of_device == 2)
	{
		cudaSetDevice(0);
		cudaFree(0);
		cudaSetDevice(1);
		cudaFree(0);
		start = clock();
		multi_gpu_search(tar, n, pat, m, output, badchar);
		end = clock();
	}
	else
	{
		printf("Invalid device type.\n");
		exit(EXIT_FAILURE);
	}

	// Print all matched count and position
	int counter = 0;
	printf("Found at pos: ");
	for (int i = 0; i < n; i++)
	{
		if (output[i] != -1)
		{
			counter++;
			if (counter < 100)
			{
				printf("%d ", output[i]);
			}
		}
	}
	if (counter == 0)
	{
		printf("No occurrence.\n");
	}
	else
	{
		if (counter >= 100)
		{
			printf("\nOver 100 results, further display is omitted.");
		}
		printf("\nTotal occurrence: %d\n", counter);
	}

	time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken is %lf\n", time_taken);

	free(tar);
	free(pat);
	delete badchar;
	delete output;
	return 0;
}

// The preprocessing function for Boyer Moore's
// bad character heuristic
void badCharHeuristic(char *pattern, int badchar[NO_OF_CHARS])
{
	int size = strlen(pattern);

	// Initialize badchar array
	for (int i = 0; i < NO_OF_CHARS; i++)
		badchar[i] = size;

	// Fill the shift of last occurrence
	// of a character
	for (int i = 0; i < size - 1; i++)
		badchar[(int)pattern[i]] = size - i - 1;
}

/* A pattern searching function that uses Bad 
   Character Heuristic of Boyer Moore Algorithm */
void seq_search(char *tar, unsigned int n, char *pat, unsigned int m, int *output, int *badchar)
{
	int i, k;
	i = m - 1;

	while (i < n)
	{
		k = 0;
		while ((k < m) && (pat[m - 1 - k] == tar[i - k]))
			k++;
		if (k == m)
		{
			output[i - k + 1] = i - k + 1;
			i++;
		}
		else
			i += badchar[tar[i]];
	}
}

void gpu_search(char *tar, unsigned int n, char *pat, unsigned int m, int *output, int *badchar)
{
	char *d_tar;
	char *d_pat;
	int *d_badchar;
	int *d_output;

	int chunk_size = NO_OF_CHUNKS;

	int num_chunks = (n - 1) / chunk_size + 1;
	int numBlocks = 0;
	for (int i = 0; i < num_chunks; i += NUM_THREADS_PER_BLOCK)
	{
		numBlocks++;
	}

	//numBlocks = (n / m + NUM_THREADS_PER_BLOCK) / NUM_THREADS_PER_BLOCK;

	printf("----Start copying data to GPU----\n");

	//cudaSetDevice(1);
	//check_CUDA_Error("Set Device 1 as current");

	cudaMalloc((void **)&d_tar, n * sizeof(char));
	cudaMalloc((void **)&d_pat, m * sizeof(char));
	cudaMalloc((void **)&d_badchar, NO_OF_CHARS * sizeof(int));
	cudaMalloc((void **)&d_output, n * sizeof(int));
	check_CUDA_Error("memory allocation on device");

	cudaMemcpy(d_tar, tar, n * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pat, pat, m * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_badchar, badchar, NO_OF_CHARS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, n * sizeof(int), cudaMemcpyHostToDevice);
	check_CUDA_Error("memory copy to device");

	printf("----Launch kernal----\n");
	//kernal_bmh<<<numBlocks, NUM_THREADS_PER_BLOCK, NO_OF_CHARS * sizeof(int) + m * sizeof(char)>>>(d_tar, n, d_pat, m, d_badchar, d_output, chunk_size, num_chunks, NO_OF_CHARS, NUM_THREADS_PER_BLOCK);
	kernal_bmh<<<numBlocks, NUM_THREADS_PER_BLOCK>>>(d_tar, n, d_pat, m, d_badchar, d_output, chunk_size, num_chunks, NO_OF_CHARS, NUM_THREADS_PER_BLOCK);
	check_CUDA_Error("Launch kernal");

	cudaDeviceSynchronize();
	check_CUDA_Error("DeviceSynchronize");

	cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
	check_CUDA_Error("memory copy to host");

	printf("----GPU search completed----\n");

	cudaFree(d_tar);
	cudaFree(d_pat);
	cudaFree(d_output);
}

void multi_gpu_search(char *tar, unsigned int n, char *pat, unsigned int m, int *output, int *badchar)
{
	int slice_len = n / 2 + m - 1;
	float slice_overlap = (float)slice_len / (float)n;
	if (slice_overlap > 0.8)
	{
		printf("----Multi GPU utilization low, switching to Single version----\n");
		gpu_search(tar, n, pat, m, output, badchar);
		return;
	}
	if (slice_overlap < 0.5)
	{
		slice_len++;
	}

	int offset = n - slice_len;

	char *d_tar_first, *d_tar_last;
	char *d_pat_first, *d_pat_last;
	int *d_badchar_first, *d_badchar_last;
	int *d_output_first, *d_output_last;

	int chunk_size = NO_OF_CHUNKS;

	int num_chunks = (slice_len - 1) / chunk_size + 1;
	int numBlocks = 0;
	for (int i = 0; i < num_chunks; i += NUM_THREADS_PER_BLOCK)
	{
		numBlocks++;
	}

	printf("----Start Multi GPU version----\n");

	// Set Device 0 as current
	cudaSetDevice(0);
	check_CUDA_Error("Set Device 0 as current");

	// allocate memory on device 0
	cudaMalloc((void **)&d_tar_first, slice_len * sizeof(char));
	cudaMalloc((void **)&d_pat_first, m * sizeof(char));
	cudaMalloc((void **)&d_badchar_first, NO_OF_CHARS * sizeof(int));
	cudaMalloc((void **)&d_output_first, slice_len * sizeof(int));
	check_CUDA_Error("memory allocation on device 0");

	// transfer data to device 0 memory
	cudaMemcpy(d_tar_first, tar, slice_len * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pat_first, pat, m * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_badchar_first, badchar, NO_OF_CHARS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_first, output, slice_len * sizeof(int), cudaMemcpyHostToDevice);
	check_CUDA_Error("memory copy to device 0");

	// Set Device 1 as current
	cudaSetDevice(1);
	check_CUDA_Error("Set Device 1 as current");

	// allocate memory on device 1
	cudaMalloc((void **)&d_tar_last, slice_len * sizeof(char));
	cudaMalloc((void **)&d_pat_last, m * sizeof(char));
	cudaMalloc((void **)&d_badchar_last, NO_OF_CHARS * sizeof(int));
	cudaMalloc((void **)&d_output_last, slice_len * sizeof(int));
	check_CUDA_Error("memory allocation on device 1");

	// transfer data to device 1 memory
	cudaMemcpy(d_tar_last, &tar[offset], slice_len * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pat_last, pat, m * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_badchar_last, badchar, NO_OF_CHARS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_last, &output[offset], slice_len * sizeof(int), cudaMemcpyHostToDevice);
	check_CUDA_Error("memory copy to device 1");

	// run kernal
	cudaSetDevice(0);
	kernal_bmh<<<numBlocks, NUM_THREADS_PER_BLOCK>>>(d_tar_first, slice_len, d_pat_first, m, d_badchar_first, d_output_first, chunk_size, num_chunks, NO_OF_CHARS, NUM_THREADS_PER_BLOCK);
	check_CUDA_Error("Launch kernal on device 0");

	cudaSetDevice(1);
	kernal_bmh<<<numBlocks, NUM_THREADS_PER_BLOCK>>>(d_tar_last, slice_len, d_pat_last, m, d_badchar_last, d_output_last, chunk_size, num_chunks, NO_OF_CHARS, NUM_THREADS_PER_BLOCK);
	check_CUDA_Error("Launch kernal on device 1");

	// wait for GPU to finish before accessing on host
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	check_CUDA_Error("DeviceSynchronize");
	cudaMemcpy(output, d_output_first, slice_len * sizeof(int), cudaMemcpyDeviceToHost);
	check_CUDA_Error("device 0 memory copy to host");

	cudaSetDevice(1);
	cudaDeviceSynchronize();
	check_CUDA_Error("DeviceSynchronize");
	cudaMemcpy(&output[offset], d_output_last, slice_len * sizeof(int), cudaMemcpyDeviceToHost);
	check_CUDA_Error("device 1 memory copy to host");

	printf("----Multi GPU search completed----\n");

	cudaSetDevice(0);
	cudaFree(d_tar_first);
	cudaFree(d_pat_first);
	cudaFree(d_output_first);

	cudaSetDevice(1);
	cudaFree(d_tar_last);
	cudaFree(d_pat_last);
	cudaFree(d_output_last);
}

__global__ void kernal_bmh(char *text, unsigned int text_len, char *pattern, unsigned int pat_len, int *badchar, int *res, unsigned int chunk_size, unsigned int num_chunks, int bt_size, int num_threads)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	/*// store badchar table and pattern in shared-memory
	extern __shared__ int s[];
	int *s_badchar = s;                        		// bt_size ints from bad char table
	char *s_pattern = (char*)&s_badchar[bt_size]; 	// pat_len chars

	if (threadIdx.x < bt_size){
		s_badchar[threadIdx.x] = badchar[threadIdx.x];
	}
	for(int i = threadIdx.x; i < pat_len; i += blockDim.x){
		s_pattern[i] = pattern[i];
	}
	__syncthreads();*/

	// discard un-used thread
	if (tid > num_chunks)
	{
		return;
	}

	// to avoid missing a pattern split over two parts of text
	int chunk_range = (chunk_size * tid) + chunk_size + pat_len - 1;

	// move cursor to the right-most char in current chunk
	int i = (tid * chunk_size) + pat_len - 1;
	int k = 0;
	while (i < chunk_range)
	{
		// reset counter for matched characters
		k = 0;
		if (i >= text_len)
		{
			// break out if i tries to step past text length
			return;
		}

		while (k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k])
		{
			// increment counter for matched characters
			k++;
		}
		if (k == pat_len)
		{
			// pattern matched and set positions in output array
			res[i - k + 1] = i - k + 1;
			i++;
		}
		else
		{
			// shift cursor based on badCharHeuristic
			i += badchar[text[i]];
		}
	}
}

void check_CUDA_Error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

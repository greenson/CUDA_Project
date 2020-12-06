#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define max_target_size 40000000
#define max_pattern_size 40000000
#define n_thread 256

#define NUM_THREADS_PER_BLOCK 256
#define SIZE_OF_CHUNK 32

using namespace std;

void buildLPS(char*, int*, int);
void seq_kmp(char*, char*, int*, int*, int, int);
void check_CUDA_Error(const char*);
double single_gpu(char*, char*, int*, int*, int, int, int);

void check_CUDA_Error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void naive(char* target, char* pattern, int* ans, int target_size, int pattern_size){
    int j;
    for (int i = 0; i < target_size - pattern_size + 1; i++){
        j = 0;
        while (j < pattern_size && target[i+j] == pattern[j]){
            j++;
        }
        if (j == pattern_size){
            ans[i] = 1;
        }
    }
}

void buildLPS(char* pattern, int* lps, int len){
    for (int i = 1, j = 0; i < len; i++){
        while (j > 0 && (pattern[i] != pattern[j])){
            j = lps[j - 1];    
        }
        if (pattern[i] == pattern[j]){
            j++;
        }
        lps[i] = j;
    }
}

void seq_kmp(char* target, char* pattern, int* lps, int* ans, int target_size, int pattern_size){
    int i = 0;
    int j = target_size;

    int k = i;

    while (i < j){
        while (k > 0 && (target[i] != pattern[k])){
            k = lps[k - 1];
        }
        if (target[i] == pattern[k]){
            k++;
        }
        if (k == pattern_size){
            ans[i - k + 1] = 1;
            k = lps[k-1];
        }
        i++;
    }
    return;
}

__global__ void kmp_kernel(char* target, char* pattern, int* lps, int* ans, int target_size, int pattern_size, int chunk_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = (idx * chunk_size);
    int j = (idx * chunk_size) + chunk_size + pattern_size;

    if (i >= target_size){
        return;
    }

    if (j > target_size){
        j = target_size;
    }

    int k = 0;

    while (i < j){
        while (k > 0 && (target[i] != pattern[k])){
            k = lps[k - 1];
        }
        if (target[i] == pattern[k]){
            k++;
        }
        if (k == pattern_size){
            ans[i - k + 1] = 1;
            k = lps[k-1];
        }
        i++;
    }

    return;
}

__global__ void kmp_kernel_share(char* target, char* pattern, int* lps, int* ans, int target_size, int pattern_size, int chunk_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = (idx * chunk_size);
    int j = (idx * chunk_size) + chunk_size + pattern_size;

    __shared__ int n_lps[800];

    __syncthreads();

    int pattern_chunk_size = ceil((double) pattern_size / NUM_THREADS_PER_BLOCK);
    int pi = threadIdx.x * pattern_chunk_size;
    int pj = threadIdx.x * pattern_chunk_size + pattern_chunk_size;

    while (pi < pattern_size && pi < pj){
        n_lps[pi] = lps[pi];
        pi++;
    }

    __syncthreads();
    
    if (i >= target_size){
        return;
    }

    if (j > target_size){
        j = target_size;
    }

    int k = 0;

    while (i < j){
        while (k > 0 && (target[i] != pattern[k])){
            k = n_lps[k - 1];
        }
        if (target[i] == pattern[k]){
            k++;
        }
        if (k == pattern_size){
            ans[i - k + 1] = 1;
            k = n_lps[k-1];
        }
        i++;
    }

    return;
}

double single_gpu(char* target, char* pattern, int* lps, int* ans, int target_size, int pattern_size, int shared){
    char* g_target;
    char* g_pattern;
    int* g_lps;
    int* g_ans;
    clock_t start, end;
    double time_taken;

    start = clock();

    cudaMalloc((void**)&g_target, target_size * sizeof(char));
    cudaMalloc((void**)&g_pattern, pattern_size * sizeof(char));
    cudaMalloc((void**)&g_lps, pattern_size * sizeof(int));
    cudaMalloc((void**)&g_ans, target_size * sizeof(int));
    check_CUDA_Error("memory allocation on device");

    cudaMemcpy(g_target, target, target_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pattern, pattern, pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_lps, lps, pattern_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ans, ans, target_size * sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_Error("memory copy to device");

    int num_chunks = (target_size - 1) / SIZE_OF_CHUNK + 1;
    int num_blocks = 0;
    for (int i = 0; i < num_chunks; i += NUM_THREADS_PER_BLOCK){
        num_blocks++;
    }
    
    dim3 numBlock(1, 1, 1);
    dim3 numThread(n_thread, 1, 1);
    
    int chunk_size = ceil((double) target_size / (n_thread));

    //kmp_kernel<<<(target_size / pattern_size + n_thread) / n_thread, n_thread>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size);
    if (shared == 0){
        kmp_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size, SIZE_OF_CHUNK);
    } else {
        kmp_kernel_share<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size, SIZE_OF_CHUNK);
    }
    //kmp_kernel<<<numBlock, numThread>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size, chunk_size);
    
    check_CUDA_Error("Launch kernal");
    
    cudaDeviceSynchronize();
    check_CUDA_Error("DeviceSynchronize");

    cudaMemcpy(ans, g_ans, target_size * sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_Error("memory copy to host");

    end = clock();

    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    cudaFree(g_target);
    cudaFree(g_pattern);
    cudaFree(g_lps);
    cudaFree(g_ans);

    return time_taken;
}


double multi_gpu(char* target, char* pattern, int* lps, int* ans, int target_size, int pattern_size, int shared){
    int slice_len = target_size / 2 + pattern_size - 1;
    float slice_overlap = (float) slice_len / (float) target_size;
    if (slice_overlap > 0.8)
    {
        printf("----Multi GPU utilization low, switching to Single version----\n");
        return single_gpu(target, pattern, lps, ans, target_size, pattern_size, shared);
    }
    if (slice_overlap < 0.5)
    {
        slice_len++;
    }

    int offset = target_size - slice_len;

    char* g_target_first;
    char* g_target_second;
    char* g_pattern_first;
    char* g_pattern_second;
    int* g_lps_first;
    int* g_lps_second;
    int* g_ans_first;
    int* g_ans_second;
    clock_t start, end;
    double time_taken;

    start = clock();

    cudaSetDevice(0);
    check_CUDA_Error("Set Device 0 as current");

    cudaMalloc((void**)&g_target_first, slice_len * sizeof(char));
    cudaMalloc((void**)&g_pattern_first, pattern_size * sizeof(char));
    cudaMalloc((void**)&g_lps_first, pattern_size * sizeof(int));
    cudaMalloc((void**)&g_ans_first, slice_len * sizeof(int));
    check_CUDA_Error("memory allocation on device 0");

    cudaMemcpy(g_target_first, target, slice_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pattern_first, pattern, pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_lps_first, lps, pattern_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ans_first, ans, slice_len * sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_Error("memory copy to device 0");

    cudaSetDevice(1);
    check_CUDA_Error("Set Device 1 as current");

    cudaMalloc((void**)&g_target_second, slice_len * sizeof(char));
    cudaMalloc((void**)&g_pattern_second, pattern_size * sizeof(char));
    cudaMalloc((void**)&g_lps_second, pattern_size * sizeof(int));
    cudaMalloc((void**)&g_ans_second, slice_len * sizeof(int));
    check_CUDA_Error("memory allocation on device 1");

    cudaMemcpy(g_target_second, &target[offset], slice_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pattern_second, pattern, pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_lps_second, lps, pattern_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ans_second, &ans[offset], slice_len * sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_Error("memory copy to device 1");

    int num_chunks = (slice_len - 1) / SIZE_OF_CHUNK + 1;
    int num_blocks = 0;
    for (int i = 0; i < num_chunks; i += NUM_THREADS_PER_BLOCK){
        num_blocks++;
    }
    
    cudaSetDevice(0);
    if (shared == 0){
        kmp_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target_first, g_pattern_first, g_lps_first, g_ans_first, slice_len, pattern_size, SIZE_OF_CHUNK);
    } else {
        kmp_kernel_share<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target_first, g_pattern_first, g_lps_first, g_ans_first, slice_len, pattern_size, SIZE_OF_CHUNK);
    }
    check_CUDA_Error("Launch kernal on device 0");
    
    cudaSetDevice(1);
    if (shared == 0){
        kmp_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target_second, g_pattern_second, g_lps_second, g_ans_second, slice_len, pattern_size, SIZE_OF_CHUNK);
    } else {
        kmp_kernel_share<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target_second, g_pattern_second, g_lps_second, g_ans_second, slice_len, pattern_size, SIZE_OF_CHUNK);
    }
    check_CUDA_Error("Launch kernal on device 1");
    
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    check_CUDA_Error("DeviceSynchronize");
    cudaMemcpy(ans, g_ans_first, slice_len * sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_Error("device 0 memory copy to host");

    cudaSetDevice(1);
    cudaDeviceSynchronize();
    check_CUDA_Error("DeviceSynchronize");
    cudaMemcpy(&ans[offset], g_ans_second, slice_len * sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_Error("device 1 memory copy to host");

    end = clock();

    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;


    cudaSetDevice(0);
    cudaFree(g_target_first);
    cudaFree(g_pattern_first);
    cudaFree(g_lps_first);
    cudaFree(g_ans_first);

    cudaSetDevice(1);
    cudaFree(g_target_second);
    cudaFree(g_pattern_second);
    cudaFree(g_lps_second);
    cudaFree(g_ans_second);

    return time_taken;
}


int main(int argc, char* argv[]){
    /*
    char* target = (char*) malloc(max_target_size * sizeof(char));
    char* pattern = (char*) malloc(max_pattern_size * sizeof(char));

    FILE* fp = fopen("test.txt", "r");
    int line = 0;
    if (fp != NULL){
        while (line < 2){
            if (line == 0){
                fgets(target, max_target_size, fp);
            } else {
                fgets(pattern, max_pattern_size, fp);
            }
            line++;
        }
    }
    fclose(fp);
    */

    char* pattern;
    char* target;
    int target_size;
    int pattern_size;

    if (argc < 5){
        printf("./kmp <shared_memory> <pinned_memory> <pattern_provided> <target_file> <pattern_file>\n");
        exit(1);
    }
    
    FILE *fp = fopen(argv[4], "r");
    if (!fp)
    {
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    int fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (atoi(argv[2]) == 0){
        target = (char *) malloc(fsize);
    } else{
        cudaHostAlloc((void **)&target, fsize, cudaHostAllocDefault);
        check_CUDA_Error("Pinned memory allocation on host - target");
    }
    fread(target, fsize, 1, fp);
    fclose(fp);
    target_size = strlen(target) - 1;

    if (atoi(argv[3]) == 0){
        const int pat_buffer_size = 40000000;
        if (atoi(argv[2]) == 0){
            pattern = (char *)malloc(pat_buffer_size * sizeof(char));
        } else{
            cudaHostAlloc((void **)&pattern, pat_buffer_size, cudaHostAllocDefault);
            check_CUDA_Error("Pinned memory allocation on host - pattern");
        }
        printf("Please type a pattern/keyword you would like to search:\n");
        cin >> pattern;
        pattern_size = strlen(pattern);
    } else {
        fp = fopen(argv[5], "r");
        if (!fp)
        {
            exit(EXIT_FAILURE);
        }
        fseek(fp, 0, SEEK_END);
        fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        if (atoi(argv[2]) == 0){
            pattern = (char *)malloc(fsize * sizeof(char));
        } else{
            cudaHostAlloc((void **)&pattern, fsize, cudaHostAllocDefault);
            check_CUDA_Error("Pinned memory allocation on host - pattern");
        }
        fread(pattern, fsize, 1, fp);
        fclose(fp);
        pattern_size = strlen(pattern) - 1;
    }

    int* lps = (int*) malloc(pattern_size * sizeof(int));
    int* ans = (int*) malloc(target_size * sizeof(int));
    int* seq_ans = (int*) malloc(target_size * sizeof(int));
    int* naive_ans = (int*) malloc(target_size * sizeof(int));

    for (int i = 0; i < target_size; i++){
        ans[i] = 0;
    }

    for (int i = 0; i < target_size; i++){
        seq_ans[i] = 0;
    }

    for (int i = 0; i < target_size; i++){
        naive_ans[i] = 0;
    }

    clock_t start, end;

    int count;
    double time_taken;

    start = clock();

    naive(target, pattern, naive_ans, target_size, pattern_size);

    end = clock();

    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    count = 0;

    for (int i = 0; i < target_size; i++){
        if (naive_ans[i] != 0){
            count++;
        }
    }

    printf("naive cpu found: %d, time taken: %lf\n", count, time_taken);

    for (int i = 0; i < pattern_size; i++){
        lps[i] = 0;
    }

    buildLPS(pattern, lps, pattern_size);

    /*
    for (int i = 0; i < pattern_size; i++){
        printf("%d ", lps[i]);
    }
    printf("\n");
    */

    start = clock();

    seq_kmp(target, pattern, lps, seq_ans, target_size, pattern_size);

    end = clock();

    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    count = 0;

    for (int i = 0; i < target_size; i++){
        if (seq_ans[i] != 0){
            count++;
        }
    }

    printf("kmp cpu found: %d, time taken: %lf\n", count, time_taken);

    for (int i = 0; i < pattern_size; i++){
        lps[i] = 0;
    }

    buildLPS(pattern, lps, pattern_size);

    int shared = atoi(argv[1]);

    time_taken = single_gpu(target, pattern, lps, ans, target_size, pattern_size, shared);

    count = 0;

    for (int i = 0; i < target_size; i++){
        if (ans[i] != 0){
            count++;
        }
    }

    printf("kmp single_gpu found: %d, time taken: %lf\n", count, time_taken);

    for (int i = 0; i < target_size; i++){
        ans[i] = 0;
    }

    time_taken = multi_gpu(target, pattern, lps, ans, target_size, pattern_size, shared);

    count = 0;

    for (int i = 0; i < target_size; i++){
        if (ans[i] != 0){
            count++;
        }

    }

    printf("kmp multi_gpu found: %d, time taken: %lf\n", count, time_taken);

    if (atoi(argv[2]) != 0){
        cudaFreeHost(target);
        cudaFreeHost(pattern);
    } else {
        free(target);
        free(pattern);
    }
    free(lps);
    free(ans);

    return 0;

}

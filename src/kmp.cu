#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define max_target_size 40000000
#define max_pattern_size 40000000
#define n_thread 256

#define NUM_THREADS_PER_BLOCK 512
#define SIZE_OF_CHUNK 32

using namespace std;

void buildLPS(char*, int*, int);
void seq_kmp(char*, char*, int*, int*, int, int);
void check_CUDA_Error(const char*);

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

    /*
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = pattern_size * idx;
    int j = pattern_size * (idx + 2);
    */

    if (i >= target_size){
        return;
    }

    if (j > target_size){
        j = target_size;
    }

    printf("%d, %d, %d, %d\n", idx, i, j, chunk_size);

    int k = 0;

    while (i < j){
        printf("%d, %d\n", i, j);
        while (k > 0 && (target[i] != pattern[k])){
            printf("%d, %d\n", i, k);
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

    
    FILE *fp = fopen(argv[1], "r");
    if (!fp)
    {
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    int fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    target = (char *) malloc(fsize);
    fread(target, fsize, 1, fp);
    fclose(fp);
    target_size = strlen(target) - 1;

    int type_of_pat = atoi(argv[2]);
    if (type_of_pat == 0){
        const int pat_buffer_size = 40000000;
        pattern = (char *)malloc(pat_buffer_size * sizeof(char));
        printf("Please type a pattern/keyword you would like to search:\n");
        cin >> pattern;
        pattern_size = strlen(pattern);
    } else {
        fp = fopen(argv[3], "r");
        if (!fp)
        {
            exit(EXIT_FAILURE);
        }
        fseek(fp, 0, SEEK_END);
        fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        pattern = (char *) malloc(fsize);
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

    char* g_target;
    char* g_pattern;

    int* g_lps;
    int* g_ans;

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

    start = clock();

    cudaMalloc((void**)&g_target, target_size * sizeof(char));
    cudaMalloc((void**)&g_pattern, pattern_size * sizeof(char));
    cudaMalloc((void**)&g_lps, pattern_size * sizeof(int));
    cudaMalloc((void**)&g_ans, target_size * sizeof(int));
    check_CUDA_Error("memory allocation on device");

    printf("memory allocation on device\n");

    cudaMemcpy(g_target, target, target_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pattern, pattern, pattern_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(g_lps, lps, pattern_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ans, ans, target_size * sizeof(int), cudaMemcpyHostToDevice);
    check_CUDA_Error("memory copy to device");

    printf("memory copy to device");

    //int dim = ceil((double) target_size / (2 * pattern_size));
    //dim3 numBlock(1, 1, 1);
    //dim3 numThread(dim, 1, 1);
    //printf("dim: %d, pat: %d, tar: %d\n", dim, target_size, pattern_size);
    //kmp_kernel<<<numBlock, numThread>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size);

    int num_chunks = (target_size - 1) / SIZE_OF_CHUNK + 1;
    int num_blocks = 0;
    for (int i = 0; i < num_chunks; i += NUM_THREADS_PER_BLOCK){
        num_blocks++;
    }

    //kmp_kernel<<<(target_size / pattern_size + n_thread) / n_thread, n_thread>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size);
    kmp_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(g_target, g_pattern, g_lps, g_ans, target_size, pattern_size, SIZE_OF_CHUNK);
    check_CUDA_Error("Launch kernal");
    
    printf("Launch kernal");

    cudaDeviceSynchronize();
    check_CUDA_Error("DeviceSynchronize");

    printf("DeviceSynchronize");

    cudaMemcpy(ans, g_ans, target_size * sizeof(int), cudaMemcpyDeviceToHost);
    check_CUDA_Error("memory copy to host");

    printf("memory copy to host");

    end = clock();

    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

    count = 0;

    for (int i = 0; i < target_size; i++){
        if (ans[i] != 0){
            count++;
        }
    }

    printf("kmp gpu found: %d, time taken: %lf\n", count, time_taken);

    cudaFree(g_target);
    cudaFree(g_pattern);
    cudaFree(g_lps);
    cudaFree(g_ans);

    free(target);
    free(pattern);
    free(lps);
    free(ans);

    return 0;

}

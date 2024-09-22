import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast
import gc
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda")

def load():
    model_name = "codellama/CodeLlama-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half() # currently working at half precision because one 4090 doesn't have sufficient main memory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model

tokenizer, model = load()
tokenizer.pad_token = tokenizer.eos_token 

def gen(prompt, maxim = 2000):
    inputs = tokenizer(prompt, return_tensors = "pt", padding = True, truncation = True).to("cuda")
    inputs = {k: v.long() if v.dtype == torch.int64 else v for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length = maxim, do_sample = True)
    code = tokenizer.decode(outputs[0])
    return code

prompt = """Optimize this code to have a smaller memory footprint:
#include <stdio.h>

// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Fill host arrays A and B
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    // Launch kernel
    add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<N; i++)
    {
        if( fabs(C[i] - 3.0) > tolerance)
        {
            exit(1);
        }
    }

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}"""

print(gen(prompt))

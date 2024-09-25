__global__ void histogram(int *input_data, int *bins, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(size > i){
        int memory = input_data[i];

        atomicAdd(&bins[memory], 1);
    }
}

__global__ void func1(int *var1, int *var2, int var3)
{
    int var4 = blockDim.x * blockIdx.x + threadIdx.x;

    if(var3 > var4){
        int var5 = var1[var4];

        atomicAdd(&var2[var5], 1);
    }
}
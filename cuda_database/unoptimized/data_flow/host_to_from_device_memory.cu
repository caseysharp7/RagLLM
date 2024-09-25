// This is my own code, though it is closely related to the optimized example

for(int i = 0; i < 2; ++i){
    cudaMemcpy(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice);
}
for(int i = 0; i < 2; ++i){
    kernel<<<100, 512>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
}
for(int i = 0; i < 2; ++i){
    cudaMemcpy(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost);
}

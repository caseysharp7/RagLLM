// My justification for this unoptimized version is that a confused user might try to use multiple streams to create a parallel graphlike structure without the understanding 
// that the synchronization damages the ability of the streams to launch kernels in parallel

cudaStream_t stream1, stream2, stream3;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

for (int i = 0; i < num; i++) {
    cuda_function(stream1);
    cuda_function(stream2);
    cuda_function(stream3);

    cudaStreamSynchronize(stream);
}

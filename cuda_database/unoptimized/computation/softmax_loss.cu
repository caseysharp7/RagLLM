__global__ void
softmax_loss_kernel(float *reduced_loss, float *predict, float *target, float *workspace, int num_outputs)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float loss = 0.0;

    for(int j = 0; j < num_outputs; j++)
        loss += target[batch_idx * num_outputs + j] * logf(predict[batch_idx * num_outputs + j]);
    workspace[batch_idx] = -loss;

    float data_loss = 0;
    for(int stride = blockDim.x; stride > 0; stride--){
        if(threadIdx.x == 0){
            data_loss += workspace[stride];
        }
    }

    if (threadIdx.x == 0) {
        reduced_loss[blockIdx.x] = data_loss;
    }
}

// This is not my code, this comes from NVIDIA's "CUDA C++ Programming Guide"
/* 
This code allocates enough memory for a 2 dimensional array with integers for the device. To optimize 
the code, reference this optimized code and allocate memory using cudaMallocPitch to create a 2 
dimensional array on the device.
*/
int width = 64, height = 64; // these two variable declare the dimensions of the 2D array for the device
float* devPtr; // this variable is a pointer that will point to the first value of the allocated 2D array on the device
size_t pitch; // this variable will keep track of the actual length of each row of the array in the memory
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);

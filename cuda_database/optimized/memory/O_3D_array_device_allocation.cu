// This is not my code, this comes from NVIDIA's "CUDA C++ Programming Guide"
/* 
This code allocates enough memory for a 3 dimensional array with integers for the device. To optimize 
the code, reference this optimized code and allocate memory using cudaMalloc3D to create a 3 
dimensional array on the device.
*/
int width = 64, height = 64, depth = 64; // these variables declare the dimensions of the 3D array
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth); // these lines create a cudaExtent structure called 'extent' that uses make_cudaExtent to conveniently create an extent that defines 3 dimensions in memory space, it takes the three dimension variables as input
cudaPitchedPtr devPitchedPtr; 
cudaMalloc3D(&devPitchedPtr, extent);

#include "gradient_gpu.cuh"

__global__
void central_differences_kernel(float *I, float *dx, float *dy, int nx, int ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < ny - 1 && j < nx - 1) {
        int p = i * nx + j;
        dx[p] = 0.5 * (I[p + 1] - I[p - 1]);
        dy[p] = 0.5 * (I[p + nx] - I[p - nx]);
    }

}

void central_differences_cuda(float *I, float *dx, float *dy, int nx, int ny) {
    int imageSize = nx * ny;
    // Allocate device memory
    float *d_I, *d_dx, *d_dy;
    cudaMalloc((void**)&d_I, imageSize * sizeof(float));
    cudaMalloc((void**)&d_dx, imageSize * sizeof(float));
    cudaMalloc((void**)&d_dy, imageSize * sizeof(float));
    // Copy input data to device
    cudaMemcpy(d_I, I, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    // Define block and grid dimensions
    dim3 blockSize(16, 16); // Adjust block size based on your GPU architecture
    dim3 gridSize(nx / blockSize.x, ny / blockSize.y);
    // Launch the kernel
    central_differences_kernel<<<gridSize, blockSize>>>(d_I, d_dx, d_dy, nx, ny);
    // Wait for kernel completion
    cudaDeviceSynchronize();
    // Copy the results back to host
    cudaMemcpy(dx, d_dx, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dy, d_dy, imageSize * sizeof(float), cudaMemcpyDeviceToHost);    
    // Free device memory
    cudaFree(d_I);
    cudaFree(d_dx);
    cudaFree(d_dy);
}

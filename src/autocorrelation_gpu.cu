#include "autocorrelation_gpu.cuh"
#include <stdio.h>
#include <cmath>

texture<float, 1, cudaReadModeElementType> texInput_x;
texture<float, 1, cudaReadModeElementType> texInput_y;

__global__ void compute_autocorrelation_matrix_kernel_texture(float *A, float *B, float *C, int size_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int p = x + y * blockDim.x * gridDim.x; // data 2D -> intuitivement grid 2D

    if (p < size_data)
    {
        float Ix_value = tex1Dfetch(texInput_x, p); // Read from texture memory
        float Iy_value = tex1Dfetch(texInput_y, p);

        A[p] = Ix_value * Ix_value;
        B[p] = Ix_value * Iy_value;
        C[p] = Iy_value * Iy_value;
    }
}

// __global__ void compute_autocorrelation_matrix_kernel(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x ;
//     int y = blockIdx.y * blockDim.y + threadIdx.y ;
//     int p = x + y*blockDim.x*gridDim.x ;
//     A[p] = Ix[p]*Ix[p];
//     B[p] = Ix[p]*Iy[p];
//     C[p] = Iy[p]*Iy[p];
//     if(p == 0)
//         printf("%f, %f ",Ix[0], A[0]);
// }

// diffuclties 2D:
// 1 - deja initialiser A, B, C : normalement c' ok -> pouvoir acceder à A[i][j] directement sans push_back
//  2- alocate in device memory space for std::vector<float *>
// 3 - Trouver i, j en fonction de chaque thread pour compute A[i][j], B[i][j], C[i][j]
// 4 - Choisir blocksize et grid et traiter le cas des threads en +
//  modifier ceil
/*
Ix = vecteur
Ix[0] -> float * Ix0
Ix0[0:...]

Ix[i][j] where i entre 0<nbr_img et j 0<nx*ny
p//(nx*ny) -> i
le reste de cette division -> j

1block -> 1image : i = block.Idx.x*grid.Dim.x+block.Idx.y

mblock -> 1image :
p/(nx*ny)
*/
// avec texture
void compute_autocorrelation_matrix_cuda(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny, int nbr_imgs)
{
    int imageSize = nx * ny * nbr_imgs;
    // Allocate device memory
    float *d_Ix, *d_Iy, *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_Ix, imageSize * sizeof(float));
    cudaMalloc((void **)&d_Iy, imageSize * sizeof(float));
    cudaMalloc((void **)&d_A, imageSize * sizeof(float));
    cudaMalloc((void **)&d_B, imageSize * sizeof(float));
    cudaMalloc((void **)&d_C, imageSize * sizeof(float));

    size_t offset = 0;

    texInput_x.addressMode[0] = cudaAddressModeBorder;
    texInput_x.addressMode[1] = cudaAddressModeBorder;
    texInput_x.filterMode = cudaFilterModePoint;
    texInput_x.normalized = false;
    texInput_y.addressMode[0] = cudaAddressModeBorder;
    texInput_y.addressMode[1] = cudaAddressModeBorder;
    texInput_y.filterMode = cudaFilterModePoint;
    texInput_y.normalized = false;
    // copy in the memory
    cudaMemcpy(d_Ix, Ix, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Iy, Iy, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(&offset, texInput_x, d_Ix, sizeof(float) * imageSize);
    cudaBindTexture(&offset, texInput_y, d_Iy, sizeof(float) * imageSize);
    // Define block and grid dimensions
    int block_size = 10; // essayer 128, 256
    dim3 blockSize(block_size, block_size);                                                     // Adjust block size based on your GPU architecture
    dim3 gridSize((int)(nx * nbr_imgs / blockSize.x), (int)(ny / blockSize.y)); // 3.2 -> 3 // ceil
    // grid -> nx*ny au moins
    // grid -> nx*ny*nbr_imgs
    // Launch the kernel
    compute_autocorrelation_matrix_kernel_texture<<<gridSize, blockSize>>>(d_A, d_B, d_C, imageSize);
    // Wait for kernel completion
    cudaDeviceSynchronize();
    // Copy the results back to host
    cudaMemcpy(A, d_A, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("A:\n");
    // for (int j = 0; j < 100; j++){
    // 	printf("%f",A[j]);
    // }
    // printf("B:\n");
    // for (int j = 0; j < imageSize; j++){
    // 	printf("%f",B[j]);
    // }
    // printf("C:\n");
    // for (int j = 0; j < imageSize; j++){
    // 	printf("%f",C[j]);
    // }
    // Free device memory
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// sans texture
// void compute_autocorrelation_matrix_cuda(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny){
//     int imageSize = nx * ny;
//     // Allocate device memory
//     float *d_Ix, *d_Iy, *d_A, *d_B, *d_C;

//     cudaMalloc((void**)&d_Ix, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_Iy, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_A, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_B, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_C, imageSize * sizeof(float));

//     // copy in the memory
//     cudaMemcpy(d_Ix, Ix, imageSize * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_Iy, Iy, imageSize * sizeof(float), cudaMemcpyHostToDevice);
//     // Define block and grid dimensions
//     dim3 blockSize(10, 10); // Adjust block size based on your GPU architecture
//     dim3 gridSize((int)(nx/blockSize.x), (int)(ny / blockSize.y));
//     // Launch the kernel
//     compute_autocorrelation_matrix_kernel<<<gridSize, blockSize>>>(d_Ix, d_Iy, d_A, d_B, d_C, nx, ny);
//     // Wait for kernel completion
//     cudaDeviceSynchronize();
//     // Copy the results back to host
//     cudaMemcpy(A, d_A, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(B, d_B, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(C, d_C, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
//     // printf("A:\n");
//     // for (int j = 0; j < 100; j++){
// 	// 	printf("%f",A[j]);
// 	// }
//     // printf("B:\n");
//     // for (int j = 0; j < imageSize; j++){
// 	// 	printf("%f",B[j]);
// 	// }
//     // printf("C:\n");
//     // for (int j = 0; j < imageSize; j++){
// 	// 	printf("%f",C[j]);
// 	// }
//     // Free device memory
//     cudaFree(d_Ix);
//     cudaFree(d_Iy);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }

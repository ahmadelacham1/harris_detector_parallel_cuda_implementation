#include <cmath>
#include "parallel_computation.cuh"
#include <stdio.h>

__global__ void convolution_lines_kernel(float *As, float *Bs, float *Cs, int nx, int ny, int nbr_imgs, double *B, int size, float *As_new, float *Bs_new, float *Cs_new)
{

    // thread index
    int p = threadIdx.x + blockIdx.x * blockDim.x; // 0 -> nx*ny (note some threads go beyond the data)
    int img_idx = nx * ny * blockIdx.y;            // 0 -> nx*ny*(nbr_imgs-1)

    if (p < nx * ny)
    {
        int k = p / nx; // k is the line index
        int i = p % nx; // i is the column index

        // Compute convolution of each line independently // sur toutes lignes de As, Bs, Cs
        if (blockIdx.z == 0)
        {
            double sum = B[0] * As[img_idx + k * nx + i];
            // printf("%d ", img_idx + k * ny + i);

            i += size;

            // As
            for (int j = 1; j < size; j++)
            {
                // R[i-j]+R[i+j] R : 0->size -> xdim+size -> xdim+2*size
                if (size <= i - j && i - j < nx + size && size <= i + j && i + j < nx + size) // outside of BC
                {
                    // printf("img_idx : %d, line index : %d, i-j-size : %d, i+j-size : %d, nx : %d\n", img_idx, k, i - j - size, i + j - size, nx);
                    //  printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (As[img_idx + k * nx + (i - j) - size] + As[img_idx + k * nx + (i + j) - size]);
                }
                else if (0 <= i - j && i - j < size && 0 <= i + j && i + j < size) // both of them in the left BC
                {
                    // printf("%d ", img_idx + k * ny - (i - j) + size);
                    sum += B[j] * (As[img_idx + k * nx - (i - j) + size] + As[img_idx + k * nx - (i + j) + size]);
                }
                else if (nx + size <= i - j && i - j < nx + 2 * size && nx + size <= i + j && i + j < nx + 2 * size) // both of them in the right BC
                {
                    // printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (As[img_idx + k * nx - (i - j - size) + 2 * nx - 1] + As[img_idx + k * nx - (i + j - size) + 2 * nx - 1]);
                }
                else if (size <= i - j && i - j < nx + size && nx + size <= i + j && i + j < nx + 2 * size) // i-j outside BC and i+j in the right BC
                {
                    // printf("img_idx : %d, line index : %d, i : %d, j : %d, nx : %d,  k * nx + (i - j) - size : %d, k * nx - (i + j) - 1 : %d \n", img_idx, k, i - size, j, nx, k * nx - (i + j) - 1, k * nx + (i - j) - size);
                    sum += B[j] * (As[img_idx + k * nx + (i - j) - size] + As[img_idx + k * nx - (i + j - size) + 2 * nx - 1]);
                }

                else if (0 <= i - j && i - j < size && size <= i + j && i + j < nx + size) // i-j in the left BC and i+j is outside BC
                {
                    sum += B[j] * (As[img_idx + k * nx - (i - j) + size] + As[img_idx + k * nx + (i + j) - size]);
                }
            }

            As_new[img_idx + k * nx + i - size] = sum;
        }

        else if (blockIdx.z == 1)
        {
            double sum = B[0] * Bs[img_idx + k * nx + i];
            // printf("%d ", img_idx + k * ny + i);

            i += size;

            // Bs
            for (int j = 1; j < size; j++)
            {
                // R[i-j]+R[i+j] R : 0->size -> xdim+size -> xdim+2*size
                if (size <= i - j && i - j < nx + size && size <= i + j && i + j < nx + size) // outside of BC
                {
                    // printf("img_idx : %d, line index : %d, i-j-size : %d, i+j-size : %d, nx : %d\n", img_idx, k, i - j - size, i + j - size, nx);
                    //  printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Bs[img_idx + k * nx + (i - j) - size] + Bs[img_idx + k * nx + (i + j) - size]);
                }
                else if (0 <= i - j && i - j < size && 0 <= i + j && i + j < size) // both of them in the left BC
                {
                    // printf("%d ", img_idx + k * ny - (i - j) + size);
                    sum += B[j] * (Bs[img_idx + k * nx - (i - j) + size] + Bs[img_idx + k * nx - (i + j) + size]);
                }
                else if (nx + size <= i - j && i - j < nx + 2 * size && nx + size <= i + j && i + j < nx + 2 * size) // both of them in the right BC
                {
                    // printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Bs[img_idx + k * nx - (i - j - size) + 2 * nx - 1] + Bs[img_idx + k * nx - (i + j - size) + 2 * nx - 1]);
                }
                else if (size <= i - j && i - j < nx + size && nx + size <= i + j && i + j < nx + 2 * size) // i-j outside BC and i+j in the right BC
                {
                    // printf("img_idx : %d, line index : %d, i : %d, j : %d, nx : %d,  k * nx + (i - j) - size : %d, k * nx - (i + j) - 1 : %d \n", img_idx, k, i - size, j, nx, k * nx - (i + j) - 1, k * nx + (i - j) - size);
                    sum += B[j] * (Bs[img_idx + k * nx + (i - j) - size] + Bs[img_idx + k * nx - (i + j - size) + 2 * nx - 1]);
                }

                else if (0 <= i - j && i - j < size && size <= i + j && i + j < nx + size) // i-j in the left BC and i+j is outside BC
                {
                    sum += B[j] * (Bs[img_idx + k * nx - (i - j) + size] + Bs[img_idx + k * nx + (i + j) - size]);
                }
            }

            Bs_new[img_idx + k * nx + i - size] = sum;
        }

        else if (blockIdx.z == 2)
        {
            double sum = B[0] * Cs[img_idx + k * nx + i];
            // printf("%d ", img_idx + k * ny + i);

            i += size;

            // Cs
            for (int j = 1; j < size; j++)
            {
                // R[i-j]+R[i+j] R : 0->size -> xdim+size -> xdim+2*size
                if (size <= i - j && i - j < nx + size && size <= i + j && i + j < nx + size) // outside of BC
                {
                    // printf("img_idx : %d, line index : %d, i-j-size : %d, i+j-size : %d, nx : %d\n", img_idx, k, i - j - size, i + j - size, nx);
                    //  printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Cs[img_idx + k * nx + (i - j) - size] + Cs[img_idx + k * nx + (i + j) - size]);
                }
                else if (0 <= i - j && i - j < size && 0 <= i + j && i + j < size) // both of them in the left BC
                {
                    // printf("%d ", img_idx + k * ny - (i - j) + size);
                    sum += B[j] * (Cs[img_idx + k * nx - (i - j) + size] + Cs[img_idx + k * nx - (i + j) + size]);
                }
                else if (nx + size <= i - j && i - j < nx + 2 * size && nx + size <= i + j && i + j < nx + 2 * size) // both of them in the right BC
                {
                    // printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Cs[img_idx + k * nx - (i - j - size) + 2 * nx - 1] + Cs[img_idx + k * nx - (i + j - size) + 2 * nx - 1]);
                }
                else if (size <= i - j && i - j < nx + size && nx + size <= i + j && i + j < nx + 2 * size) // i-j outside BC and i+j in the right BC
                {
                    // printf("img_idx : %d, line index : %d, i : %d, j : %d, nx : %d,  k * nx + (i - j) - size : %d, k * nx - (i + j) - 1 : %d \n", img_idx, k, i - size, j, nx, k * nx - (i + j) - 1, k * nx + (i - j) - size);
                    sum += B[j] * (Cs[img_idx + k * nx + (i - j) - size] + Cs[img_idx + k * nx - (i + j - size) + 2 * nx - 1]);
                }

                else if (0 <= i - j && i - j < size && size <= i + j && i + j < nx + size) // i-j in the left BC and i+j is outside BC
                {
                    sum += B[j] * (Cs[img_idx + k * nx - (i - j) + size] + Cs[img_idx + k * nx + (i + j) - size]);
                }
            }

            Cs_new[img_idx + k * nx + i - size] = sum;
        }
    }
}
__global__ void convolution_columns_kernel(float *As, float *Bs, float *Cs, int nx, int ny, int nbr_imgs, double *B, int size, float *As_new, float *Bs_new, float *Cs_new)

{
    // Compute convolution of each column independently // sur toutes colonnes de As, Bs, Cs

    // thread index
    int p = threadIdx.x + blockIdx.x * blockDim.x; // 0 -> nx*ny (note some threads go beyond the data)
    int img_idx = nx * ny * blockIdx.y;

    if (p < nx * ny)
    {
        int i = p / nx; // i is the line index
        int k = p % nx; // k is the column index

        // Compute convolution of each line independently // sur toutes lignes de As, Bs, Cs
        if (blockIdx.z == 0)
        {
            double sum = B[0] * As[img_idx + i * nx + k];
            i += size;
            // As
            for (int j = 1; j < size; j++)
            {
                // R[i-j]+R[i+j] R : 0->size -> xdim+size -> xdim+2*size
                if (size <= i - j && i - j < ny + size && size <= i + j && i + j < ny + size) // outside of BC
                {
                    // printf("img_idx : %d, line index : %d, i-j-size : %d, i+j-size : %d, nx : %d\n", img_idx, k, i - j - size, i + j - size, nx);
                    //  printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (As[img_idx + ((i - j) - size) * nx + k] + As[img_idx + ((i + j) - size) * nx + k]);
                }
                else if (0 <= i - j && i - j < size && 0 <= i + j && i + j < size) // both of them in the left BC
                {
                    // printf("%d ", img_idx + k * ny - (i - j) + size);
                    sum += B[j] * (As[img_idx + (size - (i - j)) * nx + k] + As[img_idx + (size - (i + j)) * nx + k]);
                }
                else if (ny + size <= i - j && i - j < ny + 2 * size && ny + size <= i + j && i + j < ny + 2 * size) // both of them in the right BC
                {
                    // printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (As[img_idx + (2 * ny + size - 1 - (i - j)) * nx + k] + As[img_idx + (2 * ny + size - 1 - (i + j)) * nx + k]);
                }
                else if (size <= i - j && i - j < ny + size && ny + size <= i + j && i + j < ny + 2 * size) // i-j outside BC and i+j in the right BC
                {
                    // printf("img_idx : %d, line index : %d, i : %d, j : %d, nx : %d,  k * nx + (i - j) - size : %d, k * nx - (i + j) - 1 : %d \n", img_idx, k, i - size, j, nx, k * nx - (i + j) - 1, k * nx + (i - j) - size);
                    sum += B[j] * (As[img_idx + ((i - j) - size) * nx + k] + As[img_idx + (2 * ny + size - 1 - (i + j)) * nx + k]);
                }

                else if (0 <= i - j && i - j < size && size <= i + j && i + j < ny + size) // i-j in the left BC and i+j is outside BC
                {
                    sum += B[j] * (As[img_idx + (size - (i - j)) * nx + k] + As[img_idx + ((i + j) - size) * nx + k]);
                }
            }

            As_new[img_idx + k + (i - size) * nx] = sum;
        }

        else if (blockIdx.z == 1)
        {
            double sum = B[0] * Bs[img_idx + i * nx + k];
            i += size;
            // Bs
            for (int j = 1; j < size; j++)
            {
                // R[i-j]+R[i+j] R : 0->size -> xdim+size -> xdim+2*size
                if (size <= i - j && i - j < ny + size && size <= i + j && i + j < ny + size) // outside of BC
                {
                    // printf("img_idx : %d, line index : %d, i-j-size : %d, i+j-size : %d, nx : %d\n", img_idx, k, i - j - size, i + j - size, nx);
                    //  printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Bs[img_idx + ((i - j) - size) * nx + k] + Bs[img_idx + ((i + j) - size) * nx + k]);
                }
                else if (0 <= i - j && i - j < size && 0 <= i + j && i + j < size) // both of them in the left BC
                {
                    // printf("%d ", img_idx + k * ny - (i - j) + size);
                    sum += B[j] * (Bs[img_idx + (size - (i - j)) * nx + k] + Bs[img_idx + (size - (i + j)) * nx + k]);
                }
                else if (ny + size <= i - j && i - j < ny + 2 * size && ny + size <= i + j && i + j < ny + 2 * size) // both of them in the right BC
                {
                    // printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Bs[img_idx + (2 * ny + size - 1 - (i - j)) * nx + k] + Bs[img_idx + (2 * ny + size - 1 - (i + j)) * nx + k]);
                }
                else if (size <= i - j && i - j < ny + size && ny + size <= i + j && i + j < ny + 2 * size) // i-j outside BC and i+j in the right BC
                {
                    // printf("img_idx : %d, line index : %d, i : %d, j : %d, nx : %d,  k * nx + (i - j) - size : %d, k * nx - (i + j) - 1 : %d \n", img_idx, k, i - size, j, nx, k * nx - (i + j) - 1, k * nx + (i - j) - size);
                    sum += B[j] * (Bs[img_idx + ((i - j) - size) * nx + k] + Bs[img_idx + (2 * ny + size - 1 - (i + j)) * nx + k]);
                }

                else if (0 <= i - j && i - j < size && size <= i + j && i + j < ny + size) // i-j in the left BC and i+j is outside BC
                {
                    sum += B[j] * (Bs[img_idx + (size - (i - j)) * nx + k] + Bs[img_idx + ((i + j) - size) * nx + k]);
                }
            }

            Bs_new[img_idx + k + (i - size) * nx] = sum;
        }

        else if (blockIdx.z == 2)
        {
            double sum = B[0] * Cs[img_idx + i * nx + k];
            i += size;
            // Cs
            for (int j = 1; j < size; j++)
            {
                // R[i-j]+R[i+j] R : 0->size -> xdim+size -> xdim+2*size
                if (size <= i - j && i - j < ny + size && size <= i + j && i + j < ny + size) // outside of BC
                {
                    // printf("img_idx : %d, line index : %d, i-j-size : %d, i+j-size : %d, nx : %d\n", img_idx, k, i - j - size, i + j - size, nx);
                    //  printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Cs[img_idx + ((i - j) - size) * nx + k] + Cs[img_idx + ((i + j) - size) * nx + k]);
                }
                else if (0 <= i - j && i - j < size && 0 <= i + j && i + j < size) // both of them in the left BC
                {
                    // printf("%d ", img_idx + k * ny - (i - j) + size);
                    sum += B[j] * (Cs[img_idx + (size - (i - j)) * nx + k] + Cs[img_idx + (size - (i + j)) * nx + k]);
                }
                else if (ny + size <= i - j && i - j < ny + 2 * size && ny + size <= i + j && i + j < ny + 2 * size) // both of them in the right BC
                {
                    // printf("%d ", img_idx + k * ny + i - j - size);
                    sum += B[j] * (Cs[img_idx + (2 * ny + size - 1 - (i - j)) * nx + k] + Cs[img_idx + (2 * ny + size - 1 - (i + j)) * nx + k]);
                }
                else if (size <= i - j && i - j < ny + size && ny + size <= i + j && i + j < ny + 2 * size) // i-j outside BC and i+j in the right BC
                {
                    // printf("img_idx : %d, line index : %d, i : %d, j : %d, nx : %d,  k * nx + (i - j) - size : %d, k * nx - (i + j) - 1 : %d \n", img_idx, k, i - size, j, nx, k * nx - (i + j) - 1, k * nx + (i - j) - size);
                    sum += B[j] * (Cs[img_idx + ((i - j) - size) * nx + k] + Cs[img_idx + (2 * ny + size - 1 - (i + j)) * nx + k]);
                }

                else if (0 <= i - j && i - j < size && size <= i + j && i + j < ny + size) // i-j in the left BC and i+j is outside BC
                {
                    sum += B[j] * (Cs[img_idx + (size - (i - j)) * nx + k] + Cs[img_idx + ((i + j) - size) * nx + k]);
                }
            }

            Cs_new[img_idx + k + (i - size) * nx] = sum;
        }
    }
}

void parallel_comp_cuda(float *As, float *Bs, float *Cs, int nx, int ny, int nbr_imgs, float sigma, int precision)
{
    /*
     - As : concatenation of float array A for all images
     - Bs : concatenation of float array B for all images
     - Cs : concatenation of float array C for all images
     - size = nx*ny : size of A (same for B, C)
    */
    // printf("before conv  As[0:3] : (%f, %f, %f, %f)\n", As[0], As[1], As[2], As[3]);
    int size = nx * ny;
    int bytes = size * nbr_imgs * sizeof(float);
    // Allocate device memory
    float *d_As, *d_Bs, *d_Cs, *d_As_new, *d_Bs_new, *d_Cs_new, *d_As_NN, *d_Bs_NN, *d_Cs_NN;
    cudaMalloc((void **)&d_As, bytes);
    cudaMalloc((void **)&d_Bs, bytes);
    cudaMalloc((void **)&d_Cs, bytes);
    cudaMalloc((void **)&d_As_new, bytes);
    cudaMalloc((void **)&d_Bs_new, bytes);
    cudaMalloc((void **)&d_Cs_new, bytes);
    cudaMalloc((void **)&d_As_NN, bytes);
    cudaMalloc((void **)&d_Bs_NN, bytes);
    cudaMalloc((void **)&d_Cs_NN, bytes);

    // Check for memory allocation errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        printf("CUDA Memory Allocation Error: %s\n", cudaGetErrorString(cudaError));
        // Handle the error or exit the program
        return;
    }
    // Copy array from CPU to GPU
    cudaMemcpy(d_As, As, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bs, Bs, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cs, Cs, bytes, cudaMemcpyHostToDevice);

    // Check for cudaMemcpy errors
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        printf("CUDA Memory Copy Error: %s\n", cudaGetErrorString(cudaError));
        // Handle the error or exit the program
        return;
    }

    int block_size = 64;
    dim3 blockSize(block_size); // Adjust block size based on your GPU architecture
    /*
    GridDim.x = ceil(size/blockSize.x)
    GridDim.y = nbr_imgs
    GridDim.z = 3 (A, B, C)
    */
    dim3 gridSize(ceil(size / blockSize.x), nbr_imgs, 3);

    // Compute convolution weigths -> B
    int xdim = nx;
    double den = 2 * sigma * sigma;
    int size_kernel_conv = (int)(3 * sigma) + 1;

    if (size_kernel_conv > xdim)
        return;

    // compute the coefficients of the 1D convolution kernel
    double *B = new double[size_kernel_conv];
    for (int i = 0; i < size_kernel_conv; i++)
        B[i] = 1 / (sigma * sqrt(2.0 * 3.1415926)) * exp(-i * i / den);

    double norm_ = 0;

    // normalize the 1D convolution kernel
    for (int i = 0; i < size_kernel_conv; i++)
        norm_ += B[i];

    norm_ *= 2;

    norm_ -= B[0];

    for (int i = 0; i < size_kernel_conv; i++)
        B[i] /= norm_;

    double *d_B;
    cudaMalloc((void **)&d_B, size_kernel_conv * sizeof(double));
    cudaMemcpy(d_B, B, size_kernel_conv * sizeof(double), cudaMemcpyHostToDevice);

    // printf("B kernel de conv : %f, %f, %f\n", B[0], B[1], B[2]);
    //  Launch the kernel
    convolution_lines_kernel<<<gridSize, blockSize>>>(d_As, d_Bs, d_Cs, nx, ny, nbr_imgs, d_B, size_kernel_conv, d_As_new, d_Bs_new, d_Cs_new);

    cudaDeviceSynchronize();

    cudaMemcpy(As, d_As_new, bytes, cudaMemcpyDeviceToHost);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error
    }

    cudaFree(d_As);
    cudaFree(d_Bs);
    cudaFree(d_Cs);

    convolution_columns_kernel<<<gridSize, blockSize>>>(d_As_new, d_Bs_new, d_Cs_new, nx, ny, nbr_imgs, d_B, size_kernel_conv, d_As_NN, d_Bs_NN, d_Cs_NN);

    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle error
    }
    // Copy the results back to host
    cudaMemcpy(As, d_As_NN, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Bs, d_Bs_NN, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Cs, d_Cs_NN, bytes, cudaMemcpyDeviceToHost);

    /*
    printf("after conv columns As[0:15000000] : ");
    for (int d = 0; d < 600; d++)
    {
        printf("%f, ", As[d]);
    }
    printf("\n");*/
    // Free
    cudaFree(d_As_NN);
    cudaFree(d_Bs_NN);
    cudaFree(d_Cs_NN);
    cudaFree(d_As_new);
    cudaFree(d_Bs_new);
    cudaFree(d_Cs_new);
    cudaFree(d_B);
}

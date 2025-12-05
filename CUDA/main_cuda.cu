#include <stdio.h>
#include <cuda_runtime.h>
#include "gaussian_blur.h"

// Copy kernel from CPU to GPU constant memory
__constant__ float GAUSSIAN_KERNEL_GPU[KERNEL_SIZE * KERNEL_SIZE];

// CUDA kernel (GPU)
__global__
void gaussianBlurCUDAKernel(unsigned char *input, unsigned char *output,
                            int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int ky = -RADIUS; ky <= RADIUS; ky++) {
        for (int kx = -RADIUS; kx <= RADIUS; kx++) {

            int px = x + kx;
            int py = y + ky;

            if (px >= 0 && px < width && py >= 0 && py < height) {

                int kernelIndex = (ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS);

                sum += input[py * width + px] * GAUSSIAN_KERNEL_GPU[kernelIndex];
            }
        }
    }

    output[y * width + x] = (unsigned char)sum;
}

// CUDA wrapper function
void gaussianBlurCUDA(unsigned char *input, unsigned char *output, int width, int height)
{
    unsigned char *d_input, *d_output;
    size_t size = width * height * sizeof(unsigned char);   // FIXED

    // flatten kernel for constant memory
    float hostKernel[KERNEL_SIZE * KERNEL_SIZE];
    int idx = 0;
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
            hostKernel[idx++] = GAUSSIAN_KERNEL[i][j];

    cudaMemcpyToSymbol(GAUSSIAN_KERNEL_GPU, hostKernel,
                       KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    gaussianBlurCUDAKernel<<<grid, block>>>(d_input, d_output, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }

    int width, height, maxVal;

    unsigned char *input = readPGM(argv[1], &width, &height, &maxVal);
    unsigned char *output = (unsigned char*)malloc(width * height);

    printf("Running Gaussian Blur on GPU...\n");

    gaussianBlurCUDA(input, output, width, height);

    writePGM(argv[2], output, width, height, maxVal);

    printf("Done. Output saved to %s\n", argv[2]);

    free(input);
    free(output);

    return 0;
}

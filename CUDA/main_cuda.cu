#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__constant__ float d_kernel[9] = {
    1.0f/16, 2.0f/16, 1.0f/16,
    2.0f/16, 4.0f/16, 2.0f/16,
    1.0f/16, 2.0f/16, 1.0f/16
};

unsigned char* readPGM(const char* filename, int* width, int* height, int* maxVal)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp) exit(1);

    char magic[3] = {0};
    fscanf(fp, "%2s", magic);

    int c = fgetc(fp);
    while (c == '#') { while (fgetc(fp) != '\n'); c = fgetc(fp); }
    ungetc(c, fp);

    fscanf(fp, "%d %d", width, height);
    fscanf(fp, "%d", maxVal);
    fgetc(fp);

    int size = (*width) * (*height);
    unsigned char* img = (unsigned char*)malloc(size);
    fread(img, 1, size, fp);
    fclose(fp);

    return img;
}

void writePGM(const char* filename, const unsigned char* img,
              int width, int height, int maxVal)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) exit(1);
    fprintf(fp, "P5\n%d %d\n%d\n", width, height, maxVal);
    fwrite(img, 1, width * height, fp);
    fclose(fp);
}

// Gaussian Blur CUDA Kernel
__global__ void gaussianBlurKernel(const unsigned char* input,
                                   unsigned char* output,
                                   int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int px = x + kx;
            int py = y + ky;

            if (px < 0) px = 0;
            if (px >= width) px = width - 1;
            if (py < 0) py = 0;
            if (py >= height) py = height - 1;

            size_t idx = (size_t)py * (size_t)width + (size_t)px;

            float pixel  = (float)input[idx];
            float weight = d_kernel[(ky + 1) * 3 + (kx + 1)];
            sum += pixel * weight;
        }
    }

    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    output[(size_t)y * width + x] = (unsigned char)(sum + 0.5f);
}

int main(int argc, char* argv[])
{
    if (argc < 5) {
        printf("Usage: %s input.pgm output.pgm blockSize passes\n", argv[0]);
        return 1;
    }

    const char* inFile  = argv[1];
    const char* outFile = argv[2];

    int blockSize = atoi(argv[3]);   
    int passes    = atoi(argv[4]);   
    if (blockSize <= 0) blockSize = 16;
    if (passes <= 0) passes = 1;

    printf("Block size: %d x %d\n", blockSize, blockSize);
    printf("Passes: %d\n", passes);

    int width, height, maxVal;
    unsigned char* h_input = readPGM(inFile, &width, &height, &maxVal);

    int size = width * height;
    unsigned char* h_output = (unsigned char*)malloc(size);

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(blockSize, blockSize);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < passes; i++) {
        gaussianBlurKernel<<<grid, block>>>(d_in, d_out, width, height);
        cudaDeviceSynchronize();

        unsigned char* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Total GPU time for %d passes = %.3f ms (%.3f sec)\n",
           passes, ms, ms / 1000.0f);

    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);

    writePGM(outFile, h_output, width, height, maxVal);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_input);
    free(h_output);

    printf("Output saved: %s\n", outFile);
    return 0;
}

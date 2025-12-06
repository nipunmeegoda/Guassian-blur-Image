%%writefile main_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { if ((call) != cudaSuccess) exit(1); } while (0)


// 3x3 Gaussian kernel
__constant__ float d_kernel[9] = {
    1.0f/16, 2.0f/16, 1.0f/16,
    2.0f/16, 4.0f/16, 2.0f/16,
    1.0f/16, 2.0f/16, 1.0f/16
};

// Read a binary PGM
unsigned char* readPGM(const char* filename, int* w, int* h, int* maxVal) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) exit(1);

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1) exit(1);
    if (magic[0] != 'P' || magic[1] != '5') exit(1);

    int c = fgetc(fp);
    while (c == '#') { while (fgetc(fp) != '\n'); c = fgetc(fp); }
    ungetc(c, fp);

    if (fscanf(fp, "%d %d", w, h) != 2) exit(1);
    if (fscanf(fp, "%d", maxVal) != 1) exit(1);
    fgetc(fp);

    int size = (*w) * (*h);
    unsigned char* img = (unsigned char*)malloc(size);
    if (!img) exit(1);

    if (fread(img, 1, size, fp) != (size_t)size) exit(1);
    fclose(fp);
    return img;
}

// Write a binary PGM
void writePGM(const char* filename, const unsigned char* img, int w, int h, int maxVal) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) exit(1);
    fprintf(fp, "P5\n%d %d\n%d\n", w, h, maxVal);
    fwrite(img, 1, w * h, fp);
    fclose(fp);
}

// CUDA kernel to apply a 3x3 Gaussian blur
__global__ void gaussianBlurKernel(const unsigned char* in,
                                   unsigned char* out,
                                   int w, int h) {
    // Compute the (x, y) coordinates of the pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum = 0.0f;
    // Loop over the 3x3 neighborhood 
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = x + kx;
            int py = y + ky;
            if (px < 0) px = 0; // Clamp coordinates to image boundaries
            if (px >= w) px = w - 1;
            if (py < 0) py = 0;
            if (py >= h) py = h - 1;

            size_t idx = (size_t)py * (size_t)w + (size_t)px;
            float pixel  = (float)in[idx];
            float weight = d_kernel[(ky + 1) * 3 + (kx + 1)];
            sum += pixel * weight;
        }
    }

    if (sum < 0.0f)   sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    // Store result back to output image at (x, y)
    size_t outIdx = (size_t)y * (size_t)w + (size_t)x;
    out[outIdx] = (unsigned char)(sum + 0.5f);
}

int main(int argc, char* argv[]) {
    if (argc < 3) return 1; // Usage: ./blur_cuda input.pgm output.pgm [blockSize] [passes]

    const char* inFile  = argv[1];
    const char* outFile = argv[2];

    int blockSize = (argc >= 4) ? atoi(argv[3]) : 16;
    if (blockSize < 1) blockSize = 16;
    if (blockSize * blockSize > 1024) blockSize = 32;

    int passes = (argc >= 5) ? atoi(argv[4]) : 1;
    if (passes < 1) passes = 1;

    int w, h, maxVal;
    unsigned char* h_in  = readPGM(inFile, &w, &h, &maxVal);
    int size = w * h;
    unsigned char* h_out = (unsigned char*)malloc(size);
    if (!h_out) exit(1);

    unsigned char *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  size));
    CHECK_CUDA(cudaMalloc(&d_out, size));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));


    // Each block is blockSize x blockSize threads.
    dim3 block(blockSize, blockSize);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < passes; i++) {
        gaussianBlurKernel<<<grid, block>>>(d_in, d_out, w, h);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Swap input and output pointers for next iteration
        // (so the output of this pass becomes the input of the next pass)
        unsigned char* tmp = d_in; d_in = d_out; d_out = tmp;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("%.6f ms (block %d, passes %d)\n", ms, blockSize, passes);

    CHECK_CUDA(cudaMemcpy(h_out, d_in, size, cudaMemcpyDeviceToHost));
    writePGM(outFile, h_out, w, h, maxVal);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}

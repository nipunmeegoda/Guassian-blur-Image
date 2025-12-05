// ------------------------------------------------------------
// main_cuda.cu — Single-pass CUDA Gaussian Blur for PGM (P5)
// Standalone, works on Tesla T4 (sm_75) with 64-bit safe indexing
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d → %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(1); \
        } \
    } while (0)

// 3×3 Gaussian kernel
__constant__ float d_kernel[9] = {
    1.0f/16, 2.0f/16, 1.0f/16,
    2.0f/16, 4.0f/16, 2.0f/16,
    1.0f/16, 2.0f/16, 1.0f/16
};

// ------------------------------------------------------------
// Read PGM (P5 binary, 8-bit)
// ------------------------------------------------------------
unsigned char* readPGM(const char* filename, int* width, int* height, int* maxVal)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        exit(1);
    }

    char magic[3] = {0};
    if (fscanf(fp, "%2s", magic) != 1) {
        fprintf(stderr, "Error: Failed to read magic number.\n");
        exit(1);
    }

    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Only binary P5 PGM supported (magic=%s).\n", magic);
        exit(1);
    }

    int c = fgetc(fp);
    while (c == '#') {
        while (fgetc(fp) != '\n'); // skip comment line
        c = fgetc(fp);
    }
    ungetc(c, fp);

    if (fscanf(fp, "%d %d", width, height) != 2) {
        fprintf(stderr, "Error: Failed to read width/height.\n");
        exit(1);
    }

    if (fscanf(fp, "%d", maxVal) != 1) {
        fprintf(stderr, "Error: Failed to read maxVal.\n");
        exit(1);
    }

    fgetc(fp); // consume single whitespace after maxVal

    int size = (*width) * (*height);
    unsigned char* img = (unsigned char*)malloc(size);
    if (!img) {
        fprintf(stderr, "Error: malloc failed for image of size %d\n", size);
        exit(1);
    }

    size_t readCount = fread(img, 1, size, fp);
    if (readCount != (size_t)size) {
        fprintf(stderr, "Warning: Expected %d bytes, read %zu bytes.\n",
                size, readCount);
    }

    fclose(fp);
    return img;
}

// ------------------------------------------------------------
// Write PGM (P5)
// ------------------------------------------------------------
void writePGM(const char* filename, const unsigned char* img,
              int width, int height, int maxVal)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing.\n", filename);
        exit(1);
    }

    fprintf(fp, "P5\n%d %d\n%d\n", width, height, maxVal);
    fwrite(img, 1, width * height, fp);
    fclose(fp);
}

// ------------------------------------------------------------
// Simple stats for sanity-check
// ------------------------------------------------------------
void print_stats(const char* label, const unsigned char* img, int size)
{
    int minv = 255;
    int maxv = 0;
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        int v = img[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v;
    }

    double mean = sum / (double)size;
    printf("%s → min: %d, max: %d, mean: %.3f\n", label, minv, maxv, mean);
}

// ------------------------------------------------------------
// CUDA Gaussian Blur Kernel — single pass
// ------------------------------------------------------------
__global__ void gaussianBlurKernel(const unsigned char* input,
                                   unsigned char* output,
                                   int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int px = x + kx;
            int py = y + ky;

            if (px < 0) px = 0;
            if (px >= width) px = width - 1;
            if (py < 0) py = 0;
            if (py >= height) py = height - 1;

            // 64-bit safe indexing
            size_t idx = (size_t)py * (size_t)width + (size_t)px;

            float pixel  = (float)input[idx];
            float weight = d_kernel[(ky + 1) * 3 + (kx + 1)];

            sum += pixel * weight;
        }
    }

    if (sum < 0.0f) sum = 0.0f;
    if (sum > 255.0f) sum = 255.0f;

    size_t outIdx = (size_t)y * (size_t)width + (size_t)x;
    output[outIdx] = (unsigned char)(sum + 0.5f); // round
}

// ------------------------------------------------------------
// MAIN — single-pass Gaussian blur on GPU
// ------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }

    const char* inFile  = argv[1];
    const char* outFile = argv[2];

    int width, height, maxVal;
    unsigned char* h_input = readPGM(inFile, &width, &height, &maxVal);

    int size = width * height;
    unsigned char* h_output = (unsigned char*)malloc(size);
    if (!h_output) {
        fprintf(stderr, "Error: malloc failed for output.\n");
        return 1;
    }

    print_stats("Input", h_input, size);

    // Make sure we have at least one CUDA device
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available.\n");
        return 1;
    }

    unsigned char *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    int passes = 50;   // ⭐ YOU CAN CHANGE THIS NUMBER

for (int i = 0; i < passes; i++) {
    gaussianBlurKernel<<<grid, block>>>(d_in, d_out, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Swap buffers for next iteration
    unsigned char* tmp = d_in;
    d_in = d_out;
    d_out = tmp;
}

    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n",
                cudaGetErrorString(kerr));
        return 1;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost));

    print_stats("Output", h_output, size);

    writePGM(outFile, h_output, width, height, maxVal);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_input);
    free(h_output);

    printf("Done. Output saved: %s\n", outFile);
    return 0;
}

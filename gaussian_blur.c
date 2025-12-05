#include "gaussian_blur.h"

unsigned char* readPGM(const char *filename, int *width, int *height, int *maxVal) {
    FILE *fp = fopen(filename, "rb");

    char magic[3];
    fscanf(fp, "%2s", magic);

    int c = fgetc(fp);
    while (c == '#') {
        while (fgetc(fp) != '\n') {}
        c = fgetc(fp);
    }
    ungetc(c, fp);

    fscanf(fp, "%d %d", width, height);
    fscanf(fp, "%d", maxVal);
    fgetc(fp);

    int size = (*width) * (*height);
    unsigned char *image = malloc(size);

    fread(image, 1, size, fp);
    fclose(fp);
    return image;
}

void writePGM(const char *filename, unsigned char *image, int width, int height, int maxVal) {
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "P2\n%d %d\n%d\n", width, height, maxVal);
    for (int i = 0; i < width * height; i++)
        fprintf(fp, "%d ", image[i]);

    fclose(fp);
}

void gaussianBlurWithScratch(const unsigned char *input, unsigned char *output,
                             int width, int height, unsigned short *scratch)
{
    const int lastX = width - 1;
    const int lastY = height - 1;
    const size_t rowStride = (size_t)width;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const unsigned char *row = input + y * rowStride;
        unsigned short *tmpRow = scratch + y * rowStride;

        for (int x = 0; x < width; x++) {
            int l = (x == 0) ? 0 : x - 1;
            int r = (x == lastX) ? lastX : x + 1;
            tmpRow[x] =
                row[l] * GAUSSIAN_WEIGHTS[0] +
                row[x] * GAUSSIAN_WEIGHTS[1] +
                row[r] * GAUSSIAN_WEIGHTS[2];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        int t = (y == 0) ? 0 : y - 1;
        int b = (y == lastY) ? lastY : y + 1;

        const unsigned short *top = scratch + t * rowStride;
        const unsigned short *mid = scratch + y * rowStride;
        const unsigned short *bot = scratch + b * rowStride;
        unsigned char *out = output + y * rowStride;

        for (int x = 0; x < width; x++) {
            unsigned int acc =
                top[x] * GAUSSIAN_WEIGHTS[0] +
                mid[x] * GAUSSIAN_WEIGHTS[1] +
                bot[x] * GAUSSIAN_WEIGHTS[2];

            out[x] = (unsigned char)((acc + (GAUSS_SCALE >> 1)) / GAUSS_SCALE);
        }
    }
}

void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height) {
    unsigned short *scratch = malloc((size_t)width * height * sizeof(unsigned short));
    gaussianBlurWithScratch(input, output, width, height, scratch);
    free(scratch);
}

void gaussianBlurMultiple(unsigned char *input, unsigned char *output,
                          int width, int height, int passes)
{
    size_t count = (size_t)width * height;
    unsigned short *scratch = malloc(count * sizeof(unsigned short));

    unsigned char *current = input;
    unsigned char *next = output;

    for (int i = 0; i < passes; i++) {
        gaussianBlurWithScratch(current, next, width, height, scratch);
        unsigned char *tmp = current; current = next; next = tmp;
    }

    if (current != input)
        memcpy(input, current, count);

    free(scratch);
}

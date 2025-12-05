#include "gaussian_blur.h"


unsigned char* readPGM(const char *filename, int *width, int *height, int *maxVal) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    char magic[3];
    fscanf(fp, "%2s", magic);

    if (strcmp(magic, "P2") != 0 && strcmp(magic, "P5") != 0) {
        printf("Error: Unsupported PGM format! Must be P2 or P5.\n");
        exit(1);
    }

    int c = fgetc(fp);
    while (c == '#') {
        while (fgetc(fp) != '\n') { }
        c = fgetc(fp);
    }
    ungetc(c, fp);

    fscanf(fp, "%d %d", width, height);
    fscanf(fp, "%d", maxVal);
    fgetc(fp);

    int size = (*width) * (*height);
    unsigned char *image = malloc(size);

    if (strcmp(magic, "P5") == 0) {
        fread(image, 1, size, fp);
    } else {
        for (int i = 0; i < size; i++)
            fscanf(fp, "%hhu", &image[i]);
    }

    fclose(fp);
    return image;
}

void writePGM(const char *filename, unsigned char *image, int width, int height, int maxVal) {
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", maxVal);

    for (int i = 0; i < width * height; i++) {
        fprintf(fp, "%d ", image[i]);
    }

    fclose(fp);
}

void gaussianBlurWithScratch(const unsigned char *input, unsigned char *output,
                             int width, int height, unsigned short *scratch)
{
    if (width <= 0 || height <= 0 || scratch == NULL || input == NULL || output == NULL) {
        return;
    }

    const int lastX = width - 1;    
    const int lastY = height - 1;
    const size_t rowStride = (size_t)width;

    // Horizontal pass: convolve each row with [1 2 1]
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const unsigned char *row = input + y * rowStride;
        unsigned short *tmpRow = scratch + y * rowStride;

        for (int x = 0; x < width; x++) {
            const int leftIdx = (x == 0) ? 0 : x - 1;
            const int rightIdx = (x == lastX) ? lastX : x + 1;

            const unsigned int acc =
                row[leftIdx] * GAUSSIAN_WEIGHTS[0] +
                row[x]       * GAUSSIAN_WEIGHTS[1] +
                row[rightIdx]* GAUSSIAN_WEIGHTS[2];

            tmpRow[x] = (unsigned short)acc;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const int topIdx = (y == 0) ? 0 : y - 1;
        const int bottomIdx = (y == lastY) ? lastY : y + 1;

        const unsigned short *topRow = scratch + topIdx * rowStride;
        const unsigned short *midRow = scratch + y * rowStride;
        const unsigned short *bottomRow = scratch + bottomIdx * rowStride;
        unsigned char *outRow = output + y * rowStride;

        for (int x = 0; x < width; x++) {
            const unsigned int acc =
                topRow[x]    * GAUSSIAN_WEIGHTS[0] +
                midRow[x]    * GAUSSIAN_WEIGHTS[1] +
                bottomRow[x] * GAUSSIAN_WEIGHTS[2];

            outRow[x] = (unsigned char)((acc + (GAUSS_SCALE >> 1)) / GAUSS_SCALE);
        }
    }
}


void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height) {
    const size_t scratchSize = (size_t)width * height;
    unsigned short *scratch = NULL;

    if (scratchSize > 0) {
        scratch = (unsigned short *)malloc(scratchSize * sizeof(unsigned short));
    }

    gaussianBlurWithScratch(input, output, width, height, scratch);

    free(scratch);
}


void gaussianBlurMultiple(unsigned char *input, unsigned char *output,
                          int width, int height, int passes)
{
    if (passes <= 0) {
        return;
    }

    const size_t pixelCount = (size_t)width * height;
    unsigned short *scratch = NULL;

    if (pixelCount > 0) {
        scratch = (unsigned short *)malloc(pixelCount * sizeof(unsigned short));
    }

    unsigned char *current = input;
    unsigned char *next = output;
    unsigned char *const firstBuffer = input;

    for (int i = 0; i < passes; i++) {
        gaussianBlurWithScratch(current, next, width, height, scratch);

        unsigned char *tmp = current;
        current = next;
        next = tmp;
    }

    if (current != firstBuffer && pixelCount > 0) {
        memcpy(firstBuffer, current, pixelCount);
    }

    free(scratch);
}

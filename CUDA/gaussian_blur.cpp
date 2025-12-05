#include "gaussian_blur.h"

// ------------------------------------------------------
// Read PGM (supports P2 and P5 formats)
// ------------------------------------------------------
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

    // Skip comment lines
    while (c == '#') {
        while (fgetc(fp) != '\n') { }
        c = fgetc(fp);
    }
    ungetc(c, fp);

    fscanf(fp, "%d %d", width, height);
    fscanf(fp, "%d", maxVal);
    fgetc(fp);

    int size = (*width) * (*height);
    unsigned char *image = (unsigned char*)malloc(size);

    if (strcmp(magic, "P5") == 0) {
        fread(image, 1, size, fp);
    } else {
        for (int i = 0; i < size; i++)
            fscanf(fp, "%hhu", &image[i]);
    }

    fclose(fp);
    return image;
}

// ------------------------------------------------------
// Save PGM (always ASCII P2)
// ------------------------------------------------------
void writePGM(const char *filename, unsigned char *image, int width, int height, int maxVal) {
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", maxVal);

    for (int i = 0; i < width * height; i++)
        fprintf(fp, "%d ", image[i]);

    fclose(fp);
}

// ------------------------------------------------------
// Single-pass Gaussian Blur
// ------------------------------------------------------
void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float sum = 0.0;

            for (int ky = -RADIUS; ky <= RADIUS; ky++) {
                for (int kx = -RADIUS; kx <= RADIUS; kx++) {

                    int px = x + kx;
                    int py = y + ky;

                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        sum += input[py * width + px] *
                               GAUSSIAN_KERNEL[ky + RADIUS][kx + RADIUS];
                    }
                }
            }

            output[y * width + x] = (unsigned char)sum;
        }
    }
}

// ------------------------------------------------------
// Multi-pass stronger Gaussian Blur
// ------------------------------------------------------
void gaussianBlurMultiple(unsigned char *input, unsigned char *output,
                          int width, int height, int passes)
{
    for (int i = 0; i < passes; i++) {
        gaussianBlur(input, output, width, height);

        unsigned char *tmp = input;
        input = output;
        output = tmp;
    }
}

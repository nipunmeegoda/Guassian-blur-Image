#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------
// Configuration
// --------------------------------------------

#define KERNEL_SIZE 3
#define RADIUS 1

// Fixed 3x3 Gaussian Kernel (standard blur)
static const float GAUSSIAN_KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
    {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f},
    {2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f},
    {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}
};

// --------------------------------------------
// Function Declarations
// --------------------------------------------

// File I/O
unsigned char* readPGM(const char *filename, int *width, int *height, int *maxVal);
void writePGM(const char *filename, unsigned char *image, int width, int height, int maxVal);

// Gaussian Blur (CPU)
void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height);

// Multi-pass blur (stronger blur)
void gaussianBlurMultiple(unsigned char *input, unsigned char *output,
                          int width, int height, int passes);

#endif

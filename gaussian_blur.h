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
#define GAUSS_SCALE 16

// Integer representation of the 3x3 Gaussian kernel (weights sum to 16)
static const unsigned char GAUSSIAN_WEIGHTS[KERNEL_SIZE] = {1, 2, 1};

// --------------------------------------------
// Function Declarations
// --------------------------------------------

// File I/O
unsigned char* readPGM(const char *filename, int *width, int *height, int *maxVal);
void writePGM(const char *filename, unsigned char *image, int width, int height, int maxVal);

// Gaussian Blur (CPU)
void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height);

// Optimized single-pass blur that reuses a caller-provided scratch buffer
void gaussianBlurWithScratch(const unsigned char *input, unsigned char *output,
                             int width, int height, unsigned short *scratch);

// Multi-pass blur (stronger blur)
void gaussianBlurMultiple(unsigned char *input, unsigned char *output,
                          int width, int height, int passes);

#endif

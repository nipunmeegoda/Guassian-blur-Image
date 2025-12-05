#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define KERNEL_SIZE 3
#define RADIUS 1
#define GAUSS_SCALE 16


static const unsigned char GAUSSIAN_WEIGHTS[KERNEL_SIZE] = {1, 2, 1};


unsigned char* readPGM(const char *filename, int *width, int *height, int *maxVal);
void writePGM(const char *filename, unsigned char *image, int width, int height, int maxVal);


void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height);


void gaussianBlurWithScratch(const unsigned char *input, unsigned char *output,
                             int width, int height, unsigned short *scratch);


void gaussianBlurMultiple(unsigned char *input, unsigned char *output,
                          int width, int height, int passes);

#endif

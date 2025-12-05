#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>     // for gettimeofday()
#include "gaussian_blur.h"

// High-precision wall clock timer
double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) 
{
    if (argc < 3 || argc > 4) {
        printf("Usage: %s input.pgm output.pgm [passes]\n", argv[0]);
        return 1;
    }

    int width, height, maxVal;
    int passes = (argc == 4) ? atoi(argv[3]) : 5;

    if (passes <= 0) {
        printf("Error: passes must be > 0\n");
        return 1;
    }

    // Load input image
    unsigned char *input = readPGM(argv[1], &width, &height, &maxVal);
    if (!input) {
        fprintf(stderr, "Error: failed to read input image\n");
        return 1;
    }

    unsigned char *output = malloc(width * height);
    if (!output) {
        fprintf(stderr, "Error: failed to allocate output buffer\n");
        free(input);
        return 1;
    }

    printf("Applying Gaussian Blur %d times...\n", passes);

    // ----------- Timing Start --------------
    double start_time = get_time_sec();

    gaussianBlurMultiple(input, output, width, height, passes);

    double end_time = get_time_sec();
    // ----------- Timing End ----------------

    writePGM(argv[2], input, width, height, maxVal);

    printf("Blur complete. Output saved as %s\n", argv[2]);
    printf("Time taken: %f seconds\n", end_time - start_time);

    free(input);
    free(output);

    return 0;
}

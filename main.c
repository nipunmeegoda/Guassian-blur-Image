#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "gaussian_blur.h"

double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[])
{
    int width, height, maxVal;
    int passes = (argc == 4) ? atoi(argv[3]) : 5;

    unsigned char *input = readPGM(argv[1], &width, &height, &maxVal);
    unsigned char *output = malloc(width * height);

    double start = get_time_sec();

    gaussianBlurMultiple(input, output, width, height, passes);

    double end = get_time_sec();

    writePGM(argv[2], input, width, height, maxVal);

    printf("Execution time: %.6f seconds\n", end - start);

    free(input);
    free(output);
    return 0;
}

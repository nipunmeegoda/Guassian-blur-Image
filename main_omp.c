#include <omp.h>
#include "gaussian_blur.h"

int main(int argc, char *argv[]) {
    int width, height, maxVal;
    int threads = atoi(argv[3]);
    int passes = (argc == 5) ? atoi(argv[4]) : 5;

    omp_set_num_threads(threads);

    unsigned char *input = readPGM(argv[1], &width, &height, &maxVal);
    unsigned char *output = malloc(width * height);

    double start = omp_get_wtime();
    gaussianBlurMultiple(input, output, width, height, passes);
    double end = omp_get_wtime();

    writePGM(argv[2], input, width, height, maxVal);

    printf("Execution time: %.6f seconds\n", end - start);

    free(input);
    free(output);
}

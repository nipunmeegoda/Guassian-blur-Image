#include <omp.h>
#include "gaussian_blur.h"

int main(int argc, char *argv[]) {

    if (argc < 4 || argc > 5) {
        printf("Usage: %s input.pgm output.pgm <num_threads> [passes]\n", argv[0]);
        return 1;
    }

    int width, height, maxVal;
    int threads = atoi(argv[3]);
    int passes = (argc == 5) ? atoi(argv[4]) : 5;

    if (threads <= 0) {
        printf("Error: num_threads must be > 0\n");
        return 1;
    }

    if (passes <= 0) {
        printf("Error: passes must be > 0\n");
        return 1;
    }

    omp_set_num_threads(threads);

    // Load image
    unsigned char *input = readPGM(argv[1], &width, &height, &maxVal);
    unsigned char *output = malloc(width * height);

    printf("Applying Gaussian Blur %d times using %d threads...\n",
           passes, threads);

    double start = omp_get_wtime();

    gaussianBlurMultiple(input, output, width, height, passes);

    double end = omp_get_wtime();

    // Final blurred image is in "input"
    writePGM(argv[2], input, width, height, maxVal);

    printf("\nBlur complete.\n");
    printf("Output saved to: %s\n", argv[2]);
    printf("Execution time: %.6f seconds\n", end - start);

    free(input);
    free(output);

    return 0;
}

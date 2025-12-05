#include "gaussian_blur.h"

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

    // Load image
    unsigned char *input = readPGM(argv[1], &width, &height, &maxVal);
    unsigned char *output = malloc(width * height);

    printf("Applying Gaussian Blur %d times...\n", passes);
    gaussianBlurMultiple(input, output, width, height, passes);

    // Output stored in "input" after swaps
    writePGM(argv[2], input, width, height, maxVal);

    printf("Blur complete. Output saved as %s\n", argv[2]);

    free(input);
    free(output);

    return 0;
}

#include <mpi.h>
#include <string.h>
#include "gaussian_blur.h"

// Exchange halo (border) rows between neighboring MPI ranks.
static void exchangeHalos(unsigned char *current, int localRows, int width,
                          int rank, int size)
{
    const size_t rowStride = (size_t)width;

    // ---- Top halo exchange ----
    if (rank > 0) {
        MPI_Sendrecv(current + rowStride, width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     current, width, MPI_UNSIGNED_CHAR, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Rank 0 has no upper neighbor:
        // Just copy its first real row into the top halo (replicate border)
        memcpy(current, current + rowStride, width);
    }

    // ---- Bottom halo exchange ----
    if (rank < size - 1) {
        MPI_Sendrecv(current + localRows * rowStride, width, MPI_UNSIGNED_CHAR, rank + 1, 1,
                     current + (localRows + 1) * rowStride, width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // Last rank has no lower neighbor:
        memcpy(current + (localRows + 1) * rowStride, current + localRows * rowStride, width);
    }
}

int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

     // mpirun -np <size> ./blur_mpi input.pgm output.pgm [passes]
    int passes = (argc == 4) ? atoi(argv[3]) : 5;

    int width, height, maxVal;
    unsigned char *fullImage = NULL;

    if (rank == 0) {
        fullImage = readPGM(argv[1], &width, &height, &maxVal); // Only rank 0 will hold the entire image
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *rowsPerRank = (int *)malloc(size * sizeof(int)); // local rows per rank
    int *displs      = (int *)malloc(size * sizeof(int));// starting offset (in pixels) for each rank
    int *sendCounts  = (int *)malloc(size * sizeof(int));// number of pixels sent to each rank

    int baseRows = height / size;
    int remainder = height % size;
    int offset = 0;

    for (int p = 0; p < size; p++) {
        rowsPerRank[p] = baseRows + (p < remainder ? 1 : 0);
        sendCounts[p] = rowsPerRank[p] * width;
        displs[p] = offset;
        offset += sendCounts[p];
    }

    int localRows = rowsPerRank[rank];
    size_t chunkElements = (size_t)(localRows + 2) * width;

    // bufferA and bufferB each store localRows + 2 rows (including halos)
    unsigned char *bufferA = (unsigned char *)calloc(chunkElements, sizeof(unsigned char));
    unsigned char *bufferB = (unsigned char *)calloc(chunkElements, sizeof(unsigned char));
    unsigned short *scratch = (unsigned short *)malloc(chunkElements * sizeof(unsigned short));

    // Scatter the full image from rank 0 to all ranks.
    MPI_Scatterv(fullImage, sendCounts, displs, MPI_UNSIGNED_CHAR,
                 bufferA + width, localRows * width, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    unsigned char *current = bufferA;
    unsigned char *next = bufferB;

    // Perform 'passes' consecutive Gaussian blur operations
    for (int pass = 0; pass < passes; pass++) {
        exchangeHalos(current, localRows, width, rank, size);
        gaussianBlurWithScratch(current, next, width, localRows + 2, scratch);

        unsigned char *tmp = current;
        current = next;
        next = tmp;
    }

    memcpy(bufferA + width, current + width, (size_t)localRows * width);

    double endTime = MPI_Wtime();

    // Gather all local results back to rank 0 into fullImage
    MPI_Gatherv(bufferA + width, localRows * width, MPI_UNSIGNED_CHAR,
                fullImage, sendCounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        writePGM(argv[2], fullImage, width, height, maxVal); // Rank 0 writes the final full blurred image to disk
        printf("MPI Blur Completed in %.6f seconds.",
               endTime - startTime);
    }

    free(bufferA);
    free(bufferB);
    free(scratch);
    free(rowsPerRank);
    free(displs);
    free(sendCounts);

    if (rank == 0)
        free(fullImage);

    MPI_Finalize();
    return 0;
}

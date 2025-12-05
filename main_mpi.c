#include <mpi.h>
#include <string.h>
#include "gaussian_blur.h"

static void exchangeHalos(unsigned char *current, int localRows, int width,
                          int rank, int size)
{
    const size_t rowStride = (size_t)width;

    if (rank > 0) {
        MPI_Sendrecv(current + rowStride, width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     current, width, MPI_UNSIGNED_CHAR, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(current, current + rowStride, width);
    }

    if (rank < size - 1) {
        MPI_Sendrecv(current + localRows * rowStride, width, MPI_UNSIGNED_CHAR, rank + 1, 1,
                     current + (localRows + 1) * rowStride, width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(current + (localRows + 1) * rowStride, current + localRows * rowStride, width);
    }
}

int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int passes = (argc == 4) ? atoi(argv[3]) : 5;

    int width, height, maxVal;
    unsigned char *fullImage = NULL;

    if (rank == 0) {
        fullImage = readPGM(argv[1], &width, &height, &maxVal);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *rowsPerRank = (int *)malloc(size * sizeof(int));
    int *displs      = (int *)malloc(size * sizeof(int));
    int *sendCounts  = (int *)malloc(size * sizeof(int));

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

    unsigned char *bufferA = (unsigned char *)calloc(chunkElements, sizeof(unsigned char));
    unsigned char *bufferB = (unsigned char *)calloc(chunkElements, sizeof(unsigned char));
    unsigned short *scratch = (unsigned short *)malloc(chunkElements * sizeof(unsigned short));

    MPI_Scatterv(fullImage, sendCounts, displs, MPI_UNSIGNED_CHAR,
                 bufferA + width, localRows * width, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    unsigned char *current = bufferA;
    unsigned char *next = bufferB;

    for (int pass = 0; pass < passes; pass++) {
        exchangeHalos(current, localRows, width, rank, size);
        gaussianBlurWithScratch(current, next, width, localRows + 2, scratch);

        unsigned char *tmp = current;
        current = next;
        next = tmp;
    }

    memcpy(bufferA + width, current + width, (size_t)localRows * width);

    double endTime = MPI_Wtime();

    MPI_Gatherv(bufferA + width, localRows * width, MPI_UNSIGNED_CHAR,
                fullImage, sendCounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        writePGM(argv[2], fullImage, width, height, maxVal);
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

#include <mpi.h>
#include "gaussian_blur.h"

static void exchangeHalos(unsigned char *current, int localRows, int width,
                          int rank, int size)
{
    if (localRows == 0) {
        return;
    }

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

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 || argc > 4) {
        if (rank == 0)
            printf("Usage: mpirun -np <procs> %s input.pgm output.pgm [passes]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int width = 0, height = 0, maxVal = 0;
    int passes = (argc == 4) ? atoi(argv[3]) : 5;

    if (passes <= 0) {
        if (rank == 0) {
            printf("Error: passes must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    unsigned char *fullImage = NULL;

    if (rank == 0) {
        fullImage = readPGM(argv[1], &width, &height, &maxVal);
        if (size > height) {
            printf("Error: number of processes (%d) cannot exceed image rows (%d)\n", size, height);
            free(fullImage);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *rowsPerRank = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int *sendCounts = (int *)malloc(size * sizeof(int));

    if (!rowsPerRank || !displs || !sendCounts) {
        printf("Rank %d: Memory allocation failed for metadata arrays\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int baseRows = height / size;
    int remainder = height % size;
    int offset = 0;

    for (int p = 0; p < size; p++) {
        rowsPerRank[p] = baseRows + (p < remainder ? 1 : 0);
        sendCounts[p] = rowsPerRank[p] * width;
        displs[p] = offset;
        offset += sendCounts[p];
    }

    const int localRows = rowsPerRank[rank];
    const size_t paddedRows = (size_t)(localRows + 2);
    const size_t chunkElements = paddedRows * (size_t)width;

    unsigned char *bufferA = NULL;
    unsigned char *bufferB = NULL;
    unsigned short *scratch = NULL;

    if (chunkElements > 0) {
        bufferA = (unsigned char *)calloc(chunkElements, sizeof(unsigned char));
        bufferB = (unsigned char *)calloc(chunkElements, sizeof(unsigned char));
        scratch = (unsigned short *)malloc(chunkElements * sizeof(unsigned short));
    }

    if ((chunkElements > 0) && (!bufferA || !bufferB || !scratch)) {
        printf("Rank %d: Failed to allocate buffers for %zu elements\n", rank, chunkElements);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatterv(fullImage, sendCounts, displs, MPI_UNSIGNED_CHAR,
                 bufferA ? bufferA + width : NULL,
                 localRows * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    unsigned char *current = bufferA;
    unsigned char *next = bufferB;

    for (int pass = 0; pass < passes; pass++) {
        if (current && localRows > 0) {
            exchangeHalos(current, localRows, width, rank, size);
            gaussianBlurWithScratch(current, next, width, localRows + 2, scratch);
        }

        // Swap buffers for next pass
        unsigned char *tmp = current;
        current = next;
        next = tmp;
    }

    // Ensure final data resides in bufferA for gathering
    if (current != bufferA && bufferA && localRows > 0) {
        memcpy(bufferA + width, current + width, (size_t)localRows * width);
    }

    double endTime = MPI_Wtime();

    MPI_Gatherv(bufferA ? bufferA + width : NULL,
                localRows * width, MPI_UNSIGNED_CHAR,
                fullImage, sendCounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        writePGM(argv[2], fullImage, width, height, maxVal);
        printf("MPI Blur Completed in %.6f seconds. Output saved to %s\n",
               endTime - startTime, argv[2]);
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

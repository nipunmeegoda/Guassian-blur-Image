############################################################
#   Automatic Cross-Platform Makefile for SE3082 Gaussian Blur
#   Supports: Serial, OpenMP, MPI
#   Works on: macOS Intel, macOS M1/M2/M3, Linux
############################################################

# ---------------------------------------
# Detect Homebrew GCC for OpenMP support
# ---------------------------------------
GCC_PATH := $(shell ls /opt/homebrew/bin/gcc-* /usr/local/bin/gcc-* 2>/dev/null | head -n 1)

ifeq ($(GCC_PATH),)
    OMP_CC = gcc
    $(warning ⚠ No Homebrew GCC found. If OpenMP fails, run: brew install gcc)
else
    OMP_CC = $(GCC_PATH)
endif

# ---------------------------------------
# Detect MPI compiler
# ---------------------------------------
MPI_CC := $(shell which mpicc 2>/dev/null)

ifeq ($(MPI_CC),)
    $(warning ⚠ mpicc not found. Install with: brew install open-mpi)
    MPI_CC = mpicc   # fallback (will still error if missing)
endif

# ---------------------------------------
# Compiler flags
# ---------------------------------------
CFLAGS = -O2
OMPFLAGS = -O2 -fopenmp
MPIFLAGS = -O2

SRC = gaussian_blur.c
HDR = gaussian_blur.h

# ---------------------------------------
# Build targets
# ---------------------------------------

all: serial omp mpi

# Serial version
serial:
	@echo "🔧 Building Serial Version..."
	gcc main.c $(SRC) -o blur_serial $(CFLAGS)
	@echo "✔ Output: ./blur_serial"

# OpenMP version
omp:
	@echo "🔧 Building OpenMP Version with $(OMP_CC)..."
	$(OMP_CC) main_omp.c $(SRC) -o blur_omp $(OMPFLAGS)
	@echo "✔ Output: ./blur_omp"

# MPI version
mpi:
	@echo "🔧 Building MPI Version with $(MPI_CC)..."
	$(MPI_CC) main_mpi.c $(SRC) -o blur_mpi $(MPIFLAGS)
	@echo "✔ Output: ./blur_mpi"

# Clean
clean:
	rm -f blur_serial blur_omp blur_mpi
	@echo "🧹 Clean complete."

############################################################
# End of Makefile
############################################################

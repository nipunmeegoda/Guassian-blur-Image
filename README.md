# Gaussian Blur – Serial, OpenMP, MPI & CUDA

Local Machine used to test - Apple Macbook Pro 2021 with m1 pro chip
Cloud platform used to test CUDA - Google Collab (Tesla t4 GPU)

This project implements a 3×3 Gaussian blur on grayscale PGM images using four approaches:

- **Serial (C)**
- **OpenMP (shared–memory parallel CPU)**
- **MPI (distributed–memory parallel CPU)**
- **CUDA (GPU- Google Colab)**

Source Code for Serial,OpenMp and MPI is in the same directory for the convience of using Makefile to compile the code.

Files inside the folder

- **main.c - serial implementation of the code **
- **main_omp.c - OpenMP implementation of the code **
- **main_mpi.c - MPI implementation of the code **

- **gaussian_blur.c - Implements Gaussian blur image processing **
- **gaussian_blur.c - Declares Gaussian blur function prototypes **
- **input.pgm - sample input file **

- **CUDA > main_cuda,cu - CUDA implementation of the code that can be used in Google Colab **


#Compilation of Serial,OpenMP and MPI codes

Simply execute command for the makefile
```
make
```

without makefile

Serial code
```
gcc main.c gaussian_blur.c -o blur_serial -O2
```
OpenMP code 
```
gcc-15 main_omp.c gaussian_blur.c -o blur_omp -O2 -fopenmp
```
MPI code
```
mpicc main_mpi.c gaussian_blur.c -o blur_mpi -O2
```

#Execution

Serial code
```
./blur_serial input.pgm output.pgm [passes] 
```
input.pgm – input grayscale PGM image (P5 or P2)
output.pgm – output file
[passes] – optional number of blur passes


OpenMP code
```
./blur_omp input.pgm output.pgm <num_threads> [passes]

```
<num_threads> – number of OpenMP threads
[passes] – optional number of blur passes

MPI code
```
mpirun -np <num_procs> ./blur_mpi input.pgm output.pgm [passes]
```
<num_procs> – number of MPI processes
[passes] – optional number of blur passes (default: 5)


# CUDA compilation and execution

first install cuda on Google Colab (select t4 gpu as a runtime type)
```
!sudo apt-get update
!sudo apt-get install -y cuda-toolkit-12-4
```

Compilation
```
!nvcc -arch=sm_75 main_cuda.cu -o blur_cuda
```
Execution
```
!./blur_cuda input.pgm output.pgm <block_size> <passes> 
```

#!/usr/bin/env bash

module load gcc
module load openmpi

# Go to code directory and build
make clean && make all

# Ensure results folder exists
mkdir -p results

########################################
# Strong scaling (fixed problem size)
########################################

# Serial baseline
./heat_serial 500 500 | tee results/strong_serial_n500.log

# OpenMP: vary threads, same grid
./heat_openmp 500 500 1 | tee results/strong_openmp_n500_t1.log
./heat_openmp 500 500 2 | tee results/strong_openmp_n500_t2.log
./heat_openmp 500 500 4 | tee results/strong_openmp_n500_t4.log
./heat_openmp 500 500 8 | tee results/strong_openmp_n500_t8.log

# MPI: vary ranks, same grid (label scenario="strong_n500")
mpirun -np 2 ./heat_mpi 500 500 strong_n500 | tee results/strong_mpi_n500_p2.log
mpirun -np 4 ./heat_mpi 500 500 strong_n500 | tee results/strong_mpi_n500_p4.log
mpirun -np 8 ./heat_mpi 500 500 strong_n500 | tee results/strong_mpi_n500_p8.log

# Hybrid: keep total tasks ~8, vary processes/threads
mpirun -np 1 ./heat_hybrid 500 500 8 | tee results/strong_hybrid_n500_p1_t8.log
mpirun -np 2 ./heat_hybrid 500 500 4 | tee results/strong_hybrid_n500_p2_t4.log
mpirun -np 4 ./heat_hybrid 500 500 2 | tee results/strong_hybrid_n500_p4_t2.log
mpirun -np 8 ./heat_hybrid 500 500 1 | tee results/strong_hybrid_n500_p8_t1.log

########################################
# Weak scaling (more work with more ranks)
########################################

# Baseline single-process MPI (label scenario="weak")
mpirun -np 1 ./heat_mpi 250 500 weak | tee results/weak_mpi_p1_n250.log

# Increase grid so rows per rank stay similar (same "weak" label)
mpirun -np 2 ./heat_mpi 500 500 weak | tee results/weak_mpi_p2_n500.log
mpirun -np 4 ./heat_mpi 1000 500 weak | tee results/weak_mpi_p4_n1000.log
mpirun -np 8 ./heat_mpi 2000 500 weak | tee results/weak_mpi_p8_n2000.log

########################################
# Thread-to-thread scaling (OpenMP only)
########################################

./heat_openmp 1000 500 1 | tee results/thread_openmp_n1000_t1.log
./heat_openmp 1000 500 2 | tee results/thread_openmp_n1000_t2.log
./heat_openmp 1000 500 4 | tee results/thread_openmp_n1000_t4.log
./heat_openmp 1000 500 8 | tee results/thread_openmp_n1000_t8.log

# Makefile for Heat Equation Solvers
# Supports serial, OpenMP, MPI, and hybrid MPI+OpenMP implementations

# Compilers
CC = gcc
MPICC = mpicc

# Compiler flags
CFLAGS = -O3 -Wall -std=c99
OMPFLAGS = -fopenmp
LDFLAGS = -lm

# Target executables
SERIAL_TARGET = heat_serial
OPENMP_TARGET = heat_openmp
MPI_TARGET = heat_mpi
HYBRID_TARGET = heat_hybrid

# Source files
SERIAL_SRC = src/heat_serial.c
OPENMP_SRC = src/heat_openmp.c
MPI_SRC = src/heat_mpi.c
HYBRID_SRC = src/heat_hybrid.c

# Default target
all: $(SERIAL_TARGET) $(OPENMP_TARGET) $(MPI_TARGET) $(HYBRID_TARGET)

# Serial version
$(SERIAL_TARGET): $(SERIAL_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $(SERIAL_TARGET)"

# OpenMP version
$(OPENMP_TARGET): $(OPENMP_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $(OPENMP_TARGET)"

# MPI version
$(MPI_TARGET): $(MPI_SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $(MPI_TARGET)"

# Hybrid MPI+OpenMP version
$(HYBRID_TARGET): $(HYBRID_SRC)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $(HYBRID_TARGET)"

# Individual build targets
serial: $(SERIAL_TARGET)

openmp: $(OPENMP_TARGET)

mpi: $(MPI_TARGET)

hybrid: $(HYBRID_TARGET)

# Clean up
clean:
	rm -f $(SERIAL_TARGET) $(OPENMP_TARGET) $(MPI_TARGET) $(HYBRID_TARGET)
	rm -f *.txt
	rm -f results/*.txt results/*.png results/*.dat
	@echo "Cleaned build files and results"

# Run tests (small problem size for quick verification)
test: all
	@echo "=== Running verification tests ==="
	@echo "Testing serial version..."
	./$(SERIAL_TARGET) 100 100
	@echo ""
	@echo "Testing OpenMP version (4 threads)..."
	./$(OPENMP_TARGET) 100 100 4
	@echo ""
	@echo "Testing MPI version (2 processes)..."
	mpirun -np 2 ./$(MPI_TARGET) 100 100
	@echo ""
	@echo "Testing hybrid version (2 MPI x 2 OpenMP)..."
	mpirun -np 2 ./$(HYBRID_TARGET) 100 100 2
	@echo ""
	@echo "=== All tests completed ==="

# Help target
help:
	@echo "Heat Equation Solver - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all versions (default)"
	@echo "  serial     - Build serial version only"
	@echo "  openmp     - Build OpenMP version only"
	@echo "  mpi        - Build MPI version only"
	@echo "  hybrid     - Build hybrid MPI+OpenMP version only"
	@echo "  test       - Build and run quick verification tests"
	@echo "  clean      - Remove all build files and results"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make"
	@echo "  make test"
	@echo "  make openmp"
	@echo "  ./heat_serial 1000 1000"
	@echo "  ./heat_openmp 1000 1000 8"
	@echo "  mpirun -np 4 ./heat_mpi 2000 1000"
	@echo "  mpirun -np 4 ./heat_hybrid 2000 1000 4"

.PHONY: all serial openmp mpi hybrid clean test help

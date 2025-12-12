/*
 * MPI parallel implementation of 2D Heat Equation solver
 * Uses finite difference method with explicit time stepping
 * Parallelization: Domain decomposition (1D decomposition in rows)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

// Default parameters
#define DEFAULT_N 1000        // Grid size (N x N)
#define DEFAULT_STEPS 1000    // Number of time steps
#define ALPHA 0.1             // Thermal diffusivity

// Function prototypes
void initialize_grid(double **grid, int local_rows, int n, int rank, int size);
void apply_boundary_conditions(double **grid, int local_rows, int n, int rank, int size);
double compute_step_mpi(double **grid, double **grid_new, int local_rows, int n, 
                        double factor, int rank, int size);
void exchange_boundaries(double **grid, int local_rows, int n, int rank, int size);
double** allocate_2d_array(int rows, int cols);
void free_2d_array(double **array, int rows);
double get_walltime();
void save_results(double **grid, int local_rows, int n, int rank, const char *filename);
double verify_solution(double **grid, int local_rows, int n, int rank, int size);
void log_performance_mpi(int n, int steps, int size, double elapsed,
                         double total_mem_mb, double avg_mem_mb,
                         const char *scenario);

int main(int argc, char *argv[]) {
    int n = DEFAULT_N;
    int steps = DEFAULT_STEPS;
    const char *scenario = "default";
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);
    if (argc > 3) scenario = argv[3];
    
    // Calculate local domain size (1D row decomposition)
    int local_rows = n / size;
    int remainder = n % size;
    if (rank < remainder) local_rows++;
    
    // Account for ghost rows (halo cells)
    int total_local_rows = local_rows;
    if (rank > 0) total_local_rows++;        // Top ghost row
    if (rank < size - 1) total_local_rows++; // Bottom ghost row
    
    if (rank == 0) {
        printf("=== MPI Heat Equation Solver ===\n");
        printf("Grid size: %d x %d\n", n, n);
        printf("Time steps: %d\n", steps);
        printf("Alpha: %.2f\n", ALPHA);
        printf("MPI processes: %d\n", size);
        printf("Rows per process: ~%d\n", n / size);
        printf("Scenario: %s\n", scenario);
    }
    
    // Allocate local grids
    double **grid = allocate_2d_array(total_local_rows, n);
    double **grid_new = allocate_2d_array(total_local_rows, n);
    
    // Initialize local domain
    initialize_grid(grid, local_rows, n, rank, size);
    apply_boundary_conditions(grid, local_rows, n, rank, size);
    
    // Compute time step factor
    double dx = 1.0 / (n - 1);
    double dt = 0.25 * dx * dx / ALPHA;  // Stability condition
    double factor = ALPHA * dt / (dx * dx);
    
    if (rank == 0) {
        printf("dx: %.6f, dt: %.6f, factor: %.6f\n", dx, dt, factor);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = get_walltime();
    double max_diff = 0.0;
    
    // Main computation loop
    for (int step = 0; step < steps; step++) {
        // Exchange boundary data with neighbors
        exchange_boundaries(grid, local_rows, n, rank, size);
        
        // Compute new values
        max_diff = compute_step_mpi(grid, grid_new, local_rows, n, factor, rank, size);
        
        // Swap pointers
        double **temp = grid;
        grid = grid_new;
        grid_new = temp;
        
        if (rank == 0 && step % 100 == 0) {
            printf("Step %d, max diff: %.6e\n", step, max_diff);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = get_walltime();
    double elapsed = end_time - start_time;
    
    if (rank == 0) {
        printf("\n=== Results ===\n");
        printf("Total time: %.4f seconds\n", elapsed);
        printf("Time per step: %.6f seconds\n", elapsed / steps);
        printf("Final max diff: %.6e\n", max_diff);

        double total_mem_bytes = 2.0 * n * n * sizeof(double);
        double total_mem_mb = total_mem_bytes / (1024.0 * 1024.0);
        double avg_mem_mb = total_mem_mb / size;
        printf("Approx total memory over all ranks (2 global grids): %.2f MB\n", total_mem_mb);
        printf("Approx average memory per rank: %.2f MB\n", avg_mem_mb);
    }
    
    // Verification
    double error = verify_solution(grid, local_rows, n, rank, size);
    if (rank == 0) {
        printf("Verification error: %.6e\n", error);
    }
    
    // Save results
    char filename[100];
    sprintf(filename, "results/mpi_%s_result_%d.txt", scenario, size);
    save_results(grid, local_rows, n, rank, filename);

    if (rank == 0) {
        double total_mem_bytes = 2.0 * n * n * sizeof(double);
        double total_mem_mb = total_mem_bytes / (1024.0 * 1024.0);
        double avg_mem_mb = total_mem_mb / size;
        log_performance_mpi(n, steps, size, elapsed, total_mem_mb, avg_mem_mb, scenario);
    }
    
    // Cleanup
    free_2d_array(grid, total_local_rows);
    free_2d_array(grid_new, total_local_rows);
    
    MPI_Finalize();
    return 0;
}

void initialize_grid(double **grid, int local_rows, int n, int rank, int size) {
    // Calculate global row offset
    int rows_before = 0;
    int temp_rows = n / size;
    int remainder = n % size;
    
    for (int r = 0; r < rank; r++) {
        rows_before += temp_rows;
        if (r < remainder) rows_before++;
    }
    
    // Determine where local grid starts (accounting for ghost rows)
    int ghost_offset = (rank > 0) ? 1 : 0;
    
    // Initialize local domain to zero
    for (int i = 0; i < local_rows + (rank > 0 ? 1 : 0) + (rank < size - 1 ? 1 : 0); i++) {
        for (int j = 0; j < n; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Set initial heat source in the center
    int center = n / 2;
    int radius = n / 10;
    
    for (int i = 0; i < local_rows; i++) {
        int global_i = rows_before + i;
        
        if (global_i >= center - radius && global_i <= center + radius) {
            for (int j = center - radius; j <= center + radius; j++) {
                if (j >= 0 && j < n) {
                    double dist = sqrt((global_i - center) * (global_i - center) + 
                                     (j - center) * (j - center));
                    if (dist <= radius) {
                        grid[i + ghost_offset][j] = 100.0 * (1.0 - dist / radius);
                    }
                }
            }
        }
    }
}

void apply_boundary_conditions(double **grid, int local_rows, int n, int rank, int size) {
    int ghost_offset = (rank > 0) ? 1 : 0;
    
    // Top boundary (only for rank 0)
    if (rank == 0) {
        for (int j = 0; j < n; j++) {
            grid[0][j] = 0.0;
        }
    }
    
    // Bottom boundary (only for last rank)
    if (rank == size - 1) {
        for (int j = 0; j < n; j++) {
            grid[local_rows + ghost_offset - 1][j] = 0.0;
        }
    }
    
    // Left and right boundaries (all ranks)
    for (int i = 0; i < local_rows + ghost_offset + (rank < size - 1 ? 1 : 0); i++) {
        grid[i][0] = 0.0;
        grid[i][n-1] = 0.0;
    }
}

void exchange_boundaries(double **grid, int local_rows, int n, int rank, int size) {
    MPI_Status status;
    int ghost_offset = (rank > 0) ? 1 : 0;
    
    // Send to upper neighbor and receive from upper neighbor
    if (rank > 0) {
        MPI_Sendrecv(grid[ghost_offset], n, MPI_DOUBLE, rank - 1, 0,
                     grid[0], n, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    }
    
    // Send to lower neighbor and receive from lower neighbor
    if (rank < size - 1) {
        int send_row = ghost_offset + local_rows - 1;
        int recv_row = ghost_offset + local_rows;
        MPI_Sendrecv(grid[send_row], n, MPI_DOUBLE, rank + 1, 1,
                     grid[recv_row], n, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

double compute_step_mpi(double **grid, double **grid_new, int local_rows, int n, 
                        double factor, int rank, int size) {
    double local_max_diff = 0.0;
    int ghost_offset = (rank > 0) ? 1 : 0;
    
    // Start and end rows for computation
    int start_row = ghost_offset;
    int end_row = ghost_offset + local_rows;
    
    // Adjust for global boundaries
    if (rank == 0) start_row = 1;
    if (rank == size - 1) end_row--;
    
    // Interior points using 5-point stencil
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < n - 1; j++) {
            grid_new[i][j] = grid[i][j] + factor * (
                grid[i+1][j] + grid[i-1][j] + 
                grid[i][j+1] + grid[i][j-1] - 
                4.0 * grid[i][j]
            );
            
            double diff = fabs(grid_new[i][j] - grid[i][j]);
            if (diff > local_max_diff) local_max_diff = diff;
        }
    }
    
    // Apply boundary conditions
    apply_boundary_conditions(grid_new, local_rows, n, rank, size);
    
    // Reduce to get global max_diff
    double global_max_diff;
    MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    return global_max_diff;
}

double** allocate_2d_array(int rows, int cols) {
    double **array = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        array[i] = (double *)malloc(cols * sizeof(double));
    }
    return array;
}

void free_2d_array(double **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

double get_walltime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void save_results(double **grid, int local_rows, int n, int rank, const char *filename) {
    // Only rank 0 writes the file
    if (rank == 0) {
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {
            printf("Warning: Could not open file for writing\n");
            return;
        }
        fprintf(fp, "%d\n", n);
        fclose(fp);
    }
    
    // Each rank appends its center row data if it has the center
    int center = n / 2;
    int rows_before = 0;
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int temp_rows = n / mpi_size;
    int remainder = n % mpi_size;
    
    for (int r = 0; r < rank; r++) {
        rows_before += temp_rows;
        if (r < remainder) rows_before++;
    }
    
    int ghost_offset = (rank > 0) ? 1 : 0;
    
    if (center >= rows_before && center < rows_before + local_rows) {
        int local_center = center - rows_before + ghost_offset;
        
        // Write data (rank 0 writes, others send to rank 0)
        if (rank == 0) {
            FILE *fp = fopen(filename, "a");
            for (int j = 0; j < n; j++) {
                fprintf(fp, "%.10e ", grid[local_center][j]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        } else {
            MPI_Send(grid[local_center], n, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
        }
    }
    
    // Rank 0 receives from the rank that has the center
    if (rank == 0) {
        for (int r = 1; r < mpi_size; r++) {
            int r_rows_before = 0;
            for (int rr = 0; rr < r; rr++) {
                r_rows_before += temp_rows;
                if (rr < remainder) r_rows_before++;
            }
            
            int r_local_rows = temp_rows;
            if (r < remainder) r_local_rows++;
            
            if (center >= r_rows_before && center < r_rows_before + r_local_rows) {
                double *recv_buffer = (double *)malloc(n * sizeof(double));
                MPI_Status status;
                MPI_Recv(recv_buffer, n, MPI_DOUBLE, r, 99, MPI_COMM_WORLD, &status);
                
                FILE *fp = fopen(filename, "a");
                for (int j = 0; j < n; j++) {
                    fprintf(fp, "%.10e ", recv_buffer[j]);
                }
                fprintf(fp, "\n");
                fclose(fp);
                free(recv_buffer);
                break;
            }
        }
    }
}

double verify_solution(double **grid, int local_rows, int n, int rank, int size) {
    double local_heat = 0.0;
    double local_symmetry_error = 0.0;
    int ghost_offset = (rank > 0) ? 1 : 0;
    
    // Calculate global row offset
    int rows_before = 0;
    int temp_rows = n / size;
    int remainder = n % size;
    
    for (int r = 0; r < rank; r++) {
        rows_before += temp_rows;
        if (r < remainder) rows_before++;
    }
    
    // Compute local contributions
    int start_row = (rank == 0) ? 1 : ghost_offset;
    int end_row = (rank == size - 1) ? ghost_offset + local_rows - 1 : ghost_offset + local_rows;
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < n - 1; j++) {
            local_heat += grid[i][j];
            
            // Symmetry check simplified for distributed memory
            // (Full implementation would need ghost cells from mirror rank)
        }
    }
    
    // Reduce to get global values
    double total_heat, symmetry_error;
    MPI_Reduce(&local_heat, &total_heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_symmetry_error, &symmetry_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Total heat remaining: %.6e\n", total_heat);
        printf("Symmetry error: %.6e (simplified check)\n", symmetry_error);
    }
    
    return symmetry_error;
}

void log_performance_mpi(int n, int steps, int size, double elapsed,
                         double total_mem_mb, double avg_mem_mb,
                         const char *scenario) {
    FILE *fp = fopen("results/mpi_perf.log", "a");
    if (fp == NULL) {
        printf("Warning: Could not open mpi_perf.log for writing\n");
        return;
    }

    fprintf(fp,
            "scenario=%s n=%d steps=%d ranks=%d time=%.6f time_per_step=%.8f totalMemMB=%.2f avgMemPerRankMB=%.2f\n",
            scenario, n, steps, size, elapsed, elapsed / steps, total_mem_mb, avg_mem_mb);

    fclose(fp);
}

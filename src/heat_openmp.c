/*
 * OpenMP parallel implementation of 2D Heat Equation solver
 * Uses finite difference method with explicit time stepping
 * Parallelization: Data parallelism with loop-level parallelization
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

// Default parameters
#define DEFAULT_N 1000        // Grid size (N x N)
#define DEFAULT_STEPS 1000    // Number of time steps
#define ALPHA 0.1             // Thermal diffusivity

// Function prototypes
void initialize_grid(double **grid, int n);
void apply_boundary_conditions(double **grid, int n);
double compute_step_openmp(double **grid, double **grid_new, int n, double factor);
double** allocate_2d_array(int n);
void free_2d_array(double **array, int n);
double get_walltime();
void save_results(double **grid, int n, const char *filename);
double verify_solution(double **grid, int n, int steps);
void log_performance_openmp(int n, int steps, int num_threads, double elapsed, double memory_mb);

int main(int argc, char *argv[]) {
    int n = DEFAULT_N;
    int steps = DEFAULT_STEPS;
    int num_threads = omp_get_max_threads();
    
    // Parse command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);
    if (argc > 3) {
        num_threads = atoi(argv[3]);
        omp_set_num_threads(num_threads);
    }
    
    printf("=== OpenMP Heat Equation Solver ===\n");
    printf("Grid size: %d x %d\n", n, n);
    printf("Time steps: %d\n", steps);
    printf("Alpha: %.2f\n", ALPHA);
    printf("OpenMP threads: %d\n", num_threads);
    
    // Allocate grids
    double **grid = allocate_2d_array(n);
    double **grid_new = allocate_2d_array(n);
    
    // Initialize
    initialize_grid(grid, n);
    apply_boundary_conditions(grid, n);
    
    // Compute time step factor
    double dx = 1.0 / (n - 1);
    double dt = 0.25 * dx * dx / ALPHA;  // Stability condition
    double factor = ALPHA * dt / (dx * dx);
    
    printf("dx: %.6f, dt: %.6f, factor: %.6f\n", dx, dt, factor);
    
    // Main computation loop
    double start_time = get_walltime();
    double max_diff = 0.0;
    
    for (int step = 0; step < steps; step++) {
        max_diff = compute_step_openmp(grid, grid_new, n, factor);
        
        // Swap pointers
        double **temp = grid;
        grid = grid_new;
        grid_new = temp;
        
        if (step % 100 == 0) {
            printf("Step %d, max diff: %.6e\n", step, max_diff);
        }
    }
    
    double end_time = get_walltime();
    double elapsed = end_time - start_time;
    
    printf("\n=== Results ===\n");
    printf("Total time: %.4f seconds\n", elapsed);
    printf("Time per step: %.6f seconds\n", elapsed / steps);
    printf("Final max diff: %.6e\n", max_diff);

    double memory_bytes = 2.0 * n * n * sizeof(double);
    double memory_mb = memory_bytes / (1024.0 * 1024.0);
    printf("Approx memory used (2 grids shared across threads): %.2f MB\n", memory_mb);
    
    // Verification
    double error = verify_solution(grid, n, steps);
    printf("Verification error: %.6e\n", error);
    
    // Save results
    char filename[100];
    sprintf(filename, "results/openmp_result_%d.txt", num_threads);
    save_results(grid, n, filename);

    log_performance_openmp(n, steps, num_threads, elapsed, memory_mb);
    
    // Cleanup
    free_2d_array(grid, n);
    free_2d_array(grid_new, n);
    
    return 0;
}

void initialize_grid(double **grid, int n) {
    // Initialize to zero - parallelized
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            grid[i][j] = 0.0;
        }
    }
    
    // Set initial heat source in the center
    int center = n / 2;
    int radius = n / 10;
    
    #pragma omp parallel for collapse(2)
    for (int i = center - radius; i <= center + radius; i++) {
        for (int j = center - radius; j <= center + radius; j++) {
            if (i >= 0 && i < n && j >= 0 && j < n) {
                double dist = sqrt((i - center) * (i - center) + 
                                 (j - center) * (j - center));
                if (dist <= radius) {
                    grid[i][j] = 100.0 * (1.0 - dist / radius);
                }
            }
        }
    }
}

void apply_boundary_conditions(double **grid, int n) {
    // Fixed boundary conditions (Dirichlet): temperature = 0 at boundaries
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            grid[0][i] = 0.0;
            grid[n-1][i] = 0.0;
        }
        
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            grid[i][0] = 0.0;
            grid[i][n-1] = 0.0;
        }
    }
}

double compute_step_openmp(double **grid, double **grid_new, int n, double factor) {
    double max_diff = 0.0;
    
    // Interior points using 5-point stencil
    // Use reduction for max_diff to avoid race conditions
    #pragma omp parallel for collapse(2) reduction(max:max_diff) schedule(static)
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            grid_new[i][j] = grid[i][j] + factor * (
                grid[i+1][j] + grid[i-1][j] + 
                grid[i][j+1] + grid[i][j-1] - 
                4.0 * grid[i][j]
            );
            
            double diff = fabs(grid_new[i][j] - grid[i][j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    // Apply boundary conditions
    apply_boundary_conditions(grid_new, n);
    
    return max_diff;
}

double** allocate_2d_array(int n) {
    double **array = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        array[i] = (double *)malloc(n * sizeof(double));
    }
    return array;
}

void free_2d_array(double **array, int n) {
    for (int i = 0; i < n; i++) {
        free(array[i]);
    }
    free(array);
}

double get_walltime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void save_results(double **grid, int n, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Warning: Could not open file for writing\n");
        return;
    }
    
    // Save grid dimensions
    fprintf(fp, "%d\n", n);
    
    // Save center line for quick verification
    int center = n / 2;
    for (int j = 0; j < n; j++) {
        fprintf(fp, "%.10e ", grid[center][j]);
    }
    fprintf(fp, "\n");
    
    fclose(fp);
}

double verify_solution(double **grid, int n, int steps) {
    // Simple verification: check conservation and symmetry
    double total_heat = 0.0;
    double symmetry_error = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(+:total_heat) reduction(max:symmetry_error)
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            total_heat += grid[i][j];
            
            // Check symmetry (heat source was symmetric)
            int i_mirror = n - 1 - i;
            int j_mirror = n - 1 - j;
            double sym_diff = fabs(grid[i][j] - grid[i_mirror][j_mirror]);
            if (sym_diff > symmetry_error) {
                symmetry_error = sym_diff;
            }
        }
    }
    
    printf("Total heat remaining: %.6e\n", total_heat);
    printf("Symmetry error: %.6e\n", symmetry_error);
    
    return symmetry_error;
}

void log_performance_openmp(int n, int steps, int num_threads, double elapsed, double memory_mb) {
    FILE *fp = fopen("results/openmp_perf.log", "a");
    if (fp == NULL) {
        printf("Warning: Could not open openmp_perf.log for writing\n");
        return;
    }

    fprintf(fp, "n=%d steps=%d threads=%d time=%.6f time_per_step=%.8f memMB=%.2f\n",
            n, steps, num_threads, elapsed, elapsed / steps, memory_mb);

    fclose(fp);
}

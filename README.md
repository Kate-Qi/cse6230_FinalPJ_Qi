# 2D Heat Equation – MPI, OpenMP, and Hybrid (PACE ICE)

This repository contains a 2D heat equation solver implemented in four variants:

- `heat_serial`: baseline serial implementation
- `heat_openmp`: shared-memory implementation using OpenMP
- `heat_mpi`: distributed-memory implementation using MPI (row-wise domain decomposition)
- `heat_hybrid`: hybrid MPI + OpenMP implementation

The project is designed for performance and scaling studies (strong, weak, and thread-to-thread) on Georgia Tech's PACE ICE cluster.

---

## Repository Layout

- `final.md` – project description / assignment
- `run.sh` – convenience driver for running on a local machine (uses `--oversubscribe`)
- `heat_equation/`
  - `Makefile` – builds all four executables
  - `run.sh` – main driver script for scaling experiments on PACE (no oversubscribe)
  - `src/`
    - `heat_serial.c`
    - `heat_openmp.c`
    - `heat_mpi.c`
    - `heat_hybrid.c`
  - `results/`
    - `*.txt`, `*.log` – timing and verification outputs
    - `analysis.ipynb` – Jupyter notebook for post-processing and plotting

---

## Building on PACE ICE

On a PACE ICE login node:

```bash
module load gcc
module load openmpi

cd /path/to/this/repo/heat_equation
make clean
make        # or: make all
```

Notes:
- The `Makefile` assumes `gcc` and `mpicc` are provided by the loaded modules.
- Use `make test` for a quick small verification run (good sanity check before large jobs).
- `make clean` removes executables and result files in `heat_equation/results/`.

---

## Executable Interfaces

All programs solve the same 2D heat equation with explicit time stepping. Command-line arguments are:

- **Serial**
  ```bash
  ./heat_serial N STEPS
  ```
  - `N`: grid size (produces an `N x N` grid)
  - `STEPS`: number of time steps

- **OpenMP**
  ```bash
  ./heat_openmp N STEPS NUM_THREADS
  ```
  - `NUM_THREADS` sets the number of OpenMP threads (also respects `OMP_NUM_THREADS`).

- **MPI**
  ```bash
  mpirun -np P ./heat_mpi N STEPS SCENARIO
  ```
  - `P`: number of MPI ranks
  - `SCENARIO`: label used in output filenames (e.g., `strong_n500`, `weak`).

- **Hybrid MPI + OpenMP**
  ```bash
  mpirun -np P ./heat_hybrid N STEPS NUM_THREADS
  ```
  - `P`: MPI ranks
  - `NUM_THREADS`: OpenMP threads per rank (total parallel tasks ≈ `P * NUM_THREADS`).

Each solver writes results and performance logs into `heat_equation/results/`.

---

## Running Scaling Experiments on PACE ICE

The main driver for scaling studies on PACE ICE is `heat_equation/run.sh`. It:

- Rebuilds the code (`make clean && make all`)
- Ensures `results/` exists
- Runs:
  - Serial baseline
  - OpenMP strong scaling (vary threads)
  - MPI strong scaling (vary ranks, fixed problem size)
  - Hybrid strong scaling (combinations of ranks × threads)
  - MPI weak scaling (increasing problem size with rank count)
  - OpenMP thread-to-thread scaling

From a compute allocation on PACE ICE:

```bash
module load gcc
module load openmpi

cd #/path/to/this/repo/heat_equation
sbatch run.sh
```

This will generate `.log` and `.txt` files in `heat_equation/results/`.

> On PACE, run `heat_equation/run.sh` inside a compute allocation or batch job. The top-level `run.sh` in the project root is meant for oversubscribed local testing and is not recommended on shared cluster nodes.

---

## Example SLURM Batch Script for PACE ICE

Below is a template `pace_job.sh` you can adapt. Fill in the account, partition/queue, and time limits according to PACE policy and your allocation.

```bash
#!/usr/bin/env bash
#SBATCH -J heat_eq
#SBATCH -A <your-pace-account>
#SBATCH -q <your-queue>          # e.g., "pace-ice" or course queue
#SBATCH -N 1                      # number of nodes
#SBATCH -n 8                      # total MPI ranks (adjust as needed)
#SBATCH -t 00:20:00               # walltime
#SBATCH -o heat_eq.%j.out
#SBATCH -e heat_eq.%j.err

module purge
module load gcc
module load openmpi

cd "$SLURM_SUBMIT_DIR"/heat_equation

make all
bash run.sh
```

Submit with:

```bash
sbatch pace_job.sh
```

If you want more ranks or multiple nodes, adjust `-N` and `-n` according to your scaling plan and PACE documentation.

---

## Verifying Correctness

- `make test` runs small serial, OpenMP, MPI, and hybrid problems to confirm basic correctness and that the solution does not depend on the number of tasks.
- Each solver prints a "verification error" metric; this should remain small and comparable across variants.
- For custom runs, you can compare final verification errors and/or output fields between implementations to ensure correctness before timing large jobs.

---

## Results and Post-Processing

- Raw timing and verification data are written to `heat_equation/results/` (e.g., `serial_result.txt`, `openmp_result_*.txt`, `mpi_*_result_*.txt`, `hybrid_result_*x*.txt`, and `*.log`).
- The notebook `heat_equation/results/analysis.ipynb` can be used to:
  - Parse these files
  - Compute speedup and efficiency
  - Plot strong, weak, and thread-to-thread scaling results

Open the notebook either:

- On PACE (e.g., via a remote Jupyter session), or
- By copying `heat_equation/results/` to your local machine and running Jupyter locally.

---

## Local Testing (Non-PACE)

For quick, oversubscribed experiments on a personal machine (e.g., a laptop), you can use the top-level `run.sh`:

```bash
module load gcc
module load openmpi    # or use your local MPI installation

cd /path/to/this/repo
bash run.sh
```

This script is similar to `heat_equation/run.sh` but enables `mpirun --oversubscribe` for environments with fewer physical cores than MPI ranks. Avoid oversubscription on shared PACE nodes.

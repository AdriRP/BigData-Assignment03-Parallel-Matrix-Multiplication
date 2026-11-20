# Individual Assignment 3 — Parallel Matrix Multiplication

**Author:** Adrián Rodríguez Pérez · Big Data, Grado en Ciencia e Ingeniería de Datos (ULPGC), 2026-2027

Research question: *How does the number of threads affect the runtime of a shared-memory parallel version of the
blocked matrix multiplication of Assignment 2, and when does parallel execution help?* The optional vectorisation
extension is included: every parallel configuration is also measured with the Java compiler's automatic
vectorisation disabled.

This repository contains the report, the code, configuration, raw measurements, analysis scripts and figures of
the study, in Java (computation) and Python (automation and analysis). Run every command below from the folder
that contains `config.json` and `scripts/` (the repository root).

## Methods in brief

All methods compute C = A·B for square n × n matrices of 64-bit floating-point numbers (`double`), stored as one
Java array per row.

- **Sequential baseline (`seq`):** the blocked method of Assignment 2, unchanged. It runs the i-k-j loops over
  square tiles (blocks) of 256 × 256 elements, so that the parts of B and C in use stay in the CPU cache.
- **Parallel method (`par`, p threads):** the same blocked loops, with the n rows of C split into p contiguous
  bands of equal size (they differ by at most one row). Each band is one task, run by a fixed pool of p worker
  threads (`java.util.concurrent.ThreadPoolExecutor`) created before any timed run. Every thread reads A and B,
  which are never modified, and writes only the rows of its own band, so no two threads write the same memory
  location and no locks are needed.
- **Auto-vectorisation:** the Java just-in-time (JIT) compiler, which compiles the code while the program runs,
  can compile the innermost loop so that each instruction processes several numbers. It is on by default; the
  variant `novec` disables it with the JVM option `-XX:-UseSuperWord`.
- **Thread placement:** `pinned` restricts each process (CPU affinity) to p logical processors, one per physical
  core first (see [Thread placement](#thread-placement)); `default` lets the Windows scheduler place the threads.

Every method adds the products of each entry in the same order, so all configurations give bit-identical results.

## Reused material

The sequential baseline is `DenseMethods.blocked` of Assignment 2
(https://github.com/AdriRP/BigData-Assignment02-Optimised-and-Sparse-Matrix-Multiplication), unchanged, with the same
block size (256). The input matrices use the same generator and seed as Assignments 1 and 2; for the five sizes of
this study they are byte-identical to the files of Assignment 2 (same SHA-256 hash, a fingerprint of the file
contents).

## Contents

| Path | Content |
|---|---|
| [`Assignment03_Adrian_Rodriguez_Report.pdf`](Assignment03_Adrian_Rodriguez_Report.pdf) | The report |
| `README.md` | This file |
| `src/java/DenseMethods.java` | Sequential baseline `blocked`, the band kernel `blockedRows`, and `ijk` (reference in the self-test) |
| `src/java/ParallelBlocked.java` | Parallel method: thread pool, partitioning into row bands |
| `src/java/Bench.java` | Benchmark harness (modes in [Build and run](#build-and-run)) and self-test |
| `config.json` | Sizes, thread counts, batch sizes, seed, warm-ups, repetitions, limits, block size, JVM options, placements |
| `requirements.txt` | Python dependencies with exact versions |
| `scripts/common.py` | Shared paths, configuration, Java command and the CPU sets of the `pinned` placement |
| `scripts/build.py` | Compiles the Java sources into `build/java/` |
| `scripts/gen_inputs.py` | Generates the input matrices in `data/inputs/` (see [Inputs](#inputs)) |
| `scripts/env_info.py` | Records hardware, topology (cores and caches) and software versions |
| `scripts/run_benchmarks.py` | Runs every configuration and measures time and memory |
| `scripts/validate.py` | Correctness checks (see [Results files](#results-files)) |
| `scripts/analyze.py` | Summary tables and the figures in `figures/` |
| `scripts/run_all.py` | Runs build, inputs, environment, benchmarks, validation and analysis in order |
| `diagnostics/` | Scripts and recorded output of the diagnostics described in the report (Section 4.5) |
| `results/` | Raw and summarised measurements (see [Results files](#results-files)) |
| `figures/` | Figures of the report |

`build/` (compiled classes) and `data/` are not included; the scripts regenerate them. `data/` holds the input
matrices and the outputs of the sequential baseline (511 MB in total).

## Requirements (versions used)

- **OS:** Windows 11. The memory metric, the peak working set (the largest amount of physical RAM the process
  occupied), is read with psutil's `peak_wset`, and the topology with the Windows function
  `GetLogicalProcessorInformationEx`; both exist only on Windows. Keep the files in a reasonably short path:
  Windows' 260-character path limit can stop Python from finding `scripts/common.py` in very deep folders.
- **Java:** OpenJDK 21.0.10 (Eclipse Temurin), `java` and `javac` on `PATH`.
- **Python:** CPython 3.12.0 and the packages in `requirements.txt` (NumPy 2.3.5, pandas 3.0.1, matplotlib
  3.10.8, psutil 7.2.2). NumPy only generates the inputs and validates results; it never runs inside a timed region.

## Build and run

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

The benchmarks took about 50 minutes on the machine described in the report (about 25 per placement). The same
steps, one by one:

```bash
python scripts/build.py
python scripts/gen_inputs.py
python scripts/env_info.py
python scripts/run_benchmarks.py
python scripts/validate.py
python scripts/analyze.py
```

`python scripts/run_benchmarks.py --placements pinned` runs only the placements listed and keeps the rows of the
others. `env_info.py` must run before `run_benchmarks.py`, which reads the core topology from
`results/environment.json`.

Each configuration runs in its own Java process. Exact commands used by the scripts (`config.json` holds the
parameters):

```bash
javac --release 21 -d build/java src/java/*.java
java -Xmx4g -cp build/java Bench seq <n> <A.bin> <B.bin> <blockSize> <warmups> <reps> <batch> <out.bin|-> --hold
java -Xmx4g -cp build/java Bench par <threads> <n> <A.bin> <B.bin> <blockSize> <warmups> <reps> <batch> - --hold
java -Xmx4g -XX:-UseSuperWord -cp build/java Bench ...   # the novec variant
java -cp build/java Bench --selftest
```

`-Xmx4g` limits the Java heap to 4 GB. `--hold` makes the process wait, after its last run, until the driver has
read its peak working set (the value can only be read while the process is alive). `<batch>` is the number of
multiplications in each timed sample: 50 at n = 200 and 5 at n = 500, whose single products are too short to time
reliably, and 1 otherwise; all reported times are per multiplication.

Before measuring, each process calls its kernel 200 times on a 128 × 128 matrix on the main thread and, for the
parallel method, 200 more times through the pool (pre-warm), so that the JIT compiler has fully compiled it; then
it performs 3 warm-up samples, which are recorded but not used in the statistics, and 5 measured samples. A sample
that takes more than 60 s is stopped and recorded as a timeout (none occurred).

## Thread placement

The machine has 16 physical cores, each with two logical processors (simultaneous multithreading, SMT): physical
core k contains logical processors 2k and 2k + 1, and cores 0–7 and 8–15 each share one 32 MB L3 cache
(`results/environment.json`, field `topology`). With the `pinned` placement the driver restricts each process to:

| Threads p | Logical processors allowed |
|---|---|
| p ≤ 16 (and `seq`, as p = 1) | 0, 2, …, 2(p − 1): one logical processor on each of the first p physical cores |
| 16 < p < 32 | all 16 even ones, plus the odd ones 1, 3, … (second logical processor of the first p − 16 cores) |
| p ≥ 32 | all 32 |

With `default`, no affinity is set. Both placements run at normal priority.

## Inputs

The input matrices are not included because of their size and are regenerated with:

```bash
python scripts/gen_inputs.py
```

which writes `data/inputs/A_<n>.bin` and `data/inputs/B_<n>.bin` for every size in `config.json`. For each n, A
and B are drawn from NumPy's PCG64 generator seeded with `[2026, n]`, uniform in [−1, 1), and stored as raw
little-endian row-major `float64` (8n² bytes per file, no header). `results/inputs_manifest.csv` holds the SHA-256
hash of every input file used in the study. When it is present, `gen_inputs.py` does not overwrite it: it compares
the regenerated files with it and prints, for each size, whether they are identical.

## Results files

| File | Content |
|---|---|
| `results/raw_times.csv` | One row per timed sample: `placement, method, threads, variant, n, dtype, batch, kind (warmup or measured), rep, time_ns, time_per_mult_s, checksum`. The checksum is the sum of all entries of the result, printed so the computation cannot be skipped by the compiler |
| `results/runs.csv` | One row per process: placement, method, threads, variant, n, oversubscription (more threads than logical processors), allowed logical processors, status, peak working set, dense storage of A, B and C, live JVM threads, SHA-256 of the result, wall time |
| `results/benchmark_log_pinned.txt`, `results/benchmark_log_default.txt` | Console logs of the two benchmark runs |
| `results/summary.csv`, `results/summary.md` | Median, quartiles, interquartile range (IQR, the spread of the middle half of the samples), minimum, maximum and peak memory of every configuration; tables of time, speedup and efficiency |
| `results/speedup.csv` | Speedup S(p) = T_seq / T_p and efficiency E(p) = S(p) / p from the medians, for every configuration |
| `results/validation.csv` | Result of every correctness check: Java self-test (with and without vectorisation); sequential results against NumPy's matrix product within the rounding-error tolerance of Assignment 1; every configuration bit-identical to the sequential baseline; identical checksums across samples |
| `results/environment.json` | CPU, cores, topology, caches, RAM, OS, tool versions and JVM flags |
| `results/inputs_manifest.csv` | Generator, seed and SHA-256 hash of every input file |

The `default` rows were measured first, by the same harness and commands before the driver gained the
`placement` and `affinity` columns; those two columns were then added to them (`default`, `all (not set)`), with no
other change.

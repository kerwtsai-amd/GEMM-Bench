# ROCm GEMM benchmarking container

An Apptainer image that adds the rocBLAS and hipBLASLt clients —
`rocblas-bench` and `hipblaslt-bench` — to the stock ROCm development image,
plus a script that sweeps GEMM across precisions and reports the clock and
power the GPU actually settles at rather than its nominal boost clock.

The AMD ROCm packages ship `librocblas.so` and `libhipblaslt.so` but not the
benchmark and test executables, so the clients have to be compiled from source.
This image builds them against the libraries already present in the base image,
pinned to the exact upstream commit those libraries were built from, so the two
are ABI-compatible.

## Contents

| File | Purpose |
| --- | --- |
| `apptainer.def` | Container definition: ROCm 7.14 + rocBLAS and hipBLASLt clients |
| `bench_precisions.py` | Precision sweep with clock/power sampling |
| `bench_config.json` | Which precisions and backends each chip runs |
| `bench_precisions.sh` | The original single-backend sweep, kept for reference |

`*.sif` images and `*.csv` results are gitignored; they are large and
machine-specific.

## Prerequisites

- Apptainer 1.4+ with `--fakeroot` available
- The base image, pulled once:

```bash
apptainer pull base.sif docker://rocm/dev-ubuntu-24.04:7.14.0-full
```

`apptainer.def` bootstraps from that local `base.sif` so repeated builds do not
re-download ~8 GB. To make the definition self-contained instead, swap the
`BootStrap`/`From` lines for the `docker://` variant commented at the top of
the file.

## Build

```bash
apptainer build --fakeroot rocblas.sif apptainer.def
```

Takes about 11 minutes on a 188-core host and produces a ~7.7 GB image. Only
the clients are compiled — neither library is rebuilt, and hipBLASLt is built
with `-n` so its Tensile kernels are not regenerated either. That is what keeps
the build short. `rocblas-test` is off by default because it adds hours; set
`BUILD_TESTS=ON` in `%post` if you need it. `hipblaslt-test` is cheap and is
always built.

rocBLAS device code is generated for `gfx908`, `gfx90a`, `gfx942` and `gfx950`
(MI100, MI200, MI300X, MI350X). hipBLASLt only supports CDNA2 and newer, so its
clients are built for `gfx942` and `gfx950` only.

## Running

The image carries a complete ROCm stack, so `--rocm` is unnecessary and only
causes host libraries to shadow the container's:

```bash
apptainer exec rocblas.sif rocblas-bench --version
```

On HPCFund-TW the login node has an MI100. Use Slurm to reach the datacenter
parts — `mi3001x` for a single MI300X, `mi3501x` for a single MI350X:

```bash
srun -N1 -n1 -p mi3001x --gres=gpu:1 -t 00:15:00 \
  apptainer exec rocblas.sif \
  rocblas-bench -f gemm -r f64_r \
    --transposeA T --transposeB N -m 8192 -n 8192 -k 8192 \
    --alpha 1 --beta 0 -i 300 -j 20
```

`-r` selects the precision (`f64_r`, `f32_r`, `f16_r`, `bf16_r`, `i8_r`, and
the complex variants). `-i` is the number of timed iterations and `-j` the
warm-up iterations before timing starts; the defaults of 10 and 2 are too few
to reach a steady state. Add `-v 1` to norm-check the result against CPU BLAS,
but only at small sizes — it is very slow.

`apptainer run` is wired to `rocblas-bench`, so the `exec rocblas.sif
rocblas-bench` prefix can be shortened to `run rocblas.sif`.

### hipblaslt-bench

`rocblas-bench` has no FP8 type, so FP8 has to go through hipBLASLt. Note the
different flag spellings — `--transA`/`--transB` rather than
`--transposeA`/`--transposeB`:

```bash
srun -N1 -n1 -p mi3001x --gres=gpu:1 -t 00:15:00 \
  apptainer exec rocblas.sif \
  hipblaslt-bench --transA T --transB N -m 8192 -n 8192 -k 8192 \
    --a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r -i 50 -j 10
```

Use `--a_type`/`--b_type` for FP8, not `--compute_input_typeA`/`B`. The latter
keeps the operands in f16 and downconverts every iteration, which measures the
conversion rather than the GEMM — about 30 TFLOPS instead of 1200 on MI300X.

`f8_r` is the portable spelling: hipBLASLt maps it to the `f8_fnuz_r` format on
gfx942 and to OCP FP8 on gfx950. Asking for `f8_r` through
`--compute_input_type` on gfx942 instead fails with `NO solution found`.

hipBLASLt and rocBLAS results are not directly comparable even at the same
precision — hipBLASLt reaches roughly 1.5x rocBLAS on f16 — so the sweep script
labels which library produced each row.

## Precision sweep

`bench_precisions.py` runs a sustained GEMM per precision while sampling clock
and power, then prints a summary table. Where a data type exists in both
libraries it is measured on both, one row each, so the two are directly
comparable:

```bash
srun -N1 -n1 -p mi3501x --gres=gpu:1 -t 00:40:00 \
  apptainer exec --bind "$PWD":/work rocblas.sif \
  python3 /work/bench_precisions.py -o /work/mi350x.csv
```

Only the standard library is used, so the container's own `python3` is enough,
and `bench_config.json` is read from beside the script, so binding the
repository in is all the setup there is. The recorded MI300X sweep in
`mi300x.csv` renders like this — it predates the two-backend rows, so each type
appears once:

```
┌───────────┬──────────────┬───────────────┬──────────────┬─────────────┬────────┐
│ Precision │ BLAS backend │    Throughput │ Steady Clock │ % Max Clock │  Power │
├───────────┼──────────────┼───────────────┼──────────────┼─────────────┼────────┤
│ f64_r     │ rocBLAS      │   55.4 TFLOPS │     1795 MHz │         82% │  983 W │
│ f32_r     │ rocBLAS      │  115.7 TFLOPS │     1826 MHz │         83% │  988 W │
│ f32_r     │ hipBLASLt    │  130.7 TFLOPS │     2095 MHz │         95% │  969 W │
│ tf32      │ hipBLASLt    │  392.9 TFLOPS │     1318 MHz │         60% │ 1000 W │
│ f16_r     │ rocBLAS      │ 1082.5 TFLOPS │     1129 MHz │         51% │ 1000 W │
│ f16_r     │ hipBLASLt    │ 1081.0 TFLOPS │     1126 MHz │         51% │ 1000 W │
│ bf16_r    │ rocBLAS      │ 1149.1 TFLOPS │     1256 MHz │         57% │ 1000 W │
│ bf16_r    │ hipBLASLt    │ 1146.0 TFLOPS │     1252 MHz │         57% │ 1000 W │
│ i8_r      │ rocBLAS      │   3087.7 TOPS │     2159 MHz │         98% │  970 W │
│ i8_r      │ hipBLASLt    │   2304.1 TOPS │     1465 MHz │         67% │ 1000 W │
│ f8_r      │ hipBLASLt    │ 2100.3 TFLOPS │     1501 MHz │         68% │  999 W │
└───────────┴──────────────┴───────────────┴──────────────┴─────────────┴────────┘
```

Progress goes to stderr and the table to stdout, so redirecting stdout leaves
only the results. Under a C locale the box characters fall back to ASCII.

| Option | Meaning |
| --- | --- |
| `-s SIZE` | Square GEMM dimension (default 16384) |
| `-t SECONDS` | Target duration per case (default 25) |
| `-p LIST` | Comma-separated precision subset, e.g. `-p f64_r,f16_r` |
| `-b LIST` | Comma-separated backend subset, e.g. `-b hipblaslt` |
| `-o FILE` | Also write results as CSV |
| `-f FORMAT` | Stdout format: `table`, `markdown` or `csv` |
| `-c FILE` | Configuration file (default `bench_config.json` beside the script) |
| `--chip NAME` | Use a named chip profile instead of the detected one |
| `--list-chips` | Print the profiles in the config and exit |

Iteration counts are calibrated per case from a short trial run, so every
measurement covers the same wall-clock duration regardless of how fast the
precision is. A sample counts toward the steady state only once power exceeds
1.5x idle, and the first qualifying sample is discarded because power reaches
the cap before clocks finish settling. A combination the GPU or library does not
support produces no result at all and is reported as `unsupported` rather than
aborting the sweep.

### Configuration

Which precisions run, and on which backends, is decided entirely by
`bench_config.json`; adding a data type or a chip needs no change to the script.

| Section | Contents |
| --- | --- |
| `defaults` | Matrix size, target seconds per case, sampling interval, iteration floors, output format |
| `backends` | Per-library binary name, the common arguments, and which CSV columns carry throughput and time |
| `precisions` | Per data type: the unit (`TFLOPS` or `TOPS`) and one argument template per backend |
| `chips` | Per architecture: the strings that identify it, and an ordered map of precision to backend list |

The chip profile is picked by matching each profile's `match` strings against
the GPU name `rocblas-bench` reports, falling back to `default`. `--chip`
overrides the choice, which is how you can dry-run an MI300X profile from an
MI100 login node and watch every hipBLASLt row degrade to `unsupported`.

To add a data type, add an entry under `precisions` with an argument template
for each backend that can run it, then list it under the chips that should
measure it. To add a chip, copy a profile, change `match` to something that
appears in its device name, and prune the `run` map. A precision listed for a
backend that has no template for it is a configuration error and is reported as
one before any benchmarking starts.

The two shipped profiles differ in one place: `gfx942` measures TF32 on both
libraries, while `gfx950` measures it on hipBLASLt only, because rocBLAS's
`--math_mode 1` path returns no result there.

## How the libraries relate

```mermaid
graph TD
    app["Your code"]

    hipblas["hipBLAS portable BLAS API"]
    hipblaslt["hipBLASLt — GEMM only, fused epilogues only here: fp8, fp4 shared: fp16, bf16, fp32, tf32, int8"]
    rocblas["rocBLAS — full BLAS only here: fp64, complex shared: fp32, tf32, fp16, bf16, int8"]
    cublas["cuBLAS"]
    cublaslt["cuBLASLt"]

    tensile["Tensile kernel generator + autotuner"]
    tensilelite["TensileLite hard fork of Tensile"]

    rocblaslib["librocblas.so + rocblas/library/*.dat"]
    hipbltlib["libhipblaslt.so + hipblaslt/library/*.dat"]

    app --> hipblas
    app --> hipblaslt
    app --> rocblas

    hipblas -->|AMD| rocblas
    hipblas -.->|NVIDIA| cublas
    hipblaslt -.->|NVIDIA| cublaslt
    rocblas -.->|"some GEMMs, at run time"| hipblaslt

    rocblas -->|"GEMM, at build time"| tensile
    hipblaslt -->|"GEMM, at build time"| tensilelite
    tensile -.->|forked| tensilelite

    tensile --> rocblaslib
    tensilelite --> hipbltlib
```

**hipBLAS is not an implementation.** It is a marshalling layer that exposes the
classic netlib BLAS surface and forwards each call to rocBLAS on AMD or cuBLAS
on NVIDIA. No kernel of its own ever reaches the GPU, so it shows up in a
performance discussion only as dispatch overhead.

**rocBLAS implements everything except GEMM itself.** Levels 1 and 2 are
hand-written HIP in the library. GEMM is far too sensitive to shape and
architecture for that, so those kernels come from Tensile — a Python generator
and autotuner that emits CDNA assembly at build time along with per-architecture
logic files that pick a solution at run time from the problem dimensions.
Generating and compiling that kernel set is why a full rocBLAS build takes
hours, and why this image compiles only the clients and reuses the prebuilt
`librocblas.so`.

**hipBLASLt is the counterpart of cuBLASLt.** GEMM only, but it exposes what the
classic interface cannot express: fused epilogues (bias, activation, scaling),
narrow types such as FP8, and explicit algorithm enumeration behind a reusable
plan. Its kernels come from TensileLite, a hard fork of Tensile that lives
inside the hipBLASLt tree under `tensilelite/`. The two generators share an
ancestry and a vocabulary but have diverged; a kernel tuned in one does not
exist in the other.

**The dashed rocBLAS → hipBLASLt edge is real at run time.** Recent rocBLAS
carries both backends and chooses per problem and architecture, falling back to
Tensile whenever hipBLASLt has no solution. `ROCBLAS_USE_HIPBLASLT=0` pins
Tensile and `=1` prefers hipBLASLt, which is worth knowing when a
`rocblas-bench` number is surprising and you want to establish which backend
produced it.

All of these now live in the `ROCm/rocm-libraries` monorepo — `projects/rocblas`,
`projects/hipblas`, `projects/hipblaslt` and `shared/tensile`, with TensileLite
under `projects/hipblaslt/tensilelite`. The standalone repositories are
deprecated, which is why `apptainer.def` sparse-checks-out the monorepo rather
than cloning rocBLAS directly.

### Which data type lives where

| Data type | rocBLAS / Tensile | hipBLASLt / TensileLite | In the sweep |
| --- | --- | --- | --- |
| `f64_r`, `f32_c`, `f64_c` | yes | no | `f64_r` only |
| `f32_r` | yes | yes | yes |
| TF32 compute on FP32 data | `--math_mode 1`, gfx942 only | `HIPBLAS_COMPUTE_32F_FAST_TF32`, native on gfx942, emulated on gfx950 | yes |
| `f16_r`, `bf16_r` | yes | yes, and where new tuning lands | yes |
| `i8_r` → `i32_r` | yes | yes | yes |
| FP8 `e4m3` / `e5m2` | no | yes — `fnuz` on gfx942, OCP on gfx950 | yes, via `hipblaslt-bench` |
| FP4 `e2m1` | no | input only, gfx950 | no |

The two ends of that table are exclusive, and that is the cleanest way to
remember the split. FP64 and complex belong to rocBLAS alone — the HPL and
scientific-computing side, which TensileLite never targeted. FP8 and FP4 belong
to hipBLASLt alone; rocBLAS's documented type list stops at int8. (hipBLASLt's
`hipDataType` enum does expose complex types and a `HIPBLAS_COMPUTE_64F` mode,
but its ROCm 7.14 support overview lists only int8, fp8, fp4, fp16, bf16 and
fp32, so treat FP64 and complex as rocBLAS territory in practice.)

The middle rows are where the choice actually matters, and the catch is that it
may not be yours to make: `f16_r` and `bf16_r` exist in both, hipBLASLt is
usually where the newer kernels and tuning land, and rocBLAS's auto-selection
may already be routing those calls to hipBLASLt. The FP16 and BF16 numbers in
`mi300x.csv` and `mi350x.csv` were collected without pinning a backend, so some
of them are plausibly hipBLASLt kernels measured through the rocBLAS API. The
sweep now benchmarks both libraries directly, which shows the gap but not which
kernel served the rocBLAS row; to settle that, pin the backend and compare — if
the two agree, Tensile served both:

```bash
ROCBLAS_USE_HIPBLASLT=0 python3 /work/bench_precisions.py -b rocblas -p f16_r,bf16_r -o /work/tensile.csv
ROCBLAS_USE_HIPBLASLT=1 python3 /work/bench_precisions.py -b rocblas -p f16_r,bf16_r -o /work/hipblaslt.csv
```

## Notes

**Host environment leaks into the container.** A `module load rocm` on the host
exports `HIP_PATH=/opt/rocm-7.2.0`, `CPATH` and friends. That path does not
exist inside the image, so `amdclang` fails with `cannot find HIP runtime` on
every HIP source file. `%post` unsets these before building and `%environment`
re-asserts `ROCM_PATH`/`HIP_PATH` so the runtime is protected too. If you hit
something similar with another container, `--cleanenv` is the quick diagnostic.

**FP8 needs hipblaslt-bench.** `rocblas-bench` in this version supports
precisions only up to `i8_r`, and the base image ships `libhipblaslt.so`
without its clients — hence the second build in `%post`. See the
`hipblaslt-bench` section above for the invocation that measures the GEMM
rather than an f16-to-f8 conversion.

**TF32 is MI300X-only through rocBLAS.** The `--math_mode 1` path returns no
result on gfx950 and leaves the GPU idle, consistent with the MI350X datasheet
not listing a TF32 rate while the MI300X one specifies 653.7 TFLOPS. The
capability is not absent from the part, though: hipBLASLt documents
`HIPBLAS_COMPUTE_32F_FAST_TF32` as native on gfx942 and *emulated* on gfx950,
which is why the `gfx950` profile routes TF32 to `hipblaslt-bench` with
`--compute_type xf32_r`. Any MI350X TF32 number is measuring emulation rather
than hardware.

**Larger is not always faster.** Both parts lose throughput going from 8192³ to
16384³ (MI300X 79.3 → 64.0 TFLOPS in FP64), so quote the size alongside any
number.

**rocblas-bench is single-GPU.** To load a whole node, request
`-p mi3008x --gres=gpu:8 --exclusive` and launch one process per GPU with
`--device 0` through `--device 7`.

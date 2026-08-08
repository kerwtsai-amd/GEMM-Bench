# GEMM benchmarking container

An Apptainer image that adds the rocBLAS clients — most importantly
`rocblas-bench` — to the stock ROCm development image, plus a script that
sweeps GEMM across precisions and reports the clock and power the GPU actually
settles at rather than its nominal boost clock.

The AMD ROCm packages ship `librocblas.so` but not the benchmark and test
executables, so the clients have to be compiled from source. This image builds
them against the rocBLAS library already present in the base image, pinned to
the exact upstream commit that library was built from, so the two are
ABI-compatible.

## Contents

| File | Purpose |
| --- | --- |
| `apptainer.def` | Container definition: ROCm 7.14 + rocBLAS clients |
| `bench_precisions.sh` | Precision sweep with clock/power sampling |

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

Takes about 3.5 minutes on a 188-core host and produces a ~7.6 GB image. Only
the clients are compiled; the rocBLAS library itself is not rebuilt, which is
what keeps the build short. `rocblas-test` is off by default because it adds
hours — set `BUILD_TESTS=ON` in `%post` if you need it.

Device code is generated for `gfx908`, `gfx90a`, `gfx942` and `gfx950`, i.e.
MI100, MI200, MI300X and MI350X.

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

## Precision sweep

`bench_precisions.sh` runs a sustained GEMM per precision while sampling clock
and power, then prints a summary table:

```bash
srun -N1 -n1 -p mi3501x --gres=gpu:1 -t 00:40:00 \
  apptainer exec --bind "$PWD":/work rocblas.sif \
  bash /work/bench_precisions.sh -o /work/mi350x.csv
```

| Option | Meaning |
| --- | --- |
| `-s SIZE` | Square GEMM dimension (default 16384) |
| `-t SECONDS` | Target duration per precision (default 25) |
| `-p LIST` | Comma-separated subset, e.g. `-p f64_r,f16_r` |
| `-o FILE` | Also write results as CSV |

Iteration counts are calibrated per precision from a short trial run, so every
measurement covers the same wall-clock duration regardless of how fast the
precision is. A sample counts toward the steady state only once power exceeds
1.5x idle, and the first qualifying sample is discarded because power reaches
the cap before clocks finish settling. A precision the GPU does not support
produces no samples at all and is reported as `unsupported`.

## Measured results

16384³ GEMM, `T,N` layout, 25 s sustained per precision.

**MI300X** (gfx942, 2100 MHz max SCLK, 750 W TBP):

| Precision | Throughput | Steady clock | % of max clock | Power |
| --- | --- | --- | --- | --- |
| f64_r | 64.1 TFLOPS | 931 MHz | 44% | 750 W |
| f32_r | 94.8 TFLOPS | 1270 MHz | 60% | 750 W |
| tf32 | 262.8 TFLOPS | 1013 MHz | 48% | 751 W |
| f16_r | 427.2 TFLOPS | 1513 MHz | 72% | 750 W |
| bf16_r | 443.5 TFLOPS | 1543 MHz | 73% | 750 W |
| i8_r | 1223.4 TOPS | 1644 MHz | 78% | 750 W |

**MI350X** (gfx950, 2200 MHz max SCLK, 1000 W TBP):

| Precision | Throughput | Steady clock | % of max clock | Power |
| --- | --- | --- | --- | --- |
| f64_r | 55.5 TFLOPS | 1800 MHz | 82% | 982 W |
| f32_r | 116.2 TFLOPS | 1830 MHz | 83% | 990 W |
| tf32 | unsupported | - | - | - |
| f16_r | 1083.1 TFLOPS | 1132 MHz | 51% | 1000 W |
| bf16_r | 1148.5 TFLOPS | 1257 MHz | 57% | 1000 W |
| i8_r | 3117.1 TOPS | 2163 MHz | 98% | 975 W |

Every precision on both parts pins at the power limit, and none sustain the
nominal boost clock. Which precision suffers most is inverted between the two
generations: on MI300X, FP64 collapses to 44% of max clock while low precision
holds 72-78%; on MI350X, FP64 barely drops to 82% while FP16/BF16 fall to
51-57%. This tracks CDNA4 halving the FP64 matrix rate (163.4 → 72.1 TFLOPS
peak) and doubling FP16, which moves the power bottleneck from FP64 to the
low-precision datapaths. The practical consequence is that peak clock cannot be
used to predict achievable throughput for any precision.

The power-limited behaviour was verified directly: on MI300X the same kernel
runs at 2093 MHz while power is still ramping through 176 W, then drops to
~1137 MHz the moment power pins at 750 W, with junction temperature only 73 °C
— far from any thermal limit.

## Notes

**Host environment leaks into the container.** A `module load rocm` on the host
exports `HIP_PATH=/opt/rocm-7.2.0`, `CPATH` and friends. That path does not
exist inside the image, so `amdclang` fails with `cannot find HIP runtime` on
every HIP source file. `%post` unsets these before building and `%environment`
re-asserts `ROCM_PATH`/`HIP_PATH` so the runtime is protected too. If you hit
something similar with another container, `--cleanenv` is the quick diagnostic.

**FP8 is not covered.** `rocblas-bench` in this version supports precisions only
up to `i8_r`; FP8 GEMM goes through hipBLASLt, and the base image ships
`libhipblaslt.so` without `hipblaslt-bench`. Building the hipBLASLt clients the
same way rocBLAS's are built would add it.

**TF32 is MI300X-only.** The `--math_mode 1` path returns no result on gfx950
and leaves the GPU idle, consistent with the MI350X datasheet not listing a
TF32 rate while the MI300X one specifies 653.7 TFLOPS.

**Larger is not always faster.** Both parts lose throughput going from 8192³ to
16384³ (MI300X 79.3 → 64.0 TFLOPS in FP64), so quote the size alongside any
number.

**rocblas-bench is single-GPU.** To load a whole node, request
`-p mi3008x --gres=gpu:8 --exclusive` and launch one process per GPU with
`--device 0` through `--device 7`.

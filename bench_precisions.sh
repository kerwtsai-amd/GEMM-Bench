#!/bin/bash
# Benchmark rocBLAS GEMM across precisions and report sustained throughput
# alongside the clock and power the GPU actually settles at.
#
# Run inside the container, on a compute node:
#   srun -N1 -n1 -p mi3501x --gres=gpu:1 -t 00:40:00 \
#     apptainer exec --bind $PWD:/work rocblas.sif bash /work/bench_precisions.sh
#
# Options:
#   -s SIZE     square GEMM dimension            (default 16384)
#   -t SECONDS  target duration per precision    (default 25)
#   -o FILE     also write results as CSV
#   -p LIST     comma-separated subset, e.g. -p f64_r,f16_r

set -uo pipefail

SIZE=16384
TARGET=25
CSV=""
ONLY=""

while getopts "s:t:o:p:h" opt; do
    case $opt in
        s) SIZE=$OPTARG ;;
        t) TARGET=$OPTARG ;;
        o) CSV=$OPTARG ;;
        p) ONLY=$OPTARG ;;
        h) sed -n '2,13p' "$0"; exit 0 ;;
        *) exit 1 ;;
    esac
done

command -v rocblas-bench >/dev/null || {
    echo "rocblas-bench not found; run this inside the container" >&2; exit 1; }

COMMON="--transposeA T --transposeB N -m $SIZE -n $SIZE -k $SIZE --alpha 1 --beta 0"

# name|unit|args. gemm_ex is used wherever the input and compute types differ.
PRECS=(
"f64_r|TFLOPS|-f gemm -r f64_r"
"f32_r|TFLOPS|-f gemm -r f32_r"
"tf32|TFLOPS|-f gemm_ex --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --math_mode 1"
"f16_r|TFLOPS|-f gemm_ex --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r"
"bf16_r|TFLOPS|-f gemm_ex --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r"
"i8_r|TOPS|-f gemm_ex --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r"
)

sample() {
    rocm-smi --showpower --showgpuclocks 2>/dev/null | awk '
        /sclk/          { n=$0; sub(/.*\(/,"",n); sub(/Mhz\).*/,"",n); c=n }
        /Package Power/ { p=$NF }
        END             { if (c != "" && p != "") print c, p }'
}

median() { sort -n | awk '{v[NR]=$1} END{ if(!NR) exit; m=int((NR+1)/2);
    if (NR%2) printf "%.0f", v[m]; else printf "%.0f", (v[m]+v[m+1])/2 }'; }

# Per-iteration microseconds from a rocblas-bench result line, or empty if the
# call failed (an unsupported precision prints nothing).
iter_us() { awk -F, 'NF>3 {gsub(/ /,"",$NF); u=$NF} END{ if (u+0>0) print u }'; }

info=$(rocblas-bench -f gemm -r f32_r -m 64 -n 64 -k 64 -i 1 2>&1)
GPU=$(echo "$info" | sed -n 's/^Device ID [0-9]* : //p' | head -1)
MAXCLK=$(echo "$info" | sed -n 's/.*max\. SCLK \([0-9]*\) MHz.*/\1/p' | head -1)
[ -n "$MAXCLK" ] || MAXCLK=$(amd-smi metric -g 0 2>/dev/null |
    awk '/MAX_CLK/{gsub(/[^0-9]/,"",$2); print $2; exit}')

IDLE_PWR=$(sample | awk '{print $2}')
IDLE_PWR=${IDLE_PWR:-0}
# A sample counts as loaded once power clears 1.5x idle; this drops the ramp-up
# and, for an unsupported precision, leaves nothing at all.
THRESH=$(awk -v p="$IDLE_PWR" 'BEGIN{printf "%.0f", p*1.5}')

printf 'GPU: %s\nMax SCLK: %s MHz   Idle power: %s W   Size: %d^3   Target: %ss/precision\n\n' \
    "$GPU" "$MAXCLK" "$IDLE_PWR" "$SIZE" "$TARGET" >&2

ROWS=()
for entry in "${PRECS[@]}"; do
    IFS='|' read -r name unit args <<< "$entry"
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi

    printf 'calibrating %-7s ... ' "$name" >&2
    us=$(rocblas-bench $args $COMMON -i 10 -j 2 2>/dev/null | iter_us)
    if [ -z "$us" ]; then
        echo "unsupported on this GPU" >&2
        ROWS+=("$name|unsupported|-|-|-")
        continue
    fi
    iters=$(awk -v us="$us" -v t="$TARGET" 'BEGIN{n=int(t*1e6/us); print (n<20?20:n)}')
    printf '%s us/iter -> %s iters, running ... ' "$us" "$iters" >&2

    tmp=$(mktemp)
    ( while :; do sleep 2; sample >> "$tmp"; done ) &
    sampler=$!
    line=$(rocblas-bench $args $COMMON -i "$iters" -j 20 2>/dev/null | tail -1)
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

    hot_us=$(echo "$line" | iter_us)
    gflops=$(echo "$line" | awk -F, '{gsub(/ /,"",$(NF-1)); print $(NF-1)}')
    # Drop the first loaded sample: power is at the cap but clocks are still settling.
    loaded=$(awk -v th="$THRESH" '$2>=th' "$tmp" | tail -n +2)
    clk=$(echo "$loaded" | awk '{print $1}' | median)
    pwr=$(echo "$loaded" | awk '{print $2}' | median)
    rm -f "$tmp"

    tput=$(awk -v g="$gflops" 'BEGIN{printf "%.1f", g/1000}')
    pct=$(awk -v c="${clk:-0}" -v m="${MAXCLK:-0}" 'BEGIN{if(m>0) printf "%.0f", c*100/m}')
    echo "${tput} ${unit}" >&2
    ROWS+=("$name|$tput $unit|${clk:--} MHz|${pct:--}%|${pwr:--} W")
done

echo >&2
echo "| Precision | Throughput | Steady Clock | % Max Clock | Power |"
echo "| --- | --- | --- | --- | --- |"
for r in "${ROWS[@]}"; do
    IFS='|' read -r a b c d e <<< "$r"
    printf '| %s | %s | %s | %s | %s |\n' "$a" "$b" "$c" "$d" "$e"
done

if [ -n "$CSV" ]; then
    { echo "gpu,max_sclk_mhz,size,precision,throughput,unit,clock_mhz,pct_max_clock,power_w"
      for r in "${ROWS[@]}"; do
          IFS='|' read -r a b c d e <<< "$r"
          printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$GPU" "$MAXCLK" "$SIZE" "$a" \
              "${b% *}" "${b##* }" "${c% *}" "${d%\%}" "${e% *}"
      done
    } > "$CSV"
    echo "wrote $CSV" >&2
fi

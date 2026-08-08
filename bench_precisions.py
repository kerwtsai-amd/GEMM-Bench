#!/usr/bin/env python3
r"""Benchmark rocBLAS and hipBLASLt GEMM across precisions and report sustained
throughput alongside the clock and power the GPU actually settles at.

Run inside the container, on a compute node:
  srun -N1 -n1 -p mi3501x --gres=gpu:1 -t 00:40:00 \
    apptainer exec --bind $PWD:/work rocblas.sif \
    python3 /work/bench_precisions.py -o /work/mi350x.csv

Which precisions run on which backends is decided by bench_config.json, keyed by
the GPU architecture detected at startup. Progress goes to stderr and the table
to stdout, so redirecting stdout leaves only the results.
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "bench_config.json")

# Anything the config's "defaults" section omits falls back to these.
BUILTIN_DEFAULTS = {
    "size": 16384,
    "target_seconds": 25,
    "sample_interval_s": 2.0,
    "calibration_iters": 10,
    "calibration_warmup": 2,
    "min_iters": 20,
    "warmup_iters": 20,
    "loaded_power_factor": 1.5,
    "format": "table",
    "csv": None,
}

# Likewise for the "roofline" section, which older configs do not have at all.
BUILTIN_ROOFLINE = {
    "binary": "rocprof-compute",
    "peaks_file": "machine_peaks.json",
    "bandwidth": "HBMBw",
}

DEVICE_RE = re.compile(r"^Device ID \d+ : (.+)$", re.MULTILINE)
MAXCLK_RE = re.compile(r"max\. SCLK (\d+) MHz")

AGENT_RE = re.compile(r"^Agent \d+")
VERSION_RE = re.compile(r"^.*version:.*$", re.MULTILINE)


def log(message="", end="\n"):
    sys.stderr.write(message + end)
    sys.stderr.flush()


def die(message):
    log("error: " + message)
    sys.exit(1)


def capture(argv, merge_stderr=False, env=None):
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
            universal_newlines=True,
            env=env)
    except OSError:
        return None
    return proc.stdout


# ---------------------------------------------------------------- measurement

def parse_result(output, gflops_column, time_column):
    """Returns (gflops, microseconds) from the tool's result line, or None if
    the call failed. The two benchmarks put throughput in different columns."""
    if not output:
        return None
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    fields = lines[-1].split(",")
    if len(fields) <= 3:
        return None
    try:
        gflops = float(fields[gflops_column].strip())
        microseconds = float(fields[time_column].strip())
    except (IndexError, ValueError):
        return None
    if gflops <= 0 or microseconds <= 0:
        return None
    return gflops, microseconds


def bench(backend, args, iters, warmup, size, env):
    argv = [backend["binary"]]
    argv += [a.format(size=size) for a in backend["common_args"]]
    argv += [a.format(size=size) for a in args]
    argv += [backend["iters_flag"], str(iters)]
    argv += [backend["warmup_flag"], str(warmup)]
    return parse_result(capture(argv, env=env),
                        backend["gflops_column"], backend["time_column"])


def format_bdf(domain, bdfid):
    """rocminfo packs bus, device and function into one integer."""
    return "%04x:%02x:%02x.%d" % (domain, (bdfid >> 8) & 0xff,
                                  (bdfid >> 3) & 0x1f, bdfid & 0x7)


def visible_gpu_bdfs():
    """PCI addresses of the GPUs visible to this process, in HIP device order.
    rocminfo goes through the same runtime the benchmarks do, so it already
    accounts for however the scheduler set ROCR_VISIBLE_DEVICES."""
    output = capture(["rocminfo"])
    if not output:
        return []
    bdfs = []
    agent = {}

    def flush():
        if agent.get("gpu") and "bdfid" in agent:
            bdfs.append(format_bdf(agent.get("domain", 0), agent["bdfid"]))

    for line in output.splitlines():
        stripped = line.strip()
        if AGENT_RE.match(stripped):
            flush()
            agent.clear()
        elif stripped.startswith("Device Type:"):
            agent["gpu"] = stripped.endswith("GPU")
        elif stripped.startswith("Domain:") or stripped.startswith("BDFID:"):
            key, _, value = stripped.partition(":")
            value = value.strip()
            if value.isdigit():
                agent[key.lower()] = int(value)
    flush()
    return bdfs


def amd_smi_gpus():
    output = capture(["amd-smi", "list", "--json"])
    if not output:
        return []
    try:
        return json.loads(output)
    except ValueError:
        return []


def resolve_gpu(device):
    """(amd-smi index, PCI address) of the GPU the benchmarks will use.

    amd-smi numbers GPUs by PCI address and the ROCm runtime by KFD node id,
    and on a multi-GPU node those two orders differ, so an index from one is
    meaningless to the other. Matching on the PCI address sidesteps the whole
    question and works the same under a scheduler or on a bare node."""
    bdfs = visible_gpu_bdfs()
    if device >= len(bdfs):
        return None, None
    wanted = bdfs[device]
    for gpu in amd_smi_gpus():
        if str(gpu.get("bdf", "")).lower() == wanted:
            return str(gpu.get("gpu")), wanted
    return None, wanted


def amd_smi_clock_power(index):
    """The clock and power section of amd-smi's report for one GPU, or None if
    the GPU could not be resolved, the tool is missing, or the driver is down."""
    if index is None:
        return None
    output = capture(["amd-smi", "metric", "-g", index, "-c", "-p", "--json"])
    if not output:
        return None
    try:
        return json.loads(output)["gpu_data"][0]
    except (ValueError, KeyError, IndexError):
        return None


def scalar(field):
    """amd-smi wraps each reading as {"value": ..., "unit": ...} and writes the
    string "N/A" where a sensor is absent."""
    if isinstance(field, dict):
        field = field.get("value")
    return float(field) if isinstance(field, (int, float)) else None


def gfx_clocks(data, key):
    """MI300-class parts report `key` once per XCD, as gfx_0 ... gfx_7."""
    clocks = []
    for name, entry in data.get("clock", {}).items():
        if name.startswith("gfx") and isinstance(entry, dict):
            value = scalar(entry.get(key))
            if value:
                clocks.append(value)
    return clocks


def amd_smi_max_clock(index):
    data = amd_smi_clock_power(index)
    clocks = gfx_clocks(data, "max_clk") if data else []
    return int(max(clocks)) if clocks else None


def detect_gpu(binary, index, env):
    """Returns (name, max_sclk_mhz). The banner rocblas-bench prints on any run
    carries both; amd-smi is the fallback for the clock."""
    output = capture([binary, "-f", "gemm", "-r", "f32_r",
                      "-m", "64", "-n", "64", "-k", "64", "-i", "1"],
                     merge_stderr=True, env=env)
    if output is None:
        return None, None
    found = DEVICE_RE.search(output)
    name = found.group(1).strip() if found else "unknown"
    found = MAXCLK_RE.search(output)
    return name, int(found.group(1)) if found else amd_smi_max_clock(index)


def sample_clock_power(index):
    """(sclk MHz, socket W) for one GPU. The XCDs run in lockstep under a large
    GEMM, so the fastest one is the boost the kernel actually reached while any
    XCD left idle only reports its deep-sleep clock."""
    data = amd_smi_clock_power(index)
    if not data:
        return None
    power = scalar(data.get("power", {}).get("socket_power"))
    clocks = gfx_clocks(data, "clk")
    if power is None or not clocks:
        return None
    return max(clocks), power


class Sampler(threading.Thread):
    def __init__(self, interval, index):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self.index = index
        self.samples = []
        self._done = threading.Event()

    def run(self):
        while not self._done.wait(self.interval):
            reading = sample_clock_power(self.index)
            if reading:
                self.samples.append(reading)

    def stop(self):
        self._done.set()
        self.join(timeout=self.interval + 10)
        return self.samples


def median_int(values):
    if not values:
        return None
    return "%.0f" % statistics.median(values)


def unsupported_row(name, backend_key):
    row = {"precision": name, "backend": backend_key, "unit": "unsupported",
           "throughput": "unsupported", "clock": "-", "pct": "-", "power": "-"}
    row.update(BLANK_ROOFLINE)
    return row


# ------------------------------------------------------------------- roofline

# The rendered columns show "-" where a number is missing; the CSV columns are
# left empty so a spreadsheet reads them as blank rather than as text.
BLANK_ROOFLINE = {"pct_peak": "-", "per_watt": "-", "peak": "",
                  "ai": "", "bound": "", "attainable": "", "pct_roofline": ""}


def read_roofline_csv(path):
    """The ceilings from one rocprof-compute microbenchmark run.

    Its roofline.csv carries one row per device and three columns per metric:
    the mean and the Low/High of the spread. rocprof-compute's own plots draw
    the mean, so that is what we keep, and a metric it could not measure comes
    back as zero, which we treat as absent."""
    try:
        with open(path) as handle:
            rows = list(csv.reader(handle))
    except IOError:
        return None
    if len(rows) < 2:
        return None
    peaks = {}
    for name, value in zip(rows[0], rows[1]):
        if name == "device" or name.endswith("Low") or name.endswith("High"):
            continue
        try:
            value = float(value)
        except ValueError:
            continue
        if value > 0:
            peaks[name] = value
    return peaks or None


def find_roofline_csv(root):
    """rocprof-compute parameterises its output path with the GPU model, so
    the file does not always land directly in the directory we hand it."""
    for path, _, names in os.walk(root):
        if "roofline.csv" in names:
            return os.path.join(path, "roofline.csv")
    return None


def tool_version(binary):
    output = capture([binary, "--version"], merge_stderr=True)
    found = VERSION_RE.search(output) if output else None
    return found.group(0).strip() if found else None


def calibrate_peaks(roofline, env, path, meta):
    """Measures this GPU's ceilings and records them for later sweeps.

    rocprof-compute's --bench-only mode runs its roofline microbenchmarks and
    nothing else: no application, no hardware counters. That matters twice
    over. Counter collection is often locked down on a shared cluster, and it
    would perturb exactly what this script measures, clocks and power under a
    sustained load, so the ceilings are collected once, here, and cached.

    The measured peak is what the part reaches on back-to-back MFMA with no
    memory traffic, which is a higher clock than any real GEMM sustains. It is
    an absolute ceiling, not a fair target."""
    workdir = tempfile.mkdtemp(prefix="gemm-bench-peaks-")
    try:
        # rocprof-compute numbers devices through HIP, so pinning the child to
        # one GPU makes --device 0 unambiguous no matter how the node is
        # ordered -- the same reason the benchmarks themselves are pinned.
        argv = [roofline["binary"], "profile", "--bench-only",
                "--device", "0", "--output-directory", workdir]
        log("running: %s" % " ".join(argv))
        try:
            status = subprocess.call(argv, stdout=sys.stderr,
                                     stderr=subprocess.STDOUT, env=env)
        except OSError as exc:
            die("cannot run %s: %s" % (roofline["binary"], exc))
        if status != 0:
            die("%s exited with status %d" % (roofline["binary"], status))
        found = find_roofline_csv(workdir)
        peaks = read_roofline_csv(found) if found else None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if not peaks:
        die("%s wrote no usable roofline.csv; the microbenchmark has kernels "
            "for gfx90a, gfx942 and gfx950 only" % roofline["binary"])

    profile = {
        "gpu": meta["gpu"],
        "bdf": meta["bdf"],
        "max_sclk_mhz": meta["max_clock"],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": tool_version(roofline["binary"]),
        "units": "Bw entries are GB/s; Flops and Ops entries are per second, "
                 "in units of 1e9",
        "peaks": peaks,
    }
    try:
        with open(path, "w") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except IOError as exc:
        die("cannot write %s: %s" % (path, exc))

    log()
    for name in sorted(peaks):
        log("  %-16s %12.1f" % (name, peaks[name]))
    log()
    log("wrote %s" % path)
    return profile


def load_peaks(path, gpu):
    """The cached ceilings, if they belong to the GPU in front of us.

    A profile measured on another part would silently produce a wrong
    denominator, which is worse than no percentage at all, so a mismatch is
    refused rather than used."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            profile = json.load(handle)
    except (IOError, ValueError) as exc:
        log("warning: ignoring %s: %s" % (path, exc))
        return None
    if not profile.get("peaks"):
        log("warning: %s has no peaks; re-run with --calibrate-peaks" % path)
        return None
    recorded = profile.get("gpu")
    if recorded and gpu and recorded != gpu:
        log("warning: %s was measured on '%s' but this is '%s'; ignoring it. "
            "Re-run with --calibrate-peaks." % (path, recorded, gpu))
        return None
    return profile


def arithmetic_intensity(m, n, k, ab_bytes, cd_bytes):
    """FLOP per byte of memory traffic for one GEMM.

    Counts each operand read once and D written once; beta is 0 throughout the
    sweep, so C is never read. A real kernel moves more than this whenever a
    tile has to be re-read, so this is the optimistic end of the range and puts
    the point as far right on the roofline as it could possibly sit."""
    traffic = ab_bytes * (float(m) * k + float(k) * n) + cd_bytes * float(m) * n
    if traffic <= 0:
        return None
    return 2.0 * m * n * k / traffic


def roofline_metrics(precision, peaks, bandwidth_key, m, n, k, gflops):
    """Where one measurement sits relative to the empirical ceilings.

    Both the flat compute ceiling and the sloped bandwidth ceiling are
    reported, because which one applies depends on the shape: a large square
    GEMM is far into the compute-bound region, while a skinny one is not."""
    metrics = {"peak": None, "pct_peak": None, "ai": None,
               "attainable": None, "pct_roofline": None, "bound": None}
    if not peaks:
        return metrics

    widths = precision.get("bytes") or {}
    if widths.get("ab") and widths.get("cd"):
        metrics["ai"] = arithmetic_intensity(m, n, k, widths["ab"],
                                             widths["cd"])

    key = precision.get("peak")
    peak = peaks.get(key) if key else None
    if peak:
        metrics["peak"] = peak
        if gflops:
            metrics["pct_peak"] = gflops * 100.0 / peak

    bandwidth = peaks.get(bandwidth_key)
    if peak and bandwidth and metrics["ai"]:
        ceiling = metrics["ai"] * bandwidth
        metrics["bound"] = "memory" if ceiling < peak else "compute"
        metrics["attainable"] = min(peak, ceiling)
        if gflops:
            metrics["pct_roofline"] = gflops * 100.0 / metrics["attainable"]
    return metrics


def format_roofline(metrics, gflops, power):
    """Turns the raw metrics into the cells the table and the CSV carry."""
    cells = dict(BLANK_ROOFLINE)
    if metrics["pct_peak"] is not None:
        cells["pct_peak"] = "%.1f" % metrics["pct_peak"]
    if metrics["peak"]:
        cells["peak"] = "%.1f" % (metrics["peak"] / 1000.0)
    if metrics["ai"]:
        cells["ai"] = "%.1f" % metrics["ai"]
    if metrics["attainable"]:
        cells["attainable"] = "%.1f" % (metrics["attainable"] / 1000.0)
    if metrics["pct_roofline"] is not None:
        cells["pct_roofline"] = "%.1f" % metrics["pct_roofline"]
    cells["bound"] = metrics["bound"] or ""
    if gflops and power not in (None, "-"):
        cells["per_watt"] = "%.2f" % (gflops / 1000.0 / float(power))
    return cells


# --------------------------------------------------------------------- config

def load_config(path):
    try:
        with open(path) as handle:
            config = json.load(handle)
    except IOError as exc:
        die("cannot read config %s: %s" % (path, exc))
    except ValueError as exc:
        die("%s is not valid JSON: %s" % (path, exc))
    for section in ("defaults", "backends", "precisions", "chips"):
        if section not in config:
            die("%s is missing the \"%s\" section" % (path, section))
    return config


def select_chip(config, gpu_name, override):
    chips = config["chips"]
    if override:
        if override not in chips:
            die("no chip profile named '%s'; the config defines %s"
                % (override, ", ".join(sorted(chips))))
        return override, chips[override]
    lowered = gpu_name.lower()
    for key, chip in chips.items():
        if key == "default":
            continue
        for pattern in chip.get("match", []):
            if pattern.lower() in lowered:
                return key, chip
    if "default" in chips:
        return "default", chips["default"]
    die("no chip profile matches '%s' and the config has no \"default\" "
        "profile; add one, or pass --chip" % gpu_name)


def build_cases(config, chip, only_precisions, only_backends):
    cases = []
    for name, backend_keys in chip["run"].items():
        if name not in config["precisions"]:
            die("chip profile runs precision '%s', which is not defined under "
                "\"precisions\"" % name)
        if only_precisions and name not in only_precisions:
            continue
        precision = config["precisions"][name]
        for key in backend_keys:
            if key not in config["backends"]:
                die("chip profile uses backend '%s', which is not defined "
                    "under \"backends\"" % key)
            if key not in precision.get("args", {}):
                die("precision '%s' has no '%s' argument template; add one "
                    "under precisions.%s.args" % (name, key, name))
            if only_backends and key not in only_backends:
                continue
            cases.append((name, precision, key,
                          config["backends"][key], precision["args"][key]))
    return cases


# ------------------------------------------------------------------ rendering

def box_charset():
    """Falls back to ASCII when the terminal encoding cannot carry the box
    drawing characters, which happens under a C locale."""
    unicode_set = "─│┌┬┐├┼┤└┴┘"
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        unicode_set.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "-|+++++++++"
    return unicode_set


def render_box(headers, rows, aligns):
    horizontal, vertical, tl, tm, tr, ml, mm, mr, bl, bm, br = box_charset()
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def rule(left, middle, right):
        return left + middle.join(horizontal * (w + 2) for w in widths) + right

    def line(cells):
        padded = [cell.rjust(w) if align == ">" else cell.ljust(w)
                  for cell, w, align in zip(cells, widths, aligns)]
        joiner = " " + vertical + " "
        return vertical + " " + joiner.join(padded) + " " + vertical

    out = [rule(tl, tm, tr), line(headers), rule(ml, mm, mr)]
    out += [line(row) for row in rows]
    out.append(rule(bl, bm, br))
    return "\n".join(out)


def render_markdown(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


CSV_HEADER = ["gpu", "max_sclk_mhz", "size", "precision", "library",
              "throughput", "unit", "clock_mhz", "pct_max_clock", "power_w",
              "peak", "pct_peak", "per_watt", "arith_intensity", "bound",
              "attainable", "pct_roofline"]


def csv_rows(results, gpu, max_clock, size):
    rows = []
    for result in results:
        rows.append([gpu, max_clock if max_clock else "", size,
                     result["precision"], result["backend"],
                     result["throughput"], result["unit"],
                     result["clock"], result["pct"], result["power"],
                     result["peak"],
                     result["pct_peak"] if result["pct_peak"] != "-" else "",
                     result["per_watt"] if result["per_watt"] != "-" else "",
                     result["ai"], result["bound"], result["attainable"],
                     result["pct_roofline"]])
    return rows


# ----------------------------------------------------------------------- main

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Sweep GEMM precisions on rocBLAS and hipBLASLt, reporting "
                    "sustained throughput with the clock and power the GPU "
                    "settles at.")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                        help="benchmark configuration (default: %(default)s)")
    parser.add_argument("--chip", help="chip profile to use instead of the one "
                                       "detected from the GPU name")
    parser.add_argument("-s", "--size", type=int, help="square GEMM dimension")
    parser.add_argument("-t", "--target", type=float,
                        help="target duration in seconds per case")
    parser.add_argument("-p", "--precisions",
                        help="comma-separated subset, e.g. f64_r,f16_r")
    parser.add_argument("-b", "--backends",
                        help="comma-separated backend subset, e.g. hipblaslt")
    parser.add_argument("-o", "--output", help="also write results as CSV")
    parser.add_argument("-f", "--format", choices=("table", "markdown", "csv"),
                        help="stdout format (default: table)")
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="which of the visible GPUs to use (default: 0)")
    parser.add_argument("--sample-interval", type=float,
                        help="seconds between clock/power samples")
    parser.add_argument("--peaks", help="file holding this machine's measured "
                                        "ceilings (default: machine_peaks."
                                        "json beside the script)")
    parser.add_argument("--calibrate-peaks", action="store_true",
                        help="measure this GPU's ceilings with rocprof-compute"
                             ", write them to the peaks file, and exit")
    parser.add_argument("--list-chips", action="store_true",
                        help="print the chip profiles in the config and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    defaults = dict(BUILTIN_DEFAULTS)
    defaults.update(config["defaults"])
    roofline = dict(BUILTIN_ROOFLINE)
    roofline.update(config.get("roofline", {}))

    if args.list_chips:
        for key, chip in config["chips"].items():
            print("%-10s %s" % (key, chip.get("description", "")))
            for name, backends in chip["run"].items():
                print("    %-8s %s" % (name, ", ".join(backends)))
        return 0

    size = args.size if args.size is not None else defaults["size"]
    target = (args.target if args.target is not None
              else defaults["target_seconds"])
    interval = (args.sample_interval if args.sample_interval is not None
                else defaults["sample_interval_s"])
    out_format = args.format if args.format else defaults["format"]
    csv_path = args.output if args.output else defaults["csv"]
    only_precisions = (set(args.precisions.split(","))
                       if args.precisions else None)
    only_backends = set(args.backends.split(",")) if args.backends else None

    # Pinning the children to one device makes the GPU being sampled and the
    # GPU doing the work the same by construction rather than by inference.
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(args.device)
    index, bdf = resolve_gpu(args.device)

    probe = config["backends"]["rocblas"]["binary"]
    gpu, max_clock = detect_gpu(probe, index, env)
    if gpu is None:
        die("%s not found; run this inside the container" % probe)

    peaks_file = args.peaks if args.peaks else roofline["peaks_file"]
    if not os.path.isabs(peaks_file):
        peaks_file = os.path.join(SCRIPT_DIR, peaks_file)

    if args.calibrate_peaks:
        log("GPU: %s   Device: %d -> %s" % (gpu, args.device, bdf or "?"))
        calibrate_peaks(roofline, env, peaks_file,
                        {"gpu": gpu, "bdf": bdf, "max_clock": max_clock})
        return 0

    peaks_profile = load_peaks(peaks_file, gpu)
    peaks = peaks_profile["peaks"] if peaks_profile else None

    chip_key, chip = select_chip(config, gpu, args.chip)
    cases = build_cases(config, chip, only_precisions, only_backends)
    if not cases:
        die("no cases left to run after filtering")

    idle = sample_clock_power(index)
    idle_power = idle[1] if idle else 0.0
    # A sample counts as loaded once power clears 1.5x idle; this drops the
    # ramp-up and, for an unsupported precision, leaves nothing at all.
    threshold = idle_power * defaults["loaded_power_factor"]

    if index is None:
        log("warning: could not match device %d to an amd-smi GPU%s; clock and "
            "power will be blank" % (args.device, " (%s)" % bdf if bdf else ""))
    log("GPU: %s   Device: %d -> %s (amd-smi index %s)"
        % (gpu, args.device, bdf or "?", index if index is not None else "?"))
    log("Profile: %s (%s)   Max SCLK: %s MHz   Idle power: %g W   Size: %d^3   "
        "Target: %gs/case"
        % (chip_key, chip.get("description", ""), max_clock or "?", idle_power,
           size, target))
    if peaks_profile:
        log("Peaks: %s (measured %s)"
            % (peaks_file, peaks_profile.get("generated", "?")))
    else:
        log("Peaks: none; run --calibrate-peaks for roofline columns")
    log()

    width = max(len(name) + len(config["backends"][key]["label"]) + 3
                for name, _, key, _, _ in cases)
    results = []
    for name, precision, key, backend, bench_args in cases:
        unit = precision["unit"]
        label = "%s [%s]" % (name, backend["label"])
        log("calibrating %-*s ... " % (width, label), end="")

        trial = bench(backend, bench_args, defaults["calibration_iters"],
                      defaults["calibration_warmup"], size, env)
        if trial is None:
            log("unsupported on this GPU")
            results.append(unsupported_row(name, key))
            continue

        iters = max(int(target * 1e6 / trial[1]), defaults["min_iters"])
        log("%g us/iter -> %d iters, running ... " % (trial[1], iters), end="")

        sampler = Sampler(interval, index)
        sampler.start()
        measured = bench(backend, bench_args, iters,
                         defaults["warmup_iters"], size, env)
        samples = sampler.stop()

        # Drop the first loaded sample: power is at the cap but clocks are still
        # settling.
        loaded = [s for s in samples if s[1] >= threshold][1:]
        clock = median_int([s[0] for s in loaded])
        power = median_int([s[1] for s in loaded])

        if measured is None:
            log("failed")
            results.append(unsupported_row(name, key))
            continue

        throughput = "%.1f" % (measured[0] / 1000.0)
        pct = ("%.0f" % (float(clock) * 100 / max_clock)
               if clock and max_clock else "-")
        metrics = roofline_metrics(precision, peaks, roofline["bandwidth"],
                                   size, size, size, measured[0])
        log("%s %s%s%s" % (throughput, unit,
                           "" if metrics["pct_peak"] is None
                           else "  (%.0f%% of peak)" % metrics["pct_peak"],
                           "" if loaded else "  (no loaded samples)"))
        result = {"precision": name, "backend": key, "unit": unit,
                  "throughput": throughput, "clock": clock or "-",
                  "pct": pct, "power": power or "-"}
        result.update(format_roofline(metrics, measured[0], power))
        results.append(result)
    log()

    # The roofline columns only appear once there is something to put in them,
    # so an uncalibrated machine gets the table it always had.
    show_peak = any(result["pct_peak"] != "-" for result in results)
    show_watt = any(result["per_watt"] != "-" for result in results)
    headers = ["Precision", "BLAS backend", "Throughput"]
    aligns = ["<", "<", ">"]
    if show_peak:
        headers.append("% Peak")
        aligns.append(">")
    headers += ["Steady Clock", "% Max Clock", "Power"]
    aligns += [">", ">", ">"]
    if show_watt:
        headers.append("Efficiency")
        aligns.append(">")

    rows = []
    for result in results:
        supported = result["throughput"] != "unsupported"
        row = [
            result["precision"],
            config["backends"][result["backend"]]["label"],
            "%s %s" % (result["throughput"], result["unit"]) if supported
            else "unsupported",
        ]
        if show_peak:
            row.append("%s%%" % result["pct_peak"]
                       if result["pct_peak"] != "-" else "-")
        row += [
            "%s MHz" % result["clock"] if result["clock"] != "-" else "-",
            "%s%%" % result["pct"] if result["pct"] != "-" else "-",
            "%s W" % result["power"] if result["power"] != "-" else "-",
        ]
        if show_watt:
            row.append("%s %s/W" % (result["per_watt"], result["unit"])
                       if result["per_watt"] != "-" else "-")
        rows.append(row)

    bound = sorted(set(result["precision"] for result in results
                       if result["bound"] == "memory"))
    if bound:
        log("note: at this shape %s sit under the bandwidth ceiling rather "
            "than the compute one, so read pct_roofline in the CSV instead of "
            "%% Peak" % ", ".join(bound))

    if out_format == "csv":
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows(results, gpu, max_clock, size))
    elif out_format == "markdown":
        print(render_markdown(headers, rows))
    else:
        print(render_box(headers, rows, aligns))

    if csv_path:
        with open(csv_path, "w") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CSV_HEADER)
            writer.writerows(csv_rows(results, gpu, max_clock, size))
        log("wrote %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

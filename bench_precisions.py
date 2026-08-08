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
import statistics
import subprocess
import sys
import threading

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "bench_config.json")

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

DEVICE_RE = re.compile(r"^Device ID \d+ : (.+)$", re.MULTILINE)
MAXCLK_RE = re.compile(r"max\. SCLK (\d+) MHz")

AGENT_RE = re.compile(r"^Agent \d+")


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
    return {"precision": name, "backend": backend_key, "unit": "unsupported",
            "throughput": "unsupported", "clock": "-", "pct": "-", "power": "-"}


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
            cases.append((name, precision["unit"], key,
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
              "throughput", "unit", "clock_mhz", "pct_max_clock", "power_w"]


def csv_rows(results, gpu, max_clock, size):
    rows = []
    for result in results:
        rows.append([gpu, max_clock if max_clock else "", size,
                     result["precision"], result["backend"],
                     result["throughput"], result["unit"],
                     result["clock"], result["pct"], result["power"]])
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
    parser.add_argument("--list-chips", action="store_true",
                        help="print the chip profiles in the config and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    defaults = dict(BUILTIN_DEFAULTS)
    defaults.update(config["defaults"])

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
    log()

    width = max(len(name) + len(config["backends"][key]["label"]) + 3
                for name, _, key, _, _ in cases)
    results = []
    for name, unit, key, backend, bench_args in cases:
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
        log("%s %s%s" % (throughput, unit,
                         "" if loaded else "  (no loaded samples)"))
        results.append({"precision": name, "backend": key, "unit": unit,
                        "throughput": throughput, "clock": clock or "-",
                        "pct": pct, "power": power or "-"})
    log()

    headers = ["Precision", "BLAS backend", "Throughput", "Steady Clock",
               "% Max Clock", "Power"]
    aligns = ["<", "<", ">", ">", ">", ">"]
    rows = []
    for result in results:
        supported = result["throughput"] != "unsupported"
        rows.append([
            result["precision"],
            config["backends"][result["backend"]]["label"],
            "%s %s" % (result["throughput"], result["unit"]) if supported
            else "unsupported",
            "%s MHz" % result["clock"] if result["clock"] != "-" else "-",
            "%s%%" % result["pct"] if result["pct"] != "-" else "-",
            "%s W" % result["power"] if result["power"] != "-" else "-",
        ])

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

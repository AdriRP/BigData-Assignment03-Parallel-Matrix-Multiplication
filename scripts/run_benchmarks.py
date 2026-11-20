"""Benchmark driver: every configuration runs in its own JVM process.

For each placement (config.json "placements"), each size n ("sizes") and each
variant (vec = HotSpot auto-vectorisation on, the default; novec =
-XX:-UseSuperWord, only for the sizes in "vectorisation_sizes"):
  seq             the sequential blocked baseline
  par, p threads  the parallel blocked method for every p in "threads"

Placements:
  pinned   the process is restricted (CPU affinity) to the logical processors
           of common.pinned_cpus(p): one per physical core first, then the
           second logical processor of each core; seq uses the same set as p = 1
  default  no restriction: the Windows scheduler places the threads

Each process prints one line per timed sample, then "DONE", and waits (--hold)
while the driver reads its peak working set (Windows PeakWorkingSetSize via
psutil). A sample that produces no result within time_budget_per_run_s is
killed and recorded as a timeout; larger sizes of the same series are then
recorded as not attempted.

Outputs: results/raw_times.csv (one row per timed sample) and results/runs.csv
(one row per process). With --placements, only the placements listed are run
and the rows of the others are kept.
"""
import argparse
import csv
import json
import queue
import subprocess
import threading
import time

import psutil

from common import CONFIG, OUTPUTS, RESULTS, batch, input_paths, java, output_path, pinned_cpus

LOAD_SLACK_S = 30  # extra time allowed before the first result (loading inputs, pre-warm)
INFO_KEYS = ["batch", "worker_threads", "available_processors", "jvm_live_threads", "result_sha256"]
TIMES_HEADER = ["placement", "method", "threads", "variant", "n", "dtype", "batch", "kind", "rep", "time_ns",
                "time_per_mult_s", "checksum"]
RUNS_HEADER = ["placement", "method", "threads", "variant", "n", "oversubscribed", "affinity", "status", "warmups",
               "measured_runs", "peak_working_set_bytes", "dense_matrix_bytes", *INFO_KEYS, "process_wall_s"]


def reader(stream, q):
    for line in stream:
        q.put(line.strip())
    q.put(None)


def run_process(cmd, budget, cpus):
    """Run one configuration. Returns (timings, info, peak_bytes, status, wall_s)."""
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    if cpus is not None:
        psutil.Process(proc.pid).cpu_affinity(cpus)  # applies to every thread of the process
    q = queue.Queue()
    threading.Thread(target=reader, args=(proc.stdout, q), daemon=True).start()
    timings, info, peak, status, first = [], {}, None, "ok", True
    while True:
        try:
            line = q.get(timeout=budget + (LOAD_SLACK_S if first else 0))
        except queue.Empty:
            proc.kill()
            status = f"timeout (> {budget} s per sample)"
            break
        first = False
        if line is None:
            status = f"crashed (exit {proc.wait()})"
            break
        if line == "DONE":
            peak = psutil.Process(proc.pid).memory_info().peak_wset
            proc.stdin.write("\n")
            proc.stdin.flush()
            break
        parts = line.split(",")
        if parts[0] == "T":
            _, phase, kind, idx, ns, chk = parts
            timings.append({"phase": phase, "kind": "warmup" if kind == "W" else "measured",
                            "rep": int(idx), "time_ns": int(ns), "checksum": chk})
            print(f"    {timings[-1]['kind']:8s} #{idx}: {int(ns) / 1e9:10.4f} s", flush=True)
        elif parts[0] == "I":
            info[parts[1]] = parts[2]
        else:
            print("    ", line, flush=True)
    proc.wait()
    return timings, info, peak, status, time.perf_counter() - start


def configurations(placements, cores):
    """(placement, method, threads, variant, n, series, cpus, command) in execution order."""
    w, r, bs = CONFIG["warmups"], CONFIG["repetitions"], CONFIG["block_size"]
    for pl in placements:
        for n in CONFIG["sizes"]:
            pa, pb = input_paths(n)
            variants = ["vec"] + (["novec"] if n in CONFIG["vectorisation_sizes"] else [])
            for v in variants:
                # the output of one configuration per size is kept for the NumPy check
                out = str(output_path(n)) if (v == "vec" and pl == placements[0]) else "-"
                common = [str(n), str(pa), str(pb), str(bs), str(w), str(r), str(batch(n))]
                cpus = pinned_cpus(1, cores) if pl == "pinned" else None
                yield pl, "seq", 1, v, n, f"{pl}-seq-{v}", cpus, java(v) + ["seq", *common, out]
                for p in CONFIG["threads"]:
                    cpus = pinned_cpus(p, cores) if pl == "pinned" else None
                    yield pl, "par", p, v, n, f"{pl}-par{p}-{v}", cpus, java(v) + ["par", str(p), *common, "-"]


def kept_rows(path, placements):
    """Rows of placements that are not being re-run."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in list(csv.reader(f))[1:] if r and r[0] not in placements]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placements", nargs="*", default=CONFIG["placements"],
                        help="placements to run; rows of the other placements are kept")
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "environment.json", encoding="utf-8") as f:
        cores = json.load(f)["topology"]["cores"]
    budget = CONFIG["time_budget_per_run_s"]
    logical = psutil.cpu_count(logical=True)
    stopped = set()
    old_times = kept_rows(RESULTS / "raw_times.csv", args.placements)
    old_runs = kept_rows(RESULTS / "runs.csv", args.placements)
    with open(RESULTS / "raw_times.csv", "w", newline="", encoding="utf-8") as ft, \
         open(RESULTS / "runs.csv", "w", newline="", encoding="utf-8") as fr:
        tw = csv.writer(ft)
        tw.writerow(TIMES_HEADER)
        tw.writerows(old_times)
        rw = csv.writer(fr)
        rw.writerow(RUNS_HEADER)
        rw.writerows(old_runs)
        for pl, method, p, v, n, series, cpus, cmd in configurations(args.placements, cores):
            dense_bytes = 3 * 8 * n * n
            row = [pl, method, p, v, n]
            over = "yes" if p > logical else "no"
            aff = " ".join(map(str, cpus)) if cpus is not None else "all (not set)"
            if series in stopped:
                rw.writerow(row + [over, aff, "not attempted (smaller size timed out)", CONFIG["warmups"], 0, "",
                                   dense_bytes] + [""] * len(INFO_KEYS) + [""])
                continue
            if dense_bytes > CONFIG["memory_budget_bytes"]:
                rw.writerow(row + [over, aff, "not attempted (memory budget)", CONFIG["warmups"], 0, "", dense_bytes]
                            + [""] * len(INFO_KEYS) + [""])
                continue
            print(f"{pl} {method} p={p} {v} n={n} cpus={aff}", flush=True)
            timings, info, peak, status, wall = run_process(cmd, budget, cpus)
            b = batch(n)
            for t in timings:
                tw.writerow(row + [CONFIG["dtype"], b, t["kind"], t["rep"], t["time_ns"],
                                   f"{t['time_ns'] / 1e9 / b:.9f}", t["checksum"]])
            measured = sum(t["kind"] == "measured" for t in timings)
            rw.writerow(row + [over, aff, status, CONFIG["warmups"], measured, peak or "", dense_bytes]
                        + [info.get(k, "") for k in INFO_KEYS] + [f"{wall:.2f}"])
            ft.flush()
            fr.flush()
            if status != "ok":
                print(f"  -> {status}", flush=True)
                stopped.add(series)


if __name__ == "__main__":
    main()

"""Diagnostic: run one Bench configuration and record, every ~1 s, which logical
processors are busy (> 50 % load, psutil) and which physical cores have both
of their logical processors busy (cores = pairs 2k, 2k+1 on the study machine,
see results/environment.json).

Usage (from the repository root, after scripts/build.py and scripts/gen_inputs.py):
  python diagnostics/monitor_placement.py [--pin CPU,CPU,...] java -Xmx4g ... Bench par 8 2000 ...
--pin restricts the process to the listed logical processors.
"""
import subprocess
import sys
import threading
import time

import psutil

args = sys.argv[1:]
pin = None
if args[0] == "--pin":
    pin = [int(x) for x in args[1].split(",")]
    args = args[2:]
samples, stop = [], False


def sampler():
    psutil.cpu_percent(percpu=True)
    while not stop:
        time.sleep(0.5)
        samples.append((time.perf_counter(), psutil.cpu_percent(percpu=True)))


threading.Thread(target=sampler, daemon=True).start()
t0 = time.perf_counter()
proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
if pin is not None:
    psutil.Process(proc.pid).cpu_affinity(pin)
for line in proc.stdout:
    if line.startswith("T,"):
        print(f"[{time.perf_counter() - t0:6.1f} s] sample {int(line.split(',')[4]) / 1e9:.3f} s", flush=True)
proc.wait()
stop = True
time.sleep(0.6)
for ts, load in samples:
    busy = [i for i, x in enumerate(load) if x > 50]
    both = sorted({i // 2 for i in busy if i % 2 == 0 and i + 1 in busy})
    print(f"[{ts - t0:6.1f} s] busy logical={busy}  cores with both logical processors busy={both}")

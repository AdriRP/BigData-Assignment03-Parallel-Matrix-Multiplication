"""Diagnostic: effective clock of the busy logical processors while the parallel
kernel runs pinned with p threads (Windows counter '% de rendimiento del procesador',
i.e. % Processor Performance: effective frequency as a percentage of the base clock).
The counter name is the Spanish one used by the study machine; on an English
Windows it is '\\Processor Information(0,<cpu>)\\% Processor Performance'.

Usage (from the repository root):  python diagnostics/effective_clock.py <p> <n> <measured samples>"""
import statistics
import subprocess
import sys

import psutil

sys.path.insert(0, "scripts")
import json
from common import pinned_cpus

p, n, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cores = json.load(open("results/environment.json"))["topology"]["cores"]
cpus = pinned_cpus(p, cores)
cmd = ["java", "-Xmx4g", "-cp", "build/java", "Bench", "par", str(p), str(n), f"data/inputs/A_{n}.bin",
       f"data/inputs/B_{n}.bin", "256", "2", str(reps), "1", "-"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
psutil.Process(proc.pid).cpu_affinity(cpus)
# wait for the first warm-up sample so that the counter covers steady computation only
times = []
for line in proc.stdout:
    if line.startswith("T,kernel,W,1"):
        break
paths = ",".join(f"'\\Información del procesador(0,{c})\\% de rendimiento del procesador'" for c in cpus)
ps = (f"$OutputEncoding=[Text.Encoding]::UTF8;[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
      f"(Get-Counter -Counter @({paths}) -SampleInterval 1 -MaxSamples 8).CounterSamples | "
      f"ForEach-Object {{ $_.CookedValue }}")
mon = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps], stdout=subprocess.PIPE, text=True,
                       encoding="utf-8")
for line in proc.stdout:
    if line.startswith("T,kernel,M"):
        times.append(int(line.split(",")[4]) / 1e9)
proc.wait()
vals = [float(x.replace(",", ".")) for x in mon.communicate()[0].split()]
print(f"p={p} n={n}: sample times {['%.3f' % t for t in times]}; "
      f"busy CPUs {cpus}: % processor performance median {statistics.median(vals):.1f} "
      f"(min {min(vals):.1f}, max {max(vals):.1f}, {len(vals)} readings) "
      f"-> about {statistics.median(vals) / 100 * 2.501:.2f} GHz")

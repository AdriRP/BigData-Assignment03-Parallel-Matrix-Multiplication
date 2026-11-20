"""Correctness checks. Writes results/validation.csv.

1. The Java self-test (known products, identity, zero, sizes that are not a
   multiple of the block size or of the thread count, more threads than rows,
   a reused pool), for the sequential baseline and the parallel method.
2. The output of the sequential baseline at every size against NumPy's matrix
   product, with the rounding-error tolerance of Assignments 1 and 2:
   |C - C_ref| <= 2 gamma_n (|A||B|),  gamma_n = n u / (1 - n u),  u = 2^-53.
3. Every configuration (each placement, thread count, with and without
   auto-vectorisation) produced a result bit-identical to the sequential
   baseline at the same size (same SHA-256 of the raw bits of C).
4. Identical checksum in every timed sample of a configuration.
"""
import csv
import subprocess
from collections import defaultdict

import numpy as np

from common import CONFIG, RESULTS, input_paths, java, output_path

U = 2.0 ** -53


def load(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def main():
    rows = []

    def add(check, config, detail, ok):
        rows.append({"check": check, "configuration": config, "detail": detail, "result": "PASS" if ok else "FAIL"})
        print(f"{'PASS' if ok else 'FAIL'}  {check}  {config}  {detail}")

    for v in ("vec", "novec"):
        p = subprocess.run(java(v) + ["--selftest"], capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
        add("Java self-test", v, f"{sum(l.startswith('PASS') for l in lines)}/{len(lines)} cases",
            p.returncode == 0 and "SELFTEST OK" in p.stdout)

    for n in CONFIG["sizes"]:
        path = output_path(n)
        if not path.exists():
            continue
        pa, pb = input_paths(n)
        a, b = load(pa, n), load(pb, n)
        gamma = n * U / (1 - n * U)
        bound = 2 * gamma * (np.abs(a) @ np.abs(b))
        err = np.abs(load(path, n) - a @ b)
        add("seq vs NumPy within rounding tolerance", f"n={n}",
            f"max|err|={err.max():.3e}, max err/tol={(err / bound).max():.3e}", bool((err <= bound).all()))

    with open(RESULTS / "runs.csv", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    first = CONFIG["placements"][0]
    ref = {r["n"]: r["result_sha256"] for r in runs
           if r["placement"] == first and r["method"] == "seq" and r["variant"] == "vec"}
    for r in runs:
        if r["placement"] == first and r["method"] == "seq" and r["variant"] == "vec":
            continue
        add("bit-identical to the sequential baseline",
            f"{r['placement']} {r['method']} p={r['threads']} {r['variant']} n={r['n']}",
            "same SHA-256 of C", r["result_sha256"] == ref.get(r["n"]))

    checksums = defaultdict(set)
    with open(RESULTS / "raw_times.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            checksums[(r["placement"], r["method"], r["threads"], r["variant"], r["n"])].add(r["checksum"])
    for key, values in sorted(checksums.items(), key=lambda kv: (kv[0][0], int(kv[0][4]), kv[0][1], int(kv[0][2]),
                                                                 kv[0][3])):
        add("identical checksum in every sample", f"{key[0]} {key[1]} p={key[2]} {key[3]} n={key[4]}", "",
            len(values) == 1)

    with open(RESULTS / "validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} checks, {sum(r['result'] != 'PASS' for r in rows)} failed")


if __name__ == "__main__":
    main()

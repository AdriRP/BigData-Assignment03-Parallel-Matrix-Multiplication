"""Shared paths, configuration and the command that runs the Java harness."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
JAVA_CLASSES = BUILD / "java"
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
OUTPUTS = DATA / "outputs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

with open(ROOT / "config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)


def input_paths(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


def output_path(n):
    """Output of the sequential baseline (with auto-vectorisation), checked against NumPy."""
    return OUTPUTS / f"C_seq_{n}.bin"


def batch(n):
    """Multiplications per timed sample (more than one only for the smallest sizes)."""
    return CONFIG["batch"].get(str(n), 1)


def pinned_cpus(p, cores):
    """Logical processors allowed for p threads under the 'pinned' placement.

    cores lists, for each physical core in order, its logical processors (from
    env_info.py). One logical processor per physical core first, in core
    order; beyond the number of cores, the second logical processor of each
    core in the same order. More threads than logical processors: all of them.
    """
    order = [c[0] for c in cores] + [x for c in cores for x in c[1:]]
    return sorted(order[:min(p, len(order))])


def java(variant="vec"):
    """Command prefix for the harness; variant 'novec' disables HotSpot auto-vectorisation."""
    flags = list(CONFIG["jvm_flags"])
    if variant == "novec":
        flags += CONFIG["jvm_flags_no_vectorisation"]
    return ["java", *flags, "-cp", str(JAVA_CLASSES), "Bench"]

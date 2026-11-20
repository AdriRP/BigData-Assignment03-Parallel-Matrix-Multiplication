"""Summary tables and figures.

Reads results/raw_times.csv (measured samples only), results/runs.csv and
results/environment.json and writes results/summary.csv, results/speedup.csv,
results/summary.md and the figures in figures/.

Times are per multiplication (sample time / batch). The central value of every
configuration is the median of its measured samples; speedup and efficiency
use medians: S(p) = T_seq / T_p and E(p) = S(p) / p, where T_seq is the
sequential baseline of the same size, variant (vec or novec) and placement.
The main figures use the 'pinned' placement; figures/placement.png compares it
with the 'default' placement.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from common import CONFIG, FIGURES, RESULTS

# ordinal blue ramp (one step per size, light -> dark) and one marker per size
SIZE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SIZE_MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MAIN = "pinned"
KEYS = ["placement", "method", "threads", "variant", "n"]


def style(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, which="both")


def thread_axis(ax, threads, logical, physical):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads, [str(p) for p in threads], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Threads p")
    for x, label in ((physical, f"{physical} physical cores"), (logical, f"{logical} logical processors")):
        ax.axvline(x, color="#898781", linestyle=":", linewidth=1)
        ax.text(x * 1.04, 0.98, label, transform=ax.get_xaxis_transform(), fontsize=7.5, color=MUTED,
                va="top", rotation=90)


def summarise(raw):
    meas = raw[raw.kind == "measured"]
    g = meas.groupby(KEYS)["time_per_mult_s"]
    s = pd.DataFrame({"samples": g.size(), "median_s": g.median(), "q1_s": g.quantile(0.25),
                      "q3_s": g.quantile(0.75), "min_s": g.min(), "max_s": g.max()}).reset_index()
    s["iqr_s"] = s.q3_s - s.q1_s
    s["rel_iqr_pct"] = 100 * s.iqr_s / s.median_s
    return s


def speedups(s):
    seq = s[s.method == "seq"].set_index(["placement", "variant", "n"]).median_s
    par = s[s.method == "par"].copy()
    par["t_seq_s"] = [seq[k] for k in zip(par.placement, par.variant, par.n)]
    par["speedup"] = par.t_seq_s / par.median_s
    par["efficiency"] = par.speedup / par.threads
    return par


def fig_time(s, threads, logical, physical):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    v = s[(s.placement == MAIN) & (s.variant == "vec")]
    for (n, color, marker) in zip(CONFIG["sizes"], SIZE_COLORS, SIZE_MARKERS):
        d = v[(v.method == "par") & (v.n == n)].sort_values("threads")
        ax.errorbar(d.threads, d.median_s, yerr=[d.median_s - d.q1_s, d.q3_s - d.median_s], color=color,
                    marker=marker, linewidth=2, markersize=6, capsize=3, label=f"n = {n:,}")
        t_seq = v[(v.method == "seq") & (v.n == n)].median_s.iloc[0]
        ax.axhline(t_seq, color=color, linestyle=":", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Time per multiplication (s), median (bars: IQR)")
    thread_axis(ax, threads, logical, physical)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", title="parallel method; dotted = sequential",
              title_fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "time_threads.png", dpi=200)
    plt.close(fig)


def fig_speedup(sp, threads, logical, physical):
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    v = sp[(sp.placement == MAIN) & (sp.variant == "vec")]
    for (n, color, marker) in zip(CONFIG["sizes"], SIZE_COLORS, SIZE_MARKERS):
        d = v[v.n == n].sort_values("threads")
        ax[0].plot(d.threads, d.speedup, color=color, marker=marker, linewidth=2, markersize=6, label=f"n = {n:,}")
        ax[1].plot(d.threads, d.efficiency, color=color, marker=marker, linewidth=2, markersize=6, label=f"n = {n:,}")
    ax[0].plot(threads, threads, color=MUTED, linestyle="--", linewidth=1.2, label="ideal S(p) = p")
    ax[0].set_yscale("log", base=2)
    ax[0].set_yticks([0.5, 1, 2, 4, 8, 16, 32, 64], ["0.5", "1", "2", "4", "8", "16", "32", "64"])
    ax[0].set_ylabel("Speedup S(p) = T_seq / T_p")
    ax[1].axhline(1, color=MUTED, linestyle="--", linewidth=1.2)
    ax[1].set_ylim(0, max(1.1, v.efficiency.max() * 1.1))
    ax[1].set_ylabel("Efficiency E(p) = S(p) / p")
    for x in ax:
        thread_axis(x, threads, logical, physical)
        style(x)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "speedup_efficiency.png", dpi=200)
    plt.close(fig)


def fig_vectorisation(s, threads, logical, physical):
    sizes = CONFIG["vectorisation_sizes"]
    colors = dict(zip(CONFIG["sizes"], SIZE_COLORS))
    markers = dict(zip(CONFIG["sizes"], SIZE_MARKERS))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.8))
    par = s[(s.placement == MAIN) & (s.method == "par")]
    for n in sizes:
        vec = par[(par.n == n) & (par.variant == "vec")].set_index("threads").sort_index()
        nov = par[(par.n == n) & (par.variant == "novec")].set_index("threads").sort_index()
        ax[0].plot(vec.index, vec.median_s, color=colors[n], marker=markers[n], linewidth=2, markersize=6,
                   label=f"n = {n:,}")
        ax[0].plot(nov.index, nov.median_s, color=colors[n], marker=markers[n], linestyle="--", linewidth=2,
                   markersize=6, markerfacecolor="white", label=f"n = {n:,}, no auto-vectorisation")
        gain = (nov.median_s / vec.median_s).dropna()
        ax[1].plot(gain.index, gain.values, color=colors[n], marker=markers[n], linewidth=2, markersize=6,
                   label=f"n = {n:,}")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Time per multiplication (s), median")
    ax[1].axhline(1, color=MUTED, linestyle="--", linewidth=1.2)
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel("T (no auto-vectorisation) / T (auto-vectorised)")
    for x in ax:
        thread_axis(x, threads, logical, physical)
        style(x)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=3, handlelength=3.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIGURES / "vectorisation.png", dpi=200)
    plt.close(fig)


def fig_placement(sp, threads, logical, physical):
    """Speedup with and without CPU affinity (vec), for the two largest sizes."""
    sizes = CONFIG["sizes"][-2:]
    colors = dict(zip(CONFIG["sizes"], SIZE_COLORS))
    markers = dict(zip(CONFIG["sizes"], SIZE_MARKERS))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    v = sp[sp.variant == "vec"]
    for n in sizes:
        for pl, ls, face in (("pinned", "-", None), ("default", "--", "white")):
            d = v[(v.n == n) & (v.placement == pl)].sort_values("threads")
            if d.empty:
                continue
            # speedup range from the quartiles: T_seq / Q3(T_p) to T_seq / Q1(T_p)
            ax.errorbar(d.threads, d.speedup, yerr=[d.speedup - d.t_seq_s / d.q3_s, d.t_seq_s / d.q1_s - d.speedup],
                        color=colors[n], marker=markers[n], linestyle=ls, linewidth=2, markersize=6, capsize=3,
                        markerfacecolor=face or colors[n], label=f"n = {n:,}, {pl}")
    ax.plot(threads, threads, color=MUTED, linestyle=":", linewidth=1.2, label="ideal S(p) = p")
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    ax.set_ylabel("Speedup S(p) (bars: T_seq / Q3 to T_seq / Q1)")
    thread_axis(ax, threads, logical, physical)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=3.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "placement.png", dpi=200)
    plt.close(fig)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    FIGURES.mkdir(exist_ok=True)
    raw = pd.read_csv(RESULTS / "raw_times.csv")
    runs = pd.read_csv(RESULTS / "runs.csv")
    with open(RESULTS / "environment.json", encoding="utf-8") as f:
        env = json.load(f)
    logical, physical = env["logical_processors"], env["physical_cores"]
    threads = CONFIG["threads"]

    s = summarise(raw)
    s = s.merge(runs[KEYS + ["affinity", "peak_working_set_bytes", "jvm_live_threads", "status"]], on=KEYS, how="left")
    s["peak_working_set_mb"] = s.peak_working_set_bytes / 1e6
    s.to_csv(RESULTS / "summary.csv", index=False)
    sp = speedups(s)
    sp[["placement", "threads", "variant", "n", "median_s", "q1_s", "q3_s", "t_seq_s", "speedup", "efficiency"]].to_csv(
        RESULTS / "speedup.csv", index=False)

    fig_time(s, threads, logical, physical)
    fig_speedup(sp, threads, logical, physical)
    fig_vectorisation(s, threads, logical, physical)
    fig_placement(sp, threads, logical, physical)

    out = ["# Assignment 3 summary", "",
           "Times per multiplication, median of the measured samples; speedup and efficiency from medians.", ""]
    for pl in CONFIG["placements"]:
        for v in ("vec", "novec"):
            part = s[(s.placement == pl) & (s.variant == v)]
            if part.empty:
                continue
            out += [f"## Placement: {pl}, variant: {v}", ""]
            rows = []
            for n in sorted(part.n.unique()):
                t_seq = part[(part.method == "seq") & (part.n == n)].median_s.iloc[0]
                row = {"n": n, "seq (s)": f"{t_seq:.4g}"}
                for p in threads:
                    d = sp[(sp.placement == pl) & (sp.variant == v) & (sp.n == n) & (sp.threads == p)]
                    if not d.empty:
                        row[f"p={p}"] = (f"{d.median_s.iloc[0]:.4g} s, S={d.speedup.iloc[0]:.2f}, "
                                         f"E={d.efficiency.iloc[0]:.2f}")
                rows.append(row)
            out += [md_table(pd.DataFrame(rows)), ""]
    out += [f"## Overhead of the parallel method with one thread ({MAIN}, vec)", ""]
    rows = []
    base = s[(s.placement == MAIN) & (s.variant == "vec")]
    for n in CONFIG["sizes"]:
        t_seq = base[(base.method == "seq") & (base.n == n)].median_s.iloc[0]
        t1 = base[(base.method == "par") & (base.n == n) & (base.threads == 1)].median_s.iloc[0]
        rows.append({"n": n, "seq (s)": f"{t_seq:.4g}", "par p=1 (s)": f"{t1:.4g}", "par1/seq": f"{t1 / t_seq:.3f}"})
    out += [md_table(pd.DataFrame(rows)), ""]
    out += [f"## Peak working set (MB), {MAIN}, vec", ""]
    piv = base.assign(cfg=lambda d: d.method.where(d.method == "seq", "p=" + d.threads.astype(str)))
    piv = piv.pivot_table(index="n", columns="cfg", values="peak_working_set_mb", sort=False).round(1).reset_index()
    out += [md_table(piv), ""]
    out += ["## Variability: largest relative IQR (%) per placement and variant", ""]
    out += [md_table(s.groupby(["placement", "variant"]).rel_iqr_pct.max().round(2).reset_index()), ""]
    (RESULTS / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

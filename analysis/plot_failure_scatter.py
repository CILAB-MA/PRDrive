"""Scatter of failed maps: mean return vs fail fraction and veh+off-road load."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pufferlib.ocean.benchmark.structure import classify_structure, failed_cars

MARKER_SIZE = 36.0
CATEGORY_STYLE = {
    "localized_failure": {
        "color": "#ff7f0e",
        "label": "localized failure\nexactly 1 complete-fail car",
    },
    "globally_difficult": {
        "color": "#17becf",
        "label": "globally difficult\n2+ complete-fail cars",
    },
    "many_fail": {
        "color": "#d62728",
        "label": "many-fail\nreturn < 0 (more hits than cars)",
    },
}


def load_seed_rows(result_dir):
    paths = sorted(glob.glob(os.path.join(result_dir, "puffer_drive_*/per_scenario.csv")))
    if not paths:
        raise SystemExit(f"no puffer_drive_*/per_scenario.csv under {result_dir}")
    rows = []
    for path in paths:
        table = pd.read_csv(path)
        table["checkpoint"] = os.path.basename(os.path.dirname(path))
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def failure_table(raw, mode="self_play"):
    table = raw[raw["mode"] == mode].copy() if "mode" in raw.columns else raw.copy()
    table["failed_agents"] = table.apply(failed_cars, axis=1).round()
    table["category"] = table.apply(classify_structure, axis=1)
    fail = table[table["failed_agents"] > 0].copy()
    if fail.empty:
        return fail
    fail["agents_per_map"] = fail["n"]
    fail["fail_rate"] = fail["failed_agents"] / fail["agents_per_map"]
    fail["veh_offroad_events"] = fail["collisions_per_agent"] * fail["n"] + fail["offroad_per_agent"] * fail["n"]
    fail["veh_offroad_rate"] = fail["veh_offroad_events"] / fail["agents_per_map"]
    fail["worst_agent_hits"] = np.where(
        fail["failed_agents"] <= 1,
        fail["veh_offroad_events"],
        np.maximum(fail["veh_offroad_events"] - (fail["failed_agents"] - 1), 1.0),
    )
    return fail


def write_failure_scatter(raw, out_dir, exp_name, mode="self_play"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fail = failure_table(raw, mode=mode)
    if fail.empty:
        return None

    hits_vmax = max(float(np.percentile(fail["worst_agent_hits"], 98)), 3.0)
    n_min = float(fail["agents_per_map"].min())
    n_max = float(fail["agents_per_map"].max())
    xlim = (min(-3.8, float(fail["episode_return"].min()) - 0.1), 1.08)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), dpi=140)
    fig.subplots_adjust(wspace=0.32, top=0.80, bottom=0.16, left=0.05, right=0.98)

    sc_left = axes[0].scatter(
        fail["episode_return"],
        fail["fail_rate"],
        c=fail["worst_agent_hits"],
        cmap=plt.cm.viridis,
        norm=Normalize(vmin=1.0, vmax=hits_vmax),
        s=MARKER_SIZE,
        alpha=0.75,
        linewidths=0,
        zorder=3,
    )
    axes[0].set_ylabel("failed agents / agents per map")
    axes[0].set_title("Color: hits on the worst agent")
    axes[0].set_ylim(bottom=0)
    cbar_left = fig.colorbar(sc_left, ax=axes[0], pad=0.02)
    cbar_left.set_label("hits on the worst agent")

    mid = axes[1]
    for cat, style in CATEGORY_STYLE.items():
        part = fail[fail["category"] == cat]
        if part.empty:
            continue
        mid.scatter(
            part["episode_return"],
            part["fail_rate"],
            c=style["color"],
            s=MARKER_SIZE,
            alpha=0.78,
            linewidths=0,
            zorder=3,
            label=style["label"],
        )
    mid.set_ylabel("failed agents / agents per map")
    mid.set_title("Label: 1-car vs 2+ cars")
    mid.set_ylim(bottom=0)
    mid.legend(loc="upper left", frameon=True, fontsize=8, borderpad=0.6, handletextpad=0.5)

    sc_right = axes[2].scatter(
        fail["episode_return"],
        fail["worst_agent_hits"],
        c=fail["agents_per_map"],
        cmap=plt.cm.plasma,
        norm=Normalize(vmin=n_min, vmax=n_max),
        s=MARKER_SIZE,
        alpha=0.75,
        linewidths=0,
        zorder=3,
    )
    axes[2].set_ylabel("hits on the worst agent")
    axes[2].set_title("Same local hits, diluted by n")
    axes[2].set_ylim(0.4, hits_vmax + 0.6)
    cbar_right = fig.colorbar(sc_right, ax=axes[2], pad=0.02)
    cbar_right.set_label("# of agents per map")

    for ax in axes:
        ax.axvline(1.0, color="#2ca02c", lw=1, ls="--", alpha=0.7, zorder=1)
        ax.set_xlabel("map mean return")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.28)
        ax.set_axisbelow(True)

    fig.suptitle(f"Statistics with Failure case of ({exp_name})", fontsize=12)
    path = os.path.join(out_dir, "return_accident_scatter.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Failure-case return scatter")
    parser.add_argument("--result-dir", default="/data/prdrive/results/plr_10k")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--mode", default="self_play", choices=["self_play", "human_replay"])
    args = parser.parse_args()
    exp_name = args.exp_name or os.path.basename(os.path.abspath(args.result_dir))
    raw = load_seed_rows(args.result_dir)
    path = write_failure_scatter(raw, args.result_dir, exp_name, mode=args.mode)
    if path is None:
        raise SystemExit("no failed scenes to plot")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

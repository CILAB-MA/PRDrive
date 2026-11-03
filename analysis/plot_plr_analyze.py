
from __future__ import annotations

import argparse
import os

import numpy as np

CATEGORIES = (
    ("solved", "solved", "#4daf4a"),
    ("localized_failure", "localized failure (1 car)", "#ff7f0e"),
    ("globally_difficult", "globally difficult (2+ cars)", "#17becf"),
    ("many_fail", "many-fail (return < 0)", "#d62728"),
)

MEAN_KEYS = {cat: f"analyze/structure/plr_mean_{cat}" for cat, _, _ in CATEGORIES}
N_KEYS = {cat: f"analyze/structure/{cat}_n" for cat, _, _ in CATEGORIES}
WEIGHT_KEYS = {cat: f"analyze/structure/plr_weight_{cat}" for cat, _, _ in CATEGORIES}
TAIL_MEAN = "analyze/lowk/plr_mean"
TAIL_MIN = "analyze/lowk/plr_min"
TAIL_MAX = "analyze/lowk/plr_max"
EPOCH = "epoch"


def _history_keys():
    keys = [EPOCH, TAIL_MEAN, TAIL_MIN, TAIL_MAX, "analyze/maps", "analyze/lowk/k"]
    keys.extend(MEAN_KEYS.values())
    keys.extend(N_KEYS.values())
    keys.extend(WEIGHT_KEYS.values())
    return keys


def fetch_runs(entity, project, group=None, run_ids=None):
    import wandb

    api = wandb.Api()
    if run_ids:
        return [api.run(f"{entity}/{project}/{rid.strip()}") for rid in run_ids]
    filters = {"group": group} if group else None
    return list(api.runs(f"{entity}/{project}", filters=filters, order="-created_at"))


def run_frame(run):
    import pandas as pd

    keys = _history_keys()
    df = run.history(keys=keys, pandas=True, samples=20000)
    if df is None or df.empty or EPOCH not in df.columns:
        return pd.DataFrame()
    df = df[df[EPOCH].notna()].copy()
    if TAIL_MEAN in df.columns:
        df = df[df[TAIL_MEAN].notna()]
    df = df.sort_values(EPOCH)
    cfg = run.config or {}
    train = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    df["run_id"] = run.id
    df["run_name"] = run.name
    df["seed"] = train.get("seed")
    df["map_score"] = env.get("map_score")
    df["state"] = run.state
    return df


def _xy(df, mean_key, n_key):
    if mean_key not in df.columns:
        return np.array([]), np.array([])
    n = df[n_key] if n_key in df.columns else None
    mask = df[mean_key].notna()
    if n is not None:
        mask = mask & n.notna() & (n > 0)
    return df.loc[mask, EPOCH].to_numpy(), df.loc[mask, mean_key].to_numpy()


def _plot_mean_lines(ax, df, band=True, legend=True):
    epochs = df[EPOCH].to_numpy()
    if band and TAIL_MIN in df.columns and TAIL_MAX in df.columns:
        ax.fill_between(
            epochs,
            df[TAIL_MIN].to_numpy(),
            df[TAIL_MAX].to_numpy(),
            color="#bbbbbb",
            alpha=0.35,
            label="hard-k min–max" if legend else None,
            zorder=0,
        )
    if TAIL_MEAN in df.columns:
        ax.plot(
            epochs,
            df[TAIL_MEAN].to_numpy(),
            color="#444444",
            ls="--",
            lw=1.4,
            label="hard-k mean" if legend else None,
            zorder=1,
        )
    for i, (cat, label, color) in enumerate(CATEGORIES):
        x, y = _xy(df, MEAN_KEYS[cat], N_KEYS[cat])
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            color=color,
            lw=2.0,
            marker="o",
            ms=4.5,
            label=label if legend else None,
            zorder=2 + i,
        )


def plot_run(ax_mean, ax_zoom, ax_n, df, title):
    _plot_mean_lines(ax_mean, df, band=True, legend=True)
    ax_mean.set_ylabel("stored MaxMC")
    ax_mean.set_title(title)
    ax_mean.grid(True, alpha=0.28)
    ax_mean.set_axisbelow(True)
    ax_mean.legend(loc="upper right", fontsize=8, framealpha=0.92)

    late = df[df[EPOCH] >= 1000]
    _plot_mean_lines(ax_zoom, late if not late.empty else df, band=True, legend=False)
    ys = []
    for cat, _, _ in CATEGORIES:
        _, y = _xy(late if not late.empty else df, MEAN_KEYS[cat], N_KEYS[cat])
        ys.append(y)
    if TAIL_MEAN in (late if not late.empty else df).columns:
        ys.append((late if not late.empty else df)[TAIL_MEAN].dropna().to_numpy())
    flat = np.concatenate([y for y in ys if y.size]) if ys else np.array([0.9, 1.1])
    pad = 0.02
    ax_zoom.set_ylim(float(np.min(flat) - pad), float(np.max(flat) + pad))
    ax_zoom.set_ylabel("stored MaxMC (zoom)")
    ax_zoom.grid(True, alpha=0.28)
    ax_zoom.set_axisbelow(True)

    epochs = df[EPOCH].to_numpy()
    bottom = np.zeros(len(df))
    for cat, label, color in CATEGORIES:
        n = df[N_KEYS[cat]].fillna(0).to_numpy() if N_KEYS[cat] in df.columns else np.zeros(len(df))
        ax_n.bar(epochs, n, bottom=bottom, color=color, width=120, label=label, align="center")
        bottom = bottom + n
    ax_n.axhline(200, color="#444444", ls="--", lw=1.1, label="queue k=200")
    ax_n.set_xlabel("epoch")
    ax_n.set_ylabel("maps logged")
    ax_n.set_ylim(0, 220)
    ax_n.grid(True, axis="y", alpha=0.28)
    ax_n.set_axisbelow(True)


def write_plots(frames, out_dir, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise SystemExit("no analyze/lowk/plr_mean history on those runs")

    n = len(frames)
    fig, axes = plt.subplots(3, n, figsize=(6.2 * n, 9.6), dpi=140, sharey=False, squeeze=False)
    for col, df in enumerate(frames):
        seed = df["seed"].iloc[0]
        name = df["run_name"].iloc[0]
        label = f"seed {seed} · {name}" if seed is not None else name
        plot_run(axes[0, col], axes[1, col], axes[2, col], df, label)
        if col > 0:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")
            axes[2, col].set_ylabel("")
            legend = axes[0, col].get_legend()
            if legend is not None:
                legend.remove()
        else:
            axes[2, col].legend(loc="upper right", fontsize=8, framealpha=0.92)

    fig.suptitle(
        title
        + "\nStored PLR score of the live hard-k maps, grouped by eval structure. "
        "Bottom bars are maps that produced n>0 logs, not the k=200 queue. "
        "C packs until the agent budget, so later (larger) maps log fewer scenes.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(out_dir, "plr_mean_by_structure.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    styles = ["-", "--", ":"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=140)
    for i, df in enumerate(frames):
        ls = styles[i % len(styles)]
        seed = df["seed"].iloc[0]
        for cat, label, color in CATEGORIES:
            x, y = _xy(df, MEAN_KEYS[cat], N_KEYS[cat])
            if x.size == 0:
                continue
            ax.plot(
                x,
                y,
                color=color,
                ls=ls,
                lw=1.8,
                marker="o",
                ms=3.5,
                label=label if i == 0 else None,
            )
        ax.plot([], [], color="#444444", ls=ls, label=f"seed {seed}" if seed is not None else df["run_name"].iloc[0])
    ax.set_xlabel("epoch")
    ax.set_ylabel("stored MaxMC")
    ax.set_title("All seeds · MaxMC by structure (empty-n dropped)")
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, framealpha=0.92)
    overlay = os.path.join(out_dir, "plr_mean_by_structure_overlay.png")
    fig.savefig(overlay, bbox_inches="tight")
    plt.close(fig)
    return path, overlay


def main():
    parser = argparse.ArgumentParser(description="Plot wandb analyze/ PLR means by structure")
    parser.add_argument("--entity", default="cilab-ma")
    parser.add_argument("--project", default="scp_drive")
    parser.add_argument("--group", default="maxmc")
    parser.add_argument("--run-ids", default=None, help="comma-separated wandb run ids (overrides --group)")
    parser.add_argument("--out", default="/data/prdrive/results/maxmc_10k")
    args = parser.parse_args()
    run_ids = [r for r in (args.run_ids or "").split(",") if r.strip()] or None
    runs = fetch_runs(args.entity, args.project, group=None if run_ids else args.group, run_ids=run_ids)
    if not runs:
        raise SystemExit("no wandb runs matched")
    frames = []
    for run in reversed(list(runs)):
        df = run_frame(run)
        if df.empty:
            print(f"skip {run.name} ({run.id}): no analyze PLR history")
            continue
        print(f"{run.name} seed={df['seed'].iloc[0]} evals={len(df)} state={run.state}")
        frames.append(df)
    path, overlay = write_plots(frames, args.out, f"{args.entity}/{args.project} · {args.group}")
    print(f"wrote {path}")
    print(f"wrote {overlay}")


if __name__ == "__main__":
    main()

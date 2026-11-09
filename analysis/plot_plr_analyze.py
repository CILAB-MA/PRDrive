
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pufferlib.ocean.benchmark.structure import (
    FAIL_COLLISION_EMA,
    FAIL_STRUCTURAL_TOP1,
    MIN_OUTCOME_EPISODES,
    classify_structure,
    fail_top1_share,
    fail_worst_stat,
)

CATEGORIES = (
    ("solved", "solved", "#4daf4a"),
    ("localized_failure", "localized failure (1 car)", "#ff7f0e"),
    ("globally_suboptimal", "globally suboptimal (2+ cars)", "#17becf"),
    ("many_fail", "many-fail (return < 0)", "#d62728"),
)
LF_KEYS = (
    ("analyze/structure/lf_structural_share", "LF structural", "#ff7f0e"),
    ("analyze/structure/lf_oneoff_share", "LF one-off", "#c7a36a"),
    ("analyze/structure/lf_crash_share", "LF collision", "#d62728"),
)

N_KEYS = {cat: f"analyze/structure/{cat}_n" for cat, _, _ in CATEGORIES}
SHARE_KEYS = {cat: f"analyze/structure/{cat}_share" for cat, _, _ in CATEGORIES}
GOAL_RATE = "analyze/success/rate"
NO_COLLISION = "analyze/failure/rate"
EPOCH = "epoch"
LEGACY = "globally_difficult"


def _history_keys(legacy=False):
    keys = [EPOCH, GOAL_RATE, NO_COLLISION]
    n_keys = list(N_KEYS.values())
    s_keys = list(SHARE_KEYS.values())
    if legacy:
        n_keys = [k.replace("globally_suboptimal", LEGACY) for k in n_keys]
        s_keys = [k.replace("globally_suboptimal", LEGACY) for k in s_keys]
    keys.extend(n_keys)
    keys.extend(s_keys)
    return keys


def fetch_runs(entity, project, group=None, run_ids=None):
    import wandb

    api = wandb.Api()
    if run_ids:
        return [api.run(f"{entity}/{project}/{rid.strip()}") for rid in run_ids]
    filters = {"group": group} if group else None
    return list(api.runs(f"{entity}/{project}", filters=filters, order="-created_at"))


def _alias_legacy(df):
    rename = {}
    for col in df.columns:
        if isinstance(col, str) and LEGACY in col:
            new = col.replace(LEGACY, "globally_suboptimal")
            if new not in df.columns:
                rename[col] = new
    return df.rename(columns=rename) if rename else df


def run_frame(run):
    import pandas as pd

    df = run.history(keys=_history_keys(legacy=False), pandas=True, samples=20000)
    if df is None or df.empty:
        df = run.history(keys=_history_keys(legacy=True), pandas=True, samples=20000)
    if df is None or df.empty or EPOCH not in df.columns:
        return pd.DataFrame()
    df = _alias_legacy(df[df[EPOCH].notna()].copy())
    n_cols = [c for c in (N_KEYS[cat] for cat, _, _ in CATEGORIES) if c in df.columns]
    if n_cols:
        df = df[df[n_cols].notna().any(axis=1)]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(EPOCH)
    zeros = np.zeros(len(df))
    n_series = {
        cat: (df[N_KEYS[cat]].fillna(0.0).to_numpy() if N_KEYS[cat] in df.columns else zeros)
        for cat, _, _ in CATEGORIES
    }
    total = np.sum([n_series[cat] for cat, _, _ in CATEGORIES], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        computed = {cat: np.divide(n_series[cat], total) for cat, _, _ in CATEGORIES}
    for cat, _, _ in CATEGORIES:
        s_key = SHARE_KEYS[cat]
        if s_key not in df.columns:
            df[s_key] = computed[cat]
        else:
            need = df[s_key].isna().to_numpy()
            values = np.array(df[s_key].to_numpy(), dtype=float, copy=True)
            values[need] = computed[cat][need]
            df[s_key] = values
    cfg = run.config or {}
    train = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    df["run_id"] = run.id
    df["run_name"] = run.name
    df["seed"] = train.get("seed")
    df["map_score"] = env.get("map_score")
    df["state"] = run.state
    return df


def _hardk_ids(scores, seen, k=200):
    ids = np.flatnonzero(np.asarray(seen, dtype=bool))
    if ids.size == 0:
        return ids
    k = min(int(k), int(ids.size))
    order = np.argsort(np.asarray(scores, dtype=np.float64)[ids], kind="mergesort")
    return ids[order[-k:]]


def _checkpoint_row(path, k=200):
    with np.load(path) as data:
        epoch = int(np.asarray(data["epoch"]).reshape(-1)[0])
        scores = np.asarray(data["scores"], dtype=np.float64)
        seen = np.asarray(data["seen"], dtype=bool)
        fail = np.asarray(data["fail_ema"], dtype=np.float64)
        coll = np.asarray(data["coll_ema"], dtype=np.float64)
        n = np.asarray(data["outcome_n"], dtype=np.float64)
    ids = _hardk_ids(scores, seen, k=k)
    counts = {cat: 0 for cat, _, _ in CATEGORIES}
    structural = oneoff = crash = 0
    goal, nocoll, cars = [], [], 0
    for idx in ids:
        visits = n[idx]
        valid = visits >= float(MIN_OUTCOME_EPISODES)
        if not valid.any():
            continue
        cat = classify_structure({}, fail_ema=fail[idx], outcome_n=n[idx])
        counts[cat] += 1
        if cat == "localized_failure":
            top1 = fail_top1_share(fail[idx], n[idx])
            if top1 is not None:
                if top1 >= FAIL_STRUCTURAL_TOP1:
                    structural += 1
                else:
                    oneoff += 1
            worst = fail_worst_stat(fail[idx], coll[idx], n[idx])
            if worst is not None and worst >= FAIL_COLLISION_EMA:
                crash += 1
        fail_v = fail[idx][valid]
        coll_v = coll[idx][valid]
        goal.extend((1.0 - fail_v).tolist())
        nocoll.extend((1.0 - coll_v).tolist())
        cars += int(valid.sum())
    labeled = float(sum(counts.values())) or 1.0
    row = {EPOCH: epoch, "analyze/structure/maps": labeled}
    for cat, _, _ in CATEGORIES:
        row[N_KEYS[cat]] = float(counts[cat])
        row[SHARE_KEYS[cat]] = float(counts[cat]) / labeled
    row["analyze/structure/lf_structural_share"] = structural / labeled
    row["analyze/structure/lf_oneoff_share"] = oneoff / labeled
    row["analyze/structure/lf_crash_share"] = crash / labeled
    row[GOAL_RATE] = float(np.mean(goal)) if goal else np.nan
    row[NO_COLLISION] = float(np.mean(nocoll)) if nocoll else np.nan
    row["analyze/structure/cars"] = float(cars)
    return row


def frame_from_plr_dir(plr_dir, k=200, map_score=None, seed=None):
    import pandas as pd

    paths = sorted(
        p
        for p in glob.glob(os.path.join(plr_dir, "plr_state_*.npz"))
        if os.path.basename(p) != "plr_state.latest.npz"
    )
    if not paths:
        return pd.DataFrame()
    rows = [_checkpoint_row(p, k=k) for p in paths]
    df = pd.DataFrame(rows).sort_values(EPOCH)
    name = os.path.basename(os.path.abspath(plr_dir))
    parent = os.path.basename(os.path.dirname(os.path.abspath(plr_dir)))
    df["run_id"] = name
    df["run_name"] = parent or name
    df["seed"] = seed
    df["map_score"] = map_score or parent
    df["state"] = "checkpoint"
    return df


def _xy(df, key):
    if key not in df.columns:
        return np.array([]), np.array([])
    mask = df[key].notna()
    return df.loc[mask, EPOCH].to_numpy(), df.loc[mask, key].to_numpy()


def _plot_share_lines(ax, df, legend=True):
    for cat, label, color in CATEGORIES:
        x, y = _xy(df, SHARE_KEYS[cat])
        if x.size == 0 or not np.isfinite(y).any() or np.nanmax(y) <= 0:
            continue
        ax.plot(x, 100.0 * y, color=color, lw=2.0, marker="o", ms=4.5, label=label if legend else None)
    ax.set_ylabel("% of hard-k maps")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)


def _plot_lf_severity(ax, df, legend=True):
    for key, label, color in LF_KEYS:
        x, y = _xy(df, key)
        if x.size == 0:
            continue
        ax.plot(x, 100.0 * y, color=color, lw=2.0, marker="o", ms=4.5, label=label if legend else None)
    ax.set_ylabel("% of hard-k maps")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)


def _plot_goal_vs_collision(ax, df, legend=True):
    x_g, goal = _xy(df, GOAL_RATE)
    x_c, nocoll = _xy(df, NO_COLLISION)
    if x_g.size:
        ax.plot(
            x_g,
            100.0 * goal,
            color="#4daf4a",
            lw=2.0,
            marker="o",
            ms=4.5,
            label="goal success (1 - fail EMA)" if legend else None,
        )
    if x_c.size:
        ax.plot(
            x_c,
            100.0 * nocoll,
            color="#6a3d9a",
            lw=2.0,
            marker="s",
            ms=4.0,
            label="no collision (1 - coll EMA)" if legend else None,
        )
    ys = [y for y in (goal, nocoll) if np.size(y)]
    lo = 90.0
    if ys:
        lo = max(0.0, min(90.0, 100.0 * min(np.min(y) for y in ys) - 1.5))
    ax.set_ylabel("% of cars")
    ax.set_xlabel("epoch")
    ax.set_ylim(lo, 100.4)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)


def plot_run(ax_share, ax_goal, df, title):
    _plot_share_lines(ax_share, df, legend=True)
    ax_share.set_title(title)
    if any(k in df.columns for k, _, _ in LF_KEYS):
        _plot_lf_severity(ax_goal, df, legend=True)
    else:
        _plot_goal_vs_collision(ax_goal, df, legend=True)


def _longest_per_seed(frames):
    by_seed = {}
    leftover = []
    for df in frames:
        seed = df["seed"].iloc[0]
        if seed is None:
            leftover.append(df)
            continue
        prev = by_seed.get(seed)
        if prev is None or df[EPOCH].max() > prev[EPOCH].max():
            by_seed[seed] = df
    ordered = []
    for seed in (11, 3, 42):
        if seed in by_seed:
            ordered.append(by_seed.pop(seed))
    ordered.extend(by_seed[s] for s in sorted(by_seed, key=lambda x: (x is None, x)))
    return ordered + leftover


def write_plots(frames, out_dir, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    keep_all = all(df["seed"].isna().all() if "seed" in df.columns else True for df in frames)
    if keep_all:
        frames = [f for f in frames if f is not None and not f.empty]
    else:
        frames = _longest_per_seed([f for f in frames if f is not None and not f.empty])
    if not frames:
        raise SystemExit("no analyze/structure history on those runs")

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 7.2), dpi=140, sharey=False, squeeze=False)
    for col, df in enumerate(frames):
        seed = df["seed"].iloc[0] if "seed" in df.columns else None
        name = df["run_name"].iloc[0]
        label = f"{name}" if seed is None else f"seed {seed} · {name}"
        plot_run(axes[0, col], axes[1, col], df, label)
        if col > 0:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")
            for ax in (axes[0, col], axes[1, col]):
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    fig.suptitle(
        title
        + "\nTop: hard-k map share (fail EMA). Solved / LF / GS."
        " Bottom: LF split by severity (structural vs one-off vs leftover collision), "
        "or goal vs no-collision if those keys are missing.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = os.path.join(out_dir, "plr_mean_by_structure.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    styles = ["-", "--", ":"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=140)
    for i, df in enumerate(frames):
        ls = styles[i % len(styles)]
        name = df["run_name"].iloc[0]
        for cat, label, color in CATEGORIES:
            x, y = _xy(df, SHARE_KEYS[cat])
            if x.size == 0 or not np.isfinite(y).any() or np.nanmax(y) <= 0:
                continue
            ax.plot(
                x,
                100.0 * y,
                color=color,
                ls=ls,
                lw=1.8,
                marker="o",
                ms=3.5,
                label=label if i == 0 else None,
            )
        ax.plot([], [], color="#444444", ls=ls, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("% of hard-k maps")
    ax.set_title("Structure share of the hard-k window")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, framealpha=0.92)
    overlay = os.path.join(out_dir, "plr_mean_by_structure_overlay.png")
    fig.savefig(overlay, bbox_inches="tight")
    plt.close(fig)
    return path, overlay


def main():
    parser = argparse.ArgumentParser(description="Plot structure share over epoch from PLR checkpoints or wandb")
    parser.add_argument("--entity", default="cilab-ma")
    parser.add_argument("--project", default="scp_drive")
    parser.add_argument("--group", default="maxmc")
    parser.add_argument("--run-ids", default=None, help="comma-separated wandb run ids (overrides --group)")
    parser.add_argument(
        "--plr-dirs",
        default=None,
        help="comma-separated run dirs with plr_state_*.npz (skips wandb)",
    )
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--out", default="/data/prdrive/results/maxmc_10k")
    args = parser.parse_args()

    frames = []
    if args.plr_dirs:
        for raw in args.plr_dirs.split(","):
            path = raw.strip()
            if not path:
                continue
            df = frame_from_plr_dir(path, k=args.k)
            if df.empty:
                print(f"skip {path}: no plr_state_*.npz")
                continue
            print(f"{df['run_name'].iloc[0]} checkpoints={len(df)} epoch={int(df[EPOCH].min())}-{int(df[EPOCH].max())}")
            frames.append(df)
        title = "PLR checkpoints · hard-k fail EMA"
    else:
        run_ids = [r for r in (args.run_ids or "").split(",") if r.strip()] or None
        runs = fetch_runs(args.entity, args.project, group=None if run_ids else args.group, run_ids=run_ids)
        if not runs:
            raise SystemExit("no wandb runs matched")
        for run in reversed(list(runs)):
            df = run_frame(run)
            if df.empty:
                print(f"skip {run.name} ({run.id}): no analyze structure history")
                continue
            print(f"{run.name} seed={df['seed'].iloc[0]} evals={len(df)} state={run.state} epoch={int(df[EPOCH].max())}")
            frames.append(df)
        title = f"{args.entity}/{args.project} · {args.group}"
    path, overlay = write_plots(frames, args.out, title)
    print(f"wrote {path}")
    print(f"wrote {overlay}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from per_scenario import CATEGORIES, classify_structure
from pufferlib.ocean.benchmark.structure import majority_structure

MIN_SCENES_FOR_SEED = 1000
HARM_LENSES = (
    ("success", "failed_cars", "failed cars"),
    ("collision", "collided_cars", "collided cars"),
    ("return", "return_gap", "return gap (car-units)"),
)


def load_seed_tables(result_dir):
    rows = []
    skipped = []
    for csv_path in sorted(glob.glob(os.path.join(result_dir, "puffer_drive_*/per_scenario.csv"))):
        table = pd.read_csv(csv_path)
        stem = os.path.basename(os.path.dirname(csv_path))
        if len(table) < MIN_SCENES_FOR_SEED:
            skipped.append((stem, len(table)))
            continue
        table["checkpoint"] = stem
        rows.append(table)
    if not rows:
        raise SystemExit(f"no 10k per_scenario.csv under {result_dir}")
    return pd.concat(rows, ignore_index=True), skipped


def pool_scenes(raw):
    """Car-weighted mean of rates across seeds, keyed by (mode, map_id)."""
    rate_cols = [
        c
        for c in (
            "score",
            "collision_rate",
            "at_fault_collision_rate",
            "offroad_rate",
            "episode_return",
            "completion_rate",
            "collisions_per_agent",
            "rear_collision_rate",
        )
        if c in raw.columns
    ]

    raw = raw.copy()
    if "category" not in raw.columns:
        raw["category"] = raw.apply(
            lambda row: classify_structure(
                {"n": row["n"], "score": row["score"], "episode_return": row.get("episode_return")}
            ),
            axis=1,
        )

    def _agg(group):
        n_sum = float(group["n"].sum())
        n_mean = float(group["n"].mean())
        out = {
            "n": n_mean,
            "n_sum": n_sum,
            "n_seeds": int(len(group)),
            "category": majority_structure(group["category"].tolist()),
        }
        for col in rate_cols:
            out[col] = float((group[col] * group["n"]).sum() / n_sum) if n_sum else 0.0
        sid = group["scenario_id"].dropna().astype(str)
        out["scenario_id"] = sid.iloc[0] if len(sid) else ""
        return pd.Series(out)

    pooled = raw.groupby(["mode", "map_id"], as_index=False).apply(_agg, include_groups=False)
    pooled["safety_rate"] = 1.0 - pooled["collision_rate"]
    # Harm is mean-n (typical seed), not n summed across seeds.
    pooled["failed_cars"] = (1.0 - pooled["score"]) * pooled["n"]
    pooled["collided_cars"] = pooled["collision_rate"] * pooled["n"]
    pooled["offroad_cars"] = pooled["offroad_rate"] * pooled["n"]
    pooled["at_fault_cars"] = pooled["at_fault_collision_rate"] * pooled["n"]
    pooled["return_gap"] = (1.0 - pooled["episode_return"]) * pooled["n"]
    return pooled


def summarize_harm(table, harm_col, lens):
    total_harm = float(table[harm_col].sum())
    rows = []
    for cat in CATEGORIES:
        part = table[table["category"] == cat]
        harm = float(part[harm_col].sum()) if not part.empty else 0.0
        rows.append(
            {
                "lens": lens,
                "category": cat,
                "scenarios": int(len(part)),
                "cars": float(part["n"].sum()) if not part.empty else 0.0,
                "harm": harm,
                "share_of_scenarios": len(part) / len(table) if len(table) else 0.0,
                "share_of_harm": (harm / total_harm) if total_harm > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_plots(table, summaries, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "solved": "tab:green",
        "localized_failure": "tab:orange",
        "globally_difficult": "tab:cyan",
    }
    labels = ["solved", "localized (1 car)", "global (2+ cars)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    titles = {
        "success": "Success (failed cars)",
        "collision": "Collision (collided cars)",
        "return": "Return (return gap)",
    }
    for ax, lens in zip(axes, ("success", "collision", "return")):
        part = summaries[summaries["lens"] == lens]
        ys = [100 * float(part.loc[part["category"] == c, "share_of_harm"].sum()) for c in CATEGORIES]
        ax.bar(range(len(CATEGORIES)), ys, color=[colors[c] for c in CATEGORIES])
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(titles[lens])
        ax.set_ylabel("% of harm" if ax is axes[0] else "")
        ax.set_ylim(0, max(ys + [1]) * 1.2)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Where harm sits by failure structure  (10k maps, pooled seeds)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "three_lens_harm_share.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(table, summaries, skipped, out_dir):
    n_scen = len(table)
    n_cars = float(table["n"].sum())
    n_maps = int(table["map_id"].nunique())
    n_seeds = int(table["n_seeds"].max()) if "n_seeds" in table.columns else 0
    fail = float(table["failed_cars"].sum())
    hit = float(table["collided_cars"].sum())
    off = float(table["offroad_cars"].sum())
    fault = float(table["at_fault_cars"].sum())
    gap = float(table["return_gap"].sum())
    pooled_score = float((table["score"] * table["n"]).sum() / n_cars)
    pooled_col = float((table["collision_rate"] * table["n"]).sum() / n_cars)
    pooled_ret = float((table["episode_return"] * table["n"]).sum() / n_cars)

    def _fmt(lens, harm_name):
        part = summaries[summaries["lens"] == lens]
        lines = [
            "| category | scenarios | cars | harm | % scenarios | % harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, row in part.iterrows():
            lines.append(
                f"| {row['category']} | {int(row['scenarios'])} | {row['cars']:.0f} | "
                f"{row['harm']:.1f} | {100*row['share_of_scenarios']:.1f}% | {100*row['share_of_harm']:.1f}% |"
            )
        return harm_name, "\n".join(lines)

    skip_txt = ", ".join(f"{s} ({n} rows)" for s, n in skipped) or "none"
    loc = table[table["category"] == "localized_failure"]
    mild = table[table["category"] == "globally_difficult"]
    loc_crashless = int(((loc["collision_rate"] == 0) & (loc["n"] > 0)).sum()) if not loc.empty else 0

    lines = []
    lines.append("# PLR 10k structure report")
    lines.append("")
    lines.append("Pooled car-weighted over complete 10k seeds. Labels are **failure structure**,")
    lines.append("not mean-score height:")
    lines.append("")
    lines.append("- `solved`: no complete car-fail")
    lines.append("- `localized_failure`: exactly one complete car-fail (the rest cover the mean)")
    lines.append("- `globally_difficult`: two or more complete car-fails (failure is spread)")
    lines.append("")
    lines.append("Harm is still reported three ways on **the same** structure labels.")
    lines.append("")
    lines.append(f"- Maps: {n_maps} (training binaries)")
    lines.append(f"- Pooled scenes (mode x map): {n_scen}  |  cars: {n_cars:.0f}  |  seeds used: {n_seeds}")
    lines.append(f"- Skipped incomplete seeds: {skip_txt}")
    lines.append(f"- Pooled score {pooled_score:.4f}  |  collision rate {pooled_col:.4f}  |  return {pooled_ret:.4f}")
    lines.append(
        f"- Failed cars {fail:.0f}  |  collided {hit:.0f} (at-fault {fault:.0f})  |  off-road {off:.0f}  |  return gap {gap:.0f}"
    )
    lines.append("")
    lines.append("## Process")
    lines.append("")
    lines.append("1. Load each seed `per_scenario.csv` with at least 1000 scenes.")
    lines.append("2. Label each seed row with `classify_structure`, then pool by `(mode, map_id)`.")
    lines.append("3. Pooled label is the **majority** of per-seed labels (tie → more severe).")
    lines.append("   1 fail on every seed stays localized. 2+ fails on a seed is globally_difficult.")
    lines.append("   1 fail on 1 of 3 seeds stays solved. n-sum across seeds is not used for the label.")
    lines.append("4. Rates are car-weighted. Harm uses mean n per seed.")
    lines.append("")
    for lens, _col, harm_name in HARM_LENSES:
        title = {"success": "Success", "collision": "Collision / safety", "return": "Return"}[lens]
        _, table_md = _fmt(lens, harm_name)
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Harm column is **{harm_name}**.")
        lines.append("")
        lines.append(table_md)
        lines.append("")

    lines.append("## Structure vs return (same abstract mean?)")
    lines.append("")
    if not loc.empty:
        lines.append(
            f"- Localized: {len(loc)} scenes, mean score {float((loc.score*loc.n).sum()/loc.n.sum()):.3f}, "
            f"mean return {float((loc.episode_return*loc.n).sum()/loc.n.sum()):.3f}, "
            f"mean failed cars {float(loc.failed_cars.mean()):.2f}"
        )
    else:
        lines.append("- Localized: none")
    if not mild.empty:
        lines.append(
            f"- Globally difficult (2+ cars): {len(mild)} scenes, mean score {float((mild.score*mild.n).sum()/mild.n.sum()):.3f}, "
            f"mean return {float((mild.episode_return*mild.n).sum()/mild.n.sum()):.3f}, "
            f"mean failed cars {float(mild.failed_cars.mean()):.2f}"
        )
    else:
        lines.append("- Globally difficult (2+ cars): none")
    lines.append(f"- Localized with zero vehicle collisions (off-road only): {loc_crashless}")
    lines.append("")
    lines.append("## By mode")
    lines.append("")
    for mode, part in table.groupby("mode"):
        cars = part["n"].sum()
        counts = part["category"].value_counts()
        lines.append(
            f"- **{mode}**: {len(part)} scenes, {cars:.0f} cars, "
            f"score {(part.score*part.n).sum()/cars:.4f}, "
            f"collision {(part.collision_rate*part.n).sum()/cars:.4f}, "
            f"return {(part.episode_return*part.n).sum()/cars:.4f}; "
            f"localized {int(counts.get('localized_failure', 0))}, "
            f"globally_difficult {int(counts.get('globally_difficult', 0))}"
        )
    lines.append("")
    path = os.path.join(out_dir, "three_lens_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path, {
        "n_scen": n_scen,
        "n_cars": n_cars,
        "n_maps": n_maps,
        "n_seeds": n_seeds,
        "fail": fail,
        "hit": hit,
        "off": off,
        "fault": fault,
        "gap": gap,
        "pooled_score": pooled_score,
        "pooled_col": pooled_col,
        "pooled_ret": pooled_ret,
        "n_localized": int(len(loc)),
        "n_globally_difficult": int(len(mild)),
    }


def run(result_dir):
    raw, skipped = load_seed_tables(result_dir)
    table = pool_scenes(raw)
    summaries = pd.concat(
        [summarize_harm(table, harm_col, lens) for lens, harm_col, _ in HARM_LENSES],
        ignore_index=True,
    )
    by_mode = []
    for mode, part in table.groupby("mode"):
        s = pd.concat(
            [summarize_harm(part, harm_col, lens) for lens, harm_col, _ in HARM_LENSES],
            ignore_index=True,
        )
        s.insert(0, "mode", mode)
        by_mode.append(s)
    by_mode = pd.concat(by_mode, ignore_index=True)

    table_path = os.path.join(result_dir, "three_lens_per_scenario.csv")
    sum_path = os.path.join(result_dir, "three_lens_summary.csv")
    mode_path = os.path.join(result_dir, "three_lens_by_mode.csv")
    table.to_csv(table_path, index=False)
    summaries.to_csv(sum_path, index=False)
    by_mode.to_csv(mode_path, index=False)
    plot_path = write_plots(table, summaries, result_dir)
    from plot_failure_scatter import write_failure_scatter

    exp_name = os.path.basename(os.path.abspath(result_dir))
    scatter_path = write_failure_scatter(raw, result_dir, exp_name)
    report_path, stats = write_report(table, summaries, skipped, result_dir)
    print(f"pooled {stats['n_scen']} scenes / {stats['n_maps']} maps / {stats['n_cars']:.0f} cars")
    print(f"skipped seeds: {skipped or 'none'}")
    print(summaries.to_string(index=False))
    print(f"wrote {table_path}")
    print(f"wrote {sum_path}")
    print(f"wrote {mode_path}")
    print(f"plot  {plot_path}")
    if scatter_path:
        print(f"scatter {scatter_path}")
    print(f"report {report_path}")
    return {
        "table": table,
        "summaries": summaries,
        "by_mode": by_mode,
        "stats": stats,
        "skipped": skipped,
        "paths": {
            "table": table_path,
            "summary": sum_path,
            "by_mode": mode_path,
            "plot": plot_path,
            "scatter": scatter_path,
            "report": report_path,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Three-lens 10k report (failure structure)")
    parser.add_argument("--result-dir", default="/data/prdrive/results/plr_10k")
    args = parser.parse_args()
    run(args.result_dir)


if __name__ == "__main__":
    main()

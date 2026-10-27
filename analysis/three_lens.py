
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from per_scenario import CATEGORIES, classify_success

MIN_SCENES_FOR_SEED = 1000


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

    def _agg(group):
        n = float(group["n"].sum())
        out = {"n": n, "n_seeds": int(len(group))}
        for col in rate_cols:
            out[col] = float((group[col] * group["n"]).sum() / n) if n else 0.0
        sid = group["scenario_id"].dropna().astype(str)
        out["scenario_id"] = sid.iloc[0] if len(sid) else ""
        return pd.Series(out)

    pooled = raw.groupby(["mode", "map_id"], as_index=False).apply(_agg, include_groups=False)
    pooled["safety_rate"] = 1.0 - pooled["collision_rate"]
    pooled["failed_cars"] = (1.0 - pooled["score"]) * pooled["n"]
    pooled["collided_cars"] = pooled["collision_rate"] * pooled["n"]
    pooled["offroad_cars"] = pooled["offroad_rate"] * pooled["n"]
    pooled["at_fault_cars"] = pooled["at_fault_collision_rate"] * pooled["n"]
    pooled["return_gap"] = (1.0 - pooled["episode_return"]) * pooled["n"]
    pooled["cat_success"] = pooled["score"].map(classify_success)
    pooled["cat_collision"] = pooled["safety_rate"].map(classify_success)
    pooled["cat_return"] = pooled["episode_return"].map(classify_success)
    return pooled


def summarize_lens(table, category_col, harm_col):
    total_harm = float(table[harm_col].sum())
    rows = []
    for cat in CATEGORIES:
        part = table[table[category_col] == cat]
        harm = float(part[harm_col].sum()) if not part.empty else 0.0
        rows.append(
            {
                "lens": category_col.replace("cat_", ""),
                "category": cat,
                "scenarios": int(len(part)),
                "cars": float(part["n"].sum()) if not part.empty else 0.0,
                "harm": harm,
                "share_of_scenarios": len(part) / len(table) if len(table) else 0.0,
                "share_of_harm": (harm / total_harm) if total_harm > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def crosstab_lenses(table):
    frames = []
    pairs = (("cat_success", "cat_collision"), ("cat_success", "cat_return"), ("cat_collision", "cat_return"))
    for left, right in pairs:
        ct = pd.crosstab(table[left], table[right])
        ct = ct.reindex(index=list(CATEGORIES), columns=list(CATEGORIES), fill_value=0)
        long = ct.reset_index().melt(id_vars=left, var_name="other_category", value_name="scenarios")
        long = long.rename(columns={left: "category"})
        long["row_lens"] = left.replace("cat_", "")
        long["col_lens"] = right.replace("cat_", "")
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def write_plots(table, summaries, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "solved": "tab:green",
        "localized_failure": "tab:orange",
        "intermediate": "tab:gray",
        "globally_difficult": "tab:red",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    titles = {
        "success": "Success (score)",
        "collision": "Collision (1 - collision_rate)",
        "return": "Return (episode_return)",
    }
    for ax, lens in zip(axes, ("success", "collision", "return")):
        part = summaries[summaries["lens"] == lens]
        ys = [100 * float(part.loc[part["category"] == c, "share_of_harm"].sum()) for c in CATEGORIES]
        ax.bar(range(len(CATEGORIES)), ys, color=[colors[c] for c in CATEGORIES])
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(["solved", "localized", "intermediate", "global"], rotation=20, fontsize=8)
        ax.set_title(titles[lens])
        ax.set_ylabel("% of harm" if ax is axes[0] else "")
        ax.set_ylim(0, max(ys + [1]) * 1.2)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Where harm sits under each lens  (10k maps, 2 seeds pooled)", fontsize=11)
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

    def _fmt(lens):
        part = summaries[summaries["lens"] == lens]
        lines = [
            f"| category | scenarios | cars | harm | % scenarios | % harm |",
            f"|---|---:|---:|---:|---:|---:|",
        ]
        harm_name = {"success": "failed cars", "collision": "collided cars", "return": "return gap (car-units)"}[lens]
        for _, row in part.iterrows():
            lines.append(
                f"| {row['category']} | {int(row['scenarios'])} | {row['cars']:.0f} | "
                f"{row['harm']:.1f} | {100*row['share_of_scenarios']:.1f}% | {100*row['share_of_harm']:.1f}% |"
            )
        return harm_name, "\n".join(lines)

    skip_txt = ", ".join(f"{s} ({n} rows)" for s, n in skipped) or "none"
    lines = []
    lines.append("# PLR 10k three-lens report")
    lines.append("")
    lines.append("Pooled car-weighted over complete 10k seeds. Same cutoffs on three metrics:")
    lines.append("`solved >= 0.999`, `localized_failure >= 0.8`, `intermediate` in between, `globally_difficult < 0.5`.")
    lines.append("")
    lines.append(f"- Maps: {n_maps} (training binaries)")
    lines.append(f"- Pooled scenes (mode x map): {n_scen}  |  cars: {n_cars:.0f}  |  seeds used: {n_seeds}")
    lines.append(f"- Skipped incomplete seeds: {skip_txt}")
    lines.append(f"- Pooled score {pooled_score:.4f}  |  collision rate {pooled_col:.4f}  |  return {pooled_ret:.4f}")
    lines.append(f"- Failed cars {fail:.0f}  |  collided {hit:.0f} (at-fault {fault:.0f})  |  off-road {off:.0f}  |  return gap {gap:.0f}")
    lines.append("")
    lines.append("## Process")
    lines.append("")
    lines.append("1. Load each seed `per_scenario.csv` with at least 1000 scenes.")
    lines.append("2. Pool seeds by `(mode, map_id)` with car-weighted means.")
    lines.append("3. Classify every scene three times:")
    lines.append("   - **success**: `score` (goal without collision/off-road). Harm = failed cars.")
    lines.append("   - **collision**: `1 - collision_rate`. Harm = collided cars. Off-road and no-goal do not count.")
    lines.append("   - **return**: `episode_return` unclipped. Solved scenes sit at 1.0; stacked collisions go negative. Harm = `(1 - return) * n`.")
    lines.append("4. Report scenario share vs harm share. Harm share is the quantity that says where the actual problems sit.")
    lines.append("")
    for lens, title in (("success", "Success"), ("collision", "Collision / safety"), ("return", "Return")):
        harm_name, table_md = _fmt(lens)
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Harm column is **{harm_name}**.")
        lines.append("")
        lines.append(table_md)
        lines.append("")

    lines.append("## How the lenses disagree")
    lines.append("")
    lines.append("Score failures include collision, off-road, and no-goal. Safety ignores off-road/no-goal.")
    lines.append("Return matches score at 1.0 on solved scenes, but stacked collisions (`collisions_per_agent`) pull return below 0,")
    lines.append("so some mixed scenes that look intermediate on score look globally bad on return.")
    lines.append("")
    ss, sc = table["cat_success"], table["cat_collision"]
    sr = table["cat_return"]
    lines.append(f"- Same bucket on success and safety: {(ss == sc).mean()*100:.1f}% of scenes")
    lines.append(f"- Same bucket on success and return: {(ss == sr).mean()*100:.1f}% of scenes")
    lines.append(f"- Success localized but safety solved (failures are off-road/no-goal, not crashes): {int(((ss=='localized_failure')&(sc=='solved')).sum())}")
    lines.append(f"- Success intermediate but return globally_difficult (stacked penalties): {int(((ss=='intermediate')&(sr=='globally_difficult')).sum())}")
    lines.append("")
    lines.append("## By mode")
    lines.append("")
    for mode, part in table.groupby("mode"):
        cars = part["n"].sum()
        lines.append(
            f"- **{mode}**: {len(part)} scenes, {cars:.0f} cars, "
            f"score {(part.score*part.n).sum()/cars:.4f}, "
            f"collision {(part.collision_rate*part.n).sum()/cars:.4f}, "
            f"return {(part.episode_return*part.n).sum()/cars:.4f}"
        )
    lines.append("")
    lines.append("Self-play is close to solved on all three lenses. Residual harm is concentrated in a few cars inside otherwise-good maps (localized), not in globally unsolvable maps.")
    lines.append("Human-replay has more globally_difficult scenes; SDC-only with log replay of others is the harder setting.")
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
        "agree_ss_sc": float((ss == sc).mean()),
        "agree_ss_sr": float((ss == sr).mean()),
    }


def run(result_dir):
    raw, skipped = load_seed_tables(result_dir)
    table = pool_scenes(raw)
    summaries = pd.concat(
        [
            summarize_lens(table, "cat_success", "failed_cars"),
            summarize_lens(table, "cat_collision", "collided_cars"),
            summarize_lens(table, "cat_return", "return_gap"),
        ],
        ignore_index=True,
    )
    by_mode = []
    for mode, part in table.groupby("mode"):
        s = pd.concat(
            [
                summarize_lens(part, "cat_success", "failed_cars"),
                summarize_lens(part, "cat_collision", "collided_cars"),
                summarize_lens(part, "cat_return", "return_gap"),
            ],
            ignore_index=True,
        )
        s.insert(0, "mode", mode)
        by_mode.append(s)
    by_mode = pd.concat(by_mode, ignore_index=True)
    xtab = crosstab_lenses(table)

    table_path = os.path.join(result_dir, "three_lens_per_scenario.csv")
    sum_path = os.path.join(result_dir, "three_lens_summary.csv")
    mode_path = os.path.join(result_dir, "three_lens_by_mode.csv")
    xtab_path = os.path.join(result_dir, "three_lens_crosstab.csv")
    table.to_csv(table_path, index=False)
    summaries.to_csv(sum_path, index=False)
    by_mode.to_csv(mode_path, index=False)
    xtab.to_csv(xtab_path, index=False)
    plot_path = write_plots(table, summaries, result_dir)
    report_path, stats = write_report(table, summaries, skipped, result_dir)
    print(f"pooled {stats['n_scen']} scenes / {stats['n_maps']} maps / {stats['n_cars']:.0f} cars")
    print(f"skipped seeds: {skipped or 'none'}")
    print(summaries.to_string(index=False))
    print(f"wrote {table_path}")
    print(f"wrote {sum_path}")
    print(f"wrote {mode_path}")
    print(f"wrote {xtab_path}")
    print(f"plot  {plot_path}")
    print(f"report {report_path}")
    return {
        "table": table,
        "summaries": summaries,
        "by_mode": by_mode,
        "xtab": xtab,
        "stats": stats,
        "skipped": skipped,
        "paths": {
            "table": table_path,
            "summary": sum_path,
            "by_mode": mode_path,
            "crosstab": xtab_path,
            "plot": plot_path,
            "report": report_path,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Three-lens 10k report")
    parser.add_argument("--result-dir", default="/data/prdrive/results/plr_10k")
    args = parser.parse_args()
    run(args.result_dir)


if __name__ == "__main__":
    main()

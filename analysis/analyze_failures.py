"""Section II.A: are failures in globally difficult scenarios, or localized inside mostly-solved ones?

Reads the per-map JSONL files written by eval_per_scenario.py (drive.py eval_map_log_path).
Each line = one read of one map's log: rates are averages over that map's car-episodes,
n = number of controlled cars in the map.

Usage: python analysis/analyze_failures.py --results analysis/results/<name>
"""
import argparse
import glob
import json
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True, help="Folder with *.jsonl files")
ap.add_argument("--success", default="score", choices=["completion", "score"],
                help="completion = goal reached (collisions ignored); score = goal reached without collision/off-road")
ap.add_argument("--min-ep-len", type=float, default=2.0, help="Drop entries with mean episode_length <= this (instant-goal artifacts)")
ap.add_argument("--localized", type=float, default=0.8, help="Success >= this (but < 1) => localized failure")
ap.add_argument("--globally", type=float, default=0.5, help="Success < this => globally difficult")
ap.add_argument("--min-n", type=float, default=1, help="Ignore maps with fewer cars (summed over reads) than this")
a = ap.parse_args()

out_dir = os.path.join(a.results, "analysis")
os.makedirs(out_dir, exist_ok=True)

rows = []
for f in sorted(glob.glob(os.path.join(a.results, "*.jsonl"))):
    with open(f) as fh:
        rows += [json.loads(line) for line in fh if line.strip()]
if not rows:
    raise SystemExit(f"No *.jsonl results found in {a.results}")
df = pd.DataFrame(rows)
print(f"Loaded {len(df)} log entries from {df['map_id'].nunique()} maps")
print(f"Fields: {sorted(df.columns)}")

success_col = "completion_rate" if a.success == "completion" else "score"
if success_col not in df.columns:
    raise SystemExit(f"Field {success_col!r} not in logs; try --success {'score' if a.success == 'completion' else 'completion'}")

# 1) Drop instant-goal artifacts
if "episode_length" in df.columns:
    deg = df["episode_length"] <= a.min_ep_len
    print(f"Dropped {deg.sum()} entries with mean episode_length <= {a.min_ep_len} "
          f"({df.loc[deg, 'n'].sum():.0f} of {df['n'].sum():.0f} car-weight)")
    d = df[~deg].copy()
else:
    d = df.copy()

# 2) Rates -> counts (rate x cars), then combine all reads of the same map
outcome_cols = {"success": success_col}
for name, col in [("collided", "collision_rate"), ("offroad", "offroad_rate"), ("dnf", "dnf_rate")]:
    if col in d.columns:
        outcome_cols[name] = col
for name, col in outcome_cols.items():
    d[name] = d[col] * d["n"]
g = d.groupby("map_id")[["n"] + list(outcome_cols)].sum()
g["reads"] = d.groupby("map_id").size()
g = g[g["n"] >= a.min_n].copy()
g["success_rate"] = (g["success"] / g["n"]).clip(0, 1)
g["failures"] = g["n"] - g["success"]

# 3) Classify
def classify(sr):
    if sr >= 0.999:
        return "solved"
    if sr >= a.localized:
        return "localized_failure"
    if sr < a.globally:
        return "globally_difficult"
    return "intermediate"

g["category"] = g["success_rate"].apply(classify)
order = ["solved", "localized_failure", "intermediate", "globally_difficult"]

# 4) Summary
total_fail = g["failures"].sum()
summary = g.groupby("category").agg(
    scenarios=("n", "size"), cars=("n", "sum"), failed_cars=("failures", "sum")
).reindex(order).fillna(0)
summary["share_of_scenarios_%"] = 100 * summary["scenarios"] / summary["scenarios"].sum()
summary["share_of_all_failures_%"] = 100 * summary["failed_cars"] / total_fail if total_fail > 0 else 0.0

print(f"\n=== Overall (success = {success_col}) ===")
print(f"Scenarios: {len(g)}  |  cars (summed over reads): {g['n'].sum():.0f}  |  failed: {total_fail:.0f}")
print(f"Pooled success rate:                 {g['success'].sum() / g['n'].sum():.3f}")
print(f"Mean of per-scenario success rates:  {g['success_rate'].mean():.3f}")
for name in ("collided", "offroad", "dnf"):
    if name in g.columns:
        print(f"{name:10s} share of cars:            {g[name].sum() / g['n'].sum():.3f}")
print(f"\n=== By category (localized >= {a.localized:.0%}, globally < {a.globally:.0%}) ===")
print(summary.round(1).to_string())

g.sort_values("success_rate").to_csv(os.path.join(out_dir, "per_scenario.csv"))
summary.to_csv(os.path.join(out_dir, "summary.csv"))

# 5) Plots
colors = {"solved": "tab:green", "localized_failure": "tab:orange",
          "intermediate": "tab:gray", "globally_difficult": "tab:red"}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for c in order:
    s = g[g["category"] == c]
    ax[0].scatter(s["n"], s["success_rate"], s=10, alpha=0.6, c=colors[c], label=f"{c} ({len(s)})")
ax[0].set_xscale("log")
ax[0].set_xlabel("cars in scenario (summed over reads, log)")
ax[0].set_ylabel("scenario success rate")
ax[0].set_title("Each dot = one scenario")
ax[0].legend(fontsize=8)
ax[1].hist(g["success_rate"], bins=40, color="tab:blue")
ax[1].set_xlabel("scenario success rate")
ax[1].set_ylabel("number of scenarios")
ax[1].set_title("Distribution of scenario success")
ax[2].bar(range(len(order)), summary["share_of_all_failures_%"].values, color=[colors[c] for c in order])
ax[2].set_xticks(range(len(order)))
ax[2].set_xticklabels(order, rotation=20, fontsize=8)
ax[2].set_ylabel("% of all failed cars")
ax[2].set_title("Where do failures come from?")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "failure_distribution.png"), dpi=130)
print(f"\nSaved: {out_dir}/per_scenario.csv, summary.csv, failure_distribution.png")

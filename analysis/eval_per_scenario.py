
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from per_scenario import DEFAULT_TRAIN_MAP_DIR, MAX_LOCALIZED_CARS, PerScenarioEvaluator, plot_results
from pufferlib import pufferl

ENV_NAME = "puffer_drive"
EXP_ROOT = "/data/prdrive/experiments"
RESULT_ROOT = "/data/prdrive/results"


def parse_args():
    parser = argparse.ArgumentParser(description="Per-scenario metrics (Evaluator-style)")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Run a single .pt instead of all seeds")
    parser.add_argument("--map-dir", type=str, default=DEFAULT_TRAIN_MAP_DIR)
    parser.add_argument("--num-maps", type=int, default=10000,)
    parser.add_argument("--max-agents", type=int, default=2048,)
    parser.add_argument("--mode", type=str, default="both", choices=["self_play", "human_replay", "both"])
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--success", type=str, default="score", choices=["score", "completion"])
    parser.add_argument(
        "--max-localized-cars",
        type=int,
        default=MAX_LOCALIZED_CARS,
        help="Complete car-fails at or below this count are localized_failure; above is globally_difficult",
    )
    parser.add_argument("--plot", action="store_true", help="Write PNG figures next to the CSVs")
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Skip rollout; plot an existing result dir (e.g. /data/prdrive/results/plr_10k)",
    )
    parser.add_argument(
        "--gif-failures",
        action="store_true",
        default=True,
        help="Re-roll each failed scene and save a gif (default on)",
    )
    parser.add_argument("--no-gif-failures", action="store_false", dest="gif_failures")
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Skip metric rollout; re-render failure gifs from existing map_log.jsonl",
    )
    parser.add_argument(
        "--report-only",
        type=str,
        default=None,
        help="Skip rollout; write the three-lens report from existing CSVs (e.g. /data/prdrive/results/plr_10k)",
    )
    return parser.parse_args()


def _load_config():
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        return pufferl.load_config(ENV_NAME)
    finally:
        sys.argv = argv


def find_checkpoints(exp_name, checkpoint=None):
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        return [os.path.abspath(checkpoint)]
    if not exp_name:
        raise SystemExit("pass --exp-name or --checkpoint")

    candidates = [
        os.path.join(EXP_ROOT, exp_name),
        os.path.join("experiments", exp_name),
        exp_name,
    ]
    exp_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if exp_dir is None:
        raise SystemExit(f"no experiment dir for {exp_name!r}; tried {candidates}")

    finals = sorted(path for path in glob.glob(os.path.join(exp_dir, "*.pt")) if os.path.isfile(path))
    if finals:
        return finals

    ckpts = []
    for run_dir in sorted(glob.glob(os.path.join(exp_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        models = glob.glob(os.path.join(run_dir, "model_*.pt"))
        if models:
            ckpts.append(max(models, key=lambda path: (os.path.getmtime(path), path)))
    if not ckpts:
        raise SystemExit(f"no .pt checkpoints in {exp_dir}")
    return ckpts


def _ckpt_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _tag_logs(logs, checkpoint, exp_name):
    stem = _ckpt_stem(checkpoint)
    if not logs:
        return
    for log in logs:
        if not log:
            continue
        log["checkpoint"] = stem
        log["checkpoint_path"] = checkpoint
        if exp_name:
            log["exp_name"] = exp_name


def _print_summary(label, evaluator):
    stats = evaluator.collect_stats()
    summary = evaluator.category_summary()
    print(f"\n=== {label} ===")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"category summary (localized <= {evaluator.max_localized_cars} failed cars):")
    if not summary.empty:
        printable = summary.copy()
        printable["share_of_scenarios"] = 100 * printable["share_of_scenarios"]
        printable["share_of_all_failures"] = 100 * printable["share_of_all_failures"]
        print(printable.round(1).to_string(index=False))
    return stats


def _write_plots(evaluator, result_dir):
    paths = evaluator.plot(result_dir)
    for path in paths:
        print(f"  plot -> {path}")
    return paths


def main():
    args = parse_args()
    if args.report_only:
        from three_lens import run as run_three_lens

        run_three_lens(args.report_only)
        return
    if args.plot_only:
        paths = plot_results(args.plot_only)
        if not paths:
            raise SystemExit(f"no plots written from {args.plot_only}")
        print(f"wrote {len(paths)} figures:")
        for path in paths:
            print(f"  {path}")
        return

    checkpoints = find_checkpoints(args.exp_name, args.checkpoint)
    if args.exp_name and args.checkpoint is None and len(checkpoints) != 3:
        print(f"warning: expected 3 seed checkpoints, found {len(checkpoints)}")

    result_dir = args.result_dir or os.path.join(
        RESULT_ROOT, args.exp_name or _ckpt_stem(checkpoints[0])
    )
    os.makedirs(result_dir, exist_ok=True)

    config = _load_config()
    config["train"]["device"] = args.device
    config["load_id"] = None
    config["env"]["map_dir"] = args.map_dir

    evaluator = PerScenarioEvaluator(
        config,
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        success=args.success,
        max_localized_cars=args.max_localized_cars,
    )

    run_hr = args.mode in ("human_replay", "both")
    run_sp = args.mode in ("self_play", "both")
    gif_modes = [m for m, on in (("human_replay", run_hr), ("self_play", run_sp)) if on]
    load_env_fn = lambda cfg: pufferl.load_env(ENV_NAME, cfg)

    if args.gif_only:
        _render_gifs_from_logs(args, evaluator, config, checkpoints, result_dir, load_env_fn, gif_modes)
        return
    print(
        f"Experiment {args.exp_name or _ckpt_stem(checkpoints[0])}\n"
        f"  checkpoints ({len(checkpoints)}):\n"
        + "".join(f"    {path}\n" for path in checkpoints)
        + f"  map_dir={evaluator.map_dir}\n"
        f"  result_dir={result_dir}\n"
        f"  num_maps={evaluator.num_maps}  max_agents/batch={args.max_agents}\n"
        f"  sim_steps={evaluator.sim_steps}  hr={run_hr}  sp={run_sp}  gif_failures={args.gif_failures}"
    )

    all_hr = []
    all_sp = []
    for ckpt in checkpoints:
        stem = _ckpt_stem(ckpt)
        print(f"\n--- {stem} ---")
        config["load_model_path"] = ckpt
        evaluator.human_replay_stats = None
        evaluator.self_play_stats = None

        policy = None
        if run_hr:
            policy = _rollout_mode(evaluator, config, load_env_fn, "human_replay", args.max_agents, policy)
            _tag_logs(evaluator.human_replay_stats, ckpt, args.exp_name)
            all_hr.extend(evaluator.human_replay_stats or [])
        if run_sp:
            policy = _rollout_mode(evaluator, config, load_env_fn, "self_play", args.max_agents, policy)
            _tag_logs(evaluator.self_play_stats, ckpt, args.exp_name)
            all_sp.extend(evaluator.self_play_stats or [])

        ckpt_dir = os.path.join(result_dir, stem)
        paths = evaluator.save(ckpt_dir)
        _print_summary(stem, evaluator)
        print(f"  wrote {paths['n_scenarios']} scenarios -> {paths['csv']}")
        if args.plot:
            _write_plots(evaluator, ckpt_dir)
        if args.gif_failures and policy is not None:
            try:
                gifs = evaluator.save_failure_gifs(policy, load_env_fn, ckpt_dir, modes=gif_modes)
                print(f"  wrote {len(gifs)} failure gifs -> {os.path.join(ckpt_dir, 'failures')}")
            except RuntimeError as exc:
                print(f"  skipping failure gifs: {exc}")

    evaluator.human_replay_stats = all_hr or None
    evaluator.self_play_stats = all_sp or None
    combined = evaluator.save(result_dir)
    _print_summary("combined (all seeds)", evaluator)
    print(f"\nwrote combined {combined['n_rows']} rows -> {combined['jsonl']}")
    print(f"wrote combined {combined['n_scenarios']} scenarios -> {combined['csv']}")
    print(f"wrote category summary -> {combined['summary']}")
    if args.plot:
        _write_plots(evaluator, result_dir)
    if args.gif_failures:
        print(f"failure gifs are per-seed under {result_dir}/<ckpt>/failures/")


def _load_policy(evaluator, config, load_env_fn, ckpt, gif_modes):
    config["load_model_path"] = ckpt
    mode = gif_modes[0] if gif_modes else "self_play"
    cfg = evaluator.config_for_batch(mode, [0], render=False)
    env = load_env_fn(cfg)
    try:
        policy = pufferl.load_policy(config, env, ENV_NAME)
        policy.eval()
        return policy
    finally:
        env.close()


def _render_gifs_from_logs(args, evaluator, config, checkpoints, result_dir, load_env_fn, gif_modes):
    print(
        f"gif-only from {result_dir}\n"
        f"  modes={gif_modes}\n"
        f"  checkpoints ({len(checkpoints)}):"
    )
    for ckpt in checkpoints:
        stem = _ckpt_stem(ckpt)
        jsonl = os.path.join(result_dir, stem, "map_log.jsonl")
        if not os.path.isfile(jsonl):
            print(f"\n--- {stem} --- skip, no {jsonl}")
            continue
        print(f"\n--- {stem} ---")
        evaluator.load_logs(jsonl)
        if "self_play" not in gif_modes:
            evaluator.self_play_stats = None
        if "human_replay" not in gif_modes:
            evaluator.human_replay_stats = None
        policy = _load_policy(evaluator, config, load_env_fn, ckpt, gif_modes)
        ckpt_dir = os.path.join(result_dir, stem)
        gifs = evaluator.save_failure_gifs(policy, load_env_fn, ckpt_dir, modes=gif_modes)
        print(f"  wrote {len(gifs)} failure gifs -> {os.path.join(ckpt_dir, 'failures')}")


def _rollout_mode(evaluator, config, load_env_fn, mode, max_agents, policy):
    batches = evaluator.map_batches(mode, max_agents=max_agents)
    print(f"  {mode}: {len(batches)} batches over maps 0..{evaluator.num_maps - 1}")
    for batch_idx, map_ids in enumerate(batches, start=1):
        cfg = evaluator.config_for_batch(mode, map_ids)
        env = load_env_fn(cfg)
        if policy is None:
            policy = pufferl.load_policy(config, env, ENV_NAME)
            policy.eval()
        if mode == "human_replay":
            evaluator.hr_env = env
        else:
            evaluator.sp_env = env
        try:
            tagged = evaluator.rollout(policy, mode=mode)
            print(
                f"    batch {batch_idx}/{len(batches)}  "
                f"maps {int(map_ids[0])}-{int(map_ids[-1])}  "
                f"agents={cfg['env']['num_agents']}  scenes={len(tagged)}"
            )
        finally:
            env.close()
            evaluator.hr_env = None
            evaluator.sp_env = None
    return policy


if __name__ == "__main__":
    main()

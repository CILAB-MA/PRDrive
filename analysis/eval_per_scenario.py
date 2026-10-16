"""Run a PRDrive policy WITHOUT learning and save per-map (per-scenario) results.

Directory convention:
  ckpt_dir   = <ckpt-root>/<exp-name>                            e.g. /data/prdrive/experiments/sp_10k
  result_dir = <result-root>/<exp-name>/<eval-exp-name>/<ckpt>    e.g. /data/prdrive/results/sp_10k/failure_analysis/puffer_drive_arhmtmyc

Uses the eval-only option `eval_map_log_path` in pufferlib/ocean/drive/drive.py:
every worker appends each map's log (+ map_id) to <result_dir>/map_log.<pid>.jsonl.

Usage (inside the container, venv active, from /workspace):
  python analysis/eval_per_scenario.py --exp-name sp_10k --ckpt puffer_drive_arhmtmyc.pt --epochs 350
  python analysis/eval_per_scenario.py --exp-name plr_10k --epochs 40 --device cpu    # first *.pt in ckpt_dir
"""
import argparse
import glob
import os
import sys
import time

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--exp-name", required=True, help="Experiment folder in ckpt-root, e.g. sp_10k or plr_10k")
cli_parser.add_argument("--ckpt", default=None, help="Checkpoint file in ckpt_dir (default: first final *.pt)")
cli_parser.add_argument("--eval-exp-name", default="failure_analysis", help="Name of this evaluation")
cli_parser.add_argument("--ckpt-root", default="/data/prdrive/experiments")
cli_parser.add_argument("--result-root", default="/data/prdrive/results")
cli_parser.add_argument("--map-dir", default="/data/prdrive/binaries/training_80000")
cli_parser.add_argument("--num-maps", type=int, default=None, help="Maps to sample from (default: config value)")
cli_parser.add_argument("--epochs", type=int, default=60, help="Rollout batches (each = rollout_horizon steps per car)")
cli_parser.add_argument("--device", default=None, help="cuda or cpu (default: config value)")
cli, rest = cli_parser.parse_known_args()
sys.argv = [sys.argv[0]] + rest  # leave other args for PufferLib's own parser

ckpt_dir = os.path.join(cli.ckpt_root, cli.exp_name)
if cli.ckpt:
    ckpt_path = os.path.join(ckpt_dir, cli.ckpt)
else:
    finals = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not finals:
        raise SystemExit(f"No *.pt checkpoints found in {ckpt_dir}")
    ckpt_path = finals[0]
if not os.path.isfile(ckpt_path):
    raise SystemExit(f"Checkpoint not found: {ckpt_path}")
ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
result_dir = os.path.join(cli.result_root, cli.exp_name, cli.eval_exp_name, ckpt_name)
out = os.path.join(result_dir, "map_log.jsonl")

from pufferlib import pufferl  # noqa: E402

ENV = "puffer_drive"
args = pufferl.load_config(ENV)
args["load_model_path"] = ckpt_path
args["env"]["map_dir"] = cli.map_dir
if cli.num_maps:
    args["env"]["num_maps"] = cli.num_maps
args["env"]["map_sampling_strategy"] = "uniform"  # evaluate every model on evenly sampled maps
args["env"]["eval_map_log_path"] = out
args["wandb"] = False
args["neptune"] = False
if cli.device:
    args["train"]["device"] = cli.device

os.makedirs(result_dir, exist_ok=True)
print(f"Checkpoint: {ckpt_path}")
print(f"Results:    {result_dir}/map_log.<pid>.jsonl")
print(f"Maps:       {args['env']['map_dir']}  (num_maps={args['env'].get('num_maps')})")
print(f"Dynamics:   {args['env'].get('dynamics_model')}  reg_mode={args['env'].get('reg_mode')}  device={args['train']['device']}")

vecenv = pufferl.load_env(ENV, args)
policy = pufferl.load_policy(args, vecenv, ENV)
policy.eval()

train_config = dict(**args["train"], env=ENV, eval=args.get("eval", {}))
trainer = pufferl.PuffeRL(train_config, vecenv, policy, full_args=args)

start = time.time()
for epoch in range(cli.epochs):
    trainer.evaluate()  # rollouts only; trainer.train() is never called, so nothing is learned
    print(f"epoch {epoch + 1}/{cli.epochs} done, {time.time() - start:.0f}s elapsed", flush=True)

vecenv.close()
print(f"Finished. Analyze with:\n  python analysis/analyze_failures.py --results {result_dir}")

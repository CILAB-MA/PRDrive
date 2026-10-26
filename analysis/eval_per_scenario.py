"""Run a PRDrive policy WITHOUT training and save per-map (per-scenario) results.

The driving loop follows the standard (non-WOSAC) branch of pufferl.eval():
a single vec env (num_envs=1), the LSTM state created once, then each step
policy.forward_eval -> sample_logits -> vecenv.step. No PuffeRL, no rollout
buffer, and no dependence on rollout_horizon. Rendering is skipped and the
loop stops after --steps. As in eval(), termination_mode = 0 (episodes end
at episode_length).

Per-map results come from the eval-only option `eval_map_log_path` in
pufferlib/ocean/drive/drive.py: each map's log + map_id is appended to
<result_dir>/map_log.<pid>.jsonl.

Directory convention:
  ckpt_dir   = <ckpt-root>/<exp-name>                            e.g. /data/prdrive/experiments/sp_10k
  result_dir = <result-root>/<exp-name>/<eval-exp-name>/<ckpt>    e.g. /data/prdrive/results/sp_10k/failure_analysis/puffer_drive_arhmtmyc

Usage (inside the container, venv active, from /workspace):
  python analysis/eval_per_scenario.py --exp-name sp_10k --ckpt puffer_drive_arhmtmyc.pt --steps 10000
  python analysis/eval_per_scenario.py --exp-name plr_10k --steps 300 --device cpu    # first *.pt in ckpt_dir
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
cli_parser.add_argument("--num-agents", type=int, default=None, help="Agents in the eval env (default: config value)")
cli_parser.add_argument("--steps", type=int, default=10000, help="Simulation steps to drive (0.1 s each)")
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

import numpy as np # noqa: E402
import torch # noqa: E402
import pufferlib # noqa: E402
import pufferlib.pytorch # noqa: E402
from pufferlib import pufferl  # noqa: E402

ENV = "puffer_drive"
args = pufferl.load_config(ENV)
args["load_model_path"] = ckpt_path
args["wandb"] = False
args["neptune"] = False
if cli.device:
    args["train"]["device"] = cli.device
args["env"]["termination_mode"] = 0
args["env"]["map_dir"] = cli.map_dir
if cli.num_maps:
    args["env"]["num_maps"] = cli.num_maps
if cli.num_agents:
    args["env"]["num_agents"] = cli.num_agents
args["env"]["map_sampling_strategy"] = "uniform"  # evaluate every model on evenly sampled maps
args["env"]["eval_map_log_path"] = out

backend = args["vec"]["backend"]
if backend != "PufferEnv":
    backend = "Serial"
args["vec"] = dict(backend=backend, num_envs=1)

os.makedirs(result_dir, exist_ok=True)
device = args["train"]["device"]
print(f"Checkpoint: {ckpt_path}")
print(f"Results:    {result_dir}/map_log.<pid>.jsonl")
print(f"Maps:       {args['env']['map_dir']}  (num_maps={args['env'].get('num_maps')})")
print(f"Agents:     {args['env'].get('num_agents')}  episode_length = {args['env'].get('episode_length')}  "
      f"resample_frequency = {args['env'].get('resample_frequency')}")
print(f"Dynamics:   {args['env'].get('dynamics_model')}  reg_mode = {args['env'].get('reg_mode')}  device = {args['train']['device']}")
print(f"Steps: {cli.steps}")

vecenv = pufferl.load_env(ENV, args)
policy = pufferl.load_policy(args, vecenv, ENV)
policy.eval()

ob, info = vecenv.reset()
num_agents = vecenv.observation_space.shape[0]

state = {}
if args["train"]["use_rnn"]:
    state = dict(
        lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
        lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
    )

start = time.time()
for step in range(1, cli.steps + 1):
    with torch.no_grad():
        ob_t = torch.as_tensor(ob).to(device)
        logits, value = policy.forward_eval(ob_t, state)
        action, logprob, _ =pufferlib.pytorch.sample_logits(logits)
        action = action.cpu().numpy().reshape(vecenv.action_space.shape)
    if isinstance(logits, torch.distributions.Normal):
        action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

    ob, reward, terminal, truncated, info = vecenv.step(action)

    if step % 100 == 0 or step == cli.steps:
        print(f"step {step}/{cli.steps}, {time.time() - start:.0f}s elapsed", flush = True)

vecenv.close()
print(f"Finished. Analyze with:\n python analysis/analyze_failures.py --results {result_dir}")

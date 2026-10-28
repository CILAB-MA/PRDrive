
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import pufferlib
from pufferlib.ocean.benchmark.structure import (
    MAX_LOCALIZED_CARS,
    STRUCTURE_CATEGORIES,
    classify_structure,
    majority_structure,
)

METRIC_KEYS = (
    "n",
    "score",
    "offroad_rate",
    "collision_rate",
    "at_fault_collision_rate",
    "episode_length",
    "episode_return",
    "completion_rate",
    "offroad_per_agent",
    "collisions_per_agent",
    "perc_controlled",
    "route_progress",
    "lateral_error_avg",
    "rear_collision_rate",
    "longitudinal_error_avg",
    "displacement_error_avg",
    "delta_v_sum",
    "delta_v_max",
    "delta_v_count",
    "delta_v_under_1mph",
)

CATEGORIES = STRUCTURE_CATEGORIES
RATE_KEYS = tuple(k for k in METRIC_KEYS if k != "n")
DEFAULT_TRAIN_MAP_DIR = "/data/prdrive/binaries/training_80000"


def _to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


LENS_METRICS = {
    "success": "score",
    "collision": "safety_rate",
    "return": "episode_return",
}


def _controllable_counts(map_dir, num_maps):
    path = os.path.join(map_dir, "controllable_counts.npy")
    if not os.path.isfile(path):
        return None
    counts = np.asarray(np.load(path))
    return np.asarray(counts[: min(int(num_maps), len(counts))], dtype=np.int32)


def agent_budget_for_maps(map_dir, num_maps, control_mode):
    """Agent count is determined by the maps + control mode, not num_eval_agents.

    self-play / control_agents: sum of controllable cars on map_000..map_{N-1}.
    human-replay / control_sdc_only: one SDC per map.
    """
    num_maps = int(num_maps)
    if control_mode == "control_sdc_only":
        return num_maps
    counts = _controllable_counts(map_dir, num_maps)
    if counts is not None and int(counts.sum()) > 0:
        return int(counts.sum())
    return num_maps


def _agents_for_ids(map_dir, map_ids, control_mode):
    map_ids = np.asarray(map_ids, dtype=np.int32)
    if map_ids.size == 0:
        return 1
    if control_mode == "control_sdc_only":
        return int(map_ids.size)
    counts = _controllable_counts(map_dir, int(map_ids.max()) + 1)
    if counts is None:
        return int(map_ids.size)
    total = int(sum(int(counts[i]) for i in map_ids if i < len(counts)))
    return max(total, 1)


def iter_map_batches(map_dir, num_maps, control_mode, max_agents=2048):
    """Yield contiguous map-id arrays that fit in an agent budget."""
    ids = np.arange(int(num_maps), dtype=np.int32)
    if control_mode == "control_sdc_only":
        step = max(int(max_agents), 1)
        for start in range(0, len(ids), step):
            yield ids[start : start + step]
        return
    counts = _controllable_counts(map_dir, num_maps)
    batch = []
    agents = 0
    for map_id in ids:
        n = int(counts[map_id]) if counts is not None and map_id < len(counts) else 1
        n = max(n, 1)
        if batch and agents + n > max_agents:
            yield np.asarray(batch, dtype=np.int32)
            batch, agents = [], 0
        batch.append(int(map_id))
        agents += n
    if batch:
        yield np.asarray(batch, dtype=np.int32)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_") or "unknown"


def _find_mp4(directory, scenario_id):
    exact = os.path.join(directory, f"{scenario_id}.mp4")
    if os.path.isfile(exact):
        return exact
    matches = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".mp4")]
    if len(matches) == 1:
        return matches[0]
    return None


def _start_headless_display():
    """Drive/GLFW still need an X server. Host DISPLAY=:0 is often unusable, so start Xvfb."""
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        raise RuntimeError("Xvfb not found; install xvfb or pass --no-gif-failures")
    prev = os.environ.get("DISPLAY")
    display_num = 100 + (os.getpid() % 900)
    display = f":{display_num}"
    proc = subprocess.Popen(
        [xvfb, display, "-screen", "0", "1920x1080x24", "+extension", "GLX", "-ac", "-noreset"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    lock = f"/tmp/.X{display_num}-lock"
    for _ in range(40):
        if os.path.exists(lock) or proc.poll() is not None:
            break
        time.sleep(0.1)
    time.sleep(0.2)
    if proc.poll() is not None:
        raise RuntimeError(f"Xvfb failed to start on {display}")
    os.environ["DISPLAY"] = display
    print(f"  headless display {display} (was {prev!r})")
    return {"proc": proc, "prev_display": prev}


def _stop_headless_display(state):
    if not state:
        return
    proc = state["proc"]
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    if state["prev_display"] is None:
        os.environ.pop("DISPLAY", None)
    else:
        os.environ["DISPLAY"] = state["prev_display"]


INVALID_XY = -5000.0
OBS_COLLISION_IDX = 9


def _box_corners(x, y, length, width, heading):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _polyline_segments(roads):
    xs, ys, lengths = roads["x"], roads["y"], roads["lengths"]
    segs = []
    offset = 0
    for n in lengths:
        n = int(n)
        if n >= 2:
            segs.append((xs[offset : offset + n], ys[offset : offset + n]))
        offset += n
    return segs


def _write_scene_gif(path, frames, roads, log):
    """Top-down matplotlib gif: zoom on failed cars, label who collided/offroad."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image

    if not frames:
        return None

    n_agents = frames[0]["x"].shape[0]
    col_any = np.zeros(n_agents, dtype=bool)
    off_any = np.zeros(n_agents, dtype=bool)
    for fr in frames:
        col_any |= fr["collision"]
        off_any |= fr["offroad"]
    failed = col_any | off_any
    focus = np.where(failed)[0] if failed.any() else np.arange(n_agents)

    xs, ys = [], []
    for fr in frames:
        valid = fr["x"] > INVALID_XY
        idx = np.zeros(n_agents, dtype=bool)
        idx[focus] = True
        mask = valid & idx
        if mask.any():
            xs.append(fr["x"][mask])
            ys.append(fr["y"][mask])
    if not xs:
        return None
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    pad = 35.0
    xmin, xmax = float(xs.min() - pad), float(xs.max() + pad)
    ymin, ymax = float(ys.min() - pad), float(ys.max() + pad)
    span = max(xmax - xmin, ymax - ymin, 80.0)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span

    fail_ids = [int(frames[0]["id"][i]) for i in np.where(failed)[0]]
    mode = log.get("mode", "self_play")
    map_id = int(log.get("map_id", -1))
    score = float(log.get("score", 0.0))
    segs = _polyline_segments(roads) if roads else []

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    images = []
    for t, fr in enumerate(frames):
        ax.clear()
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#111111")
        for px, py in segs:
            ax.plot(px, py, color="#888888", lw=0.8, solid_capstyle="round")
        for i in range(n_agents):
            if fr["x"][i] <= INVALID_XY:
                continue
            corners = _box_corners(fr["x"][i], fr["y"][i], fr["length"][i], fr["width"][i], fr["heading"][i])
            if fr["collision"][i]:
                fc, ec, lw = "#e03131", "#ff6b6b", 2.4
            elif fr["offroad"][i]:
                fc, ec, lw = "#f59f00", "#ffd43b", 2.4
            else:
                fc, ec, lw = "#4dabf7", "#74c0fc", 1.4
            ax.add_patch(Polygon(corners, closed=True, facecolor=fc, edgecolor=ec, lw=lw, alpha=0.9, zorder=3))
            hx = fr["x"][i] + 0.6 * fr["length"][i] * np.cos(fr["heading"][i])
            hy = fr["y"][i] + 0.6 * fr["length"][i] * np.sin(fr["heading"][i])
            ax.plot([fr["x"][i], hx], [fr["y"][i], hy], color=ec, lw=1.6, zorder=4)
            tags = [str(int(fr["id"][i]))]
            if fr["collision"][i]:
                tags.append("COL")
            if fr["offroad"][i]:
                tags.append("OFF")
            ax.text(
                fr["x"][i],
                fr["y"][i] + max(fr["width"][i], 2.0) + 1.5,
                " ".join(tags),
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="#000000", alpha=0.7, lw=0),
            )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fail_txt = ",".join(str(i) for i in fail_ids) if fail_ids else "none"
        ax.set_title(
            f"{mode}  map_{map_id:05d}  t={t:02d}  score={score:.2f}  fail_ids=[{fail_txt}]",
            color="white",
            fontsize=12,
            pad=8,
        )
        ax.text(
            0.02,
            0.02,
            "blue=policy   red=collision   yellow=offroad   label=agent id",
            transform=ax.transAxes,
            color="#dddddd",
            fontsize=9,
            va="bottom",
        )
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(Image.fromarray(buf.copy()))
    plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=False)
    return path


class PerScenarioEvaluator:

    def __init__(
        self,
        configs,
        logger=None,
        map_dir=None,
        num_maps=None,
        success="score",
        max_localized_cars=MAX_LOCALIZED_CARS,
    ):
        if success not in ("score", "completion"):
            raise ValueError("success must be 'score' or 'completion'")
        self.configs = configs
        self.logger = logger
        self.sim_steps = 90
        self.self_play_stats = None
        self.human_replay_stats = None
        self.sp_env = None
        self.hr_env = None
        self._map_dir_override = map_dir
        self._num_maps_override = num_maps
        self.success = success
        self.success_col = "completion_rate" if success == "completion" else "score"
        self.max_localized_cars = int(max_localized_cars)
        self._unpack_eval_configs(configs)

    def _unpack_eval_configs(self, configs):
        eval_config = copy.deepcopy(configs)
        eval_config["env"]["termination_mode"] = 1
        backend = eval_config["eval"].get("backend", "PufferEnv")
        eval_config["env"]["map_dir"] = (
            self._map_dir_override or configs["env"].get("map_dir") or DEFAULT_TRAIN_MAP_DIR
        )
        num_maps = int(self._num_maps_override) if self._num_maps_override is not None else 10000
        eval_config["env"]["num_maps"] = num_maps
        eval_config["env"]["episode_length"] = 130
        eval_config["vec"] = dict(backend=backend, num_envs=1)
        eval_config["env"]["fix_lambdas"] = True
        eval_config["env"]["fix_rewards"] = True
        eval_config["env"]["async_resets"] = False
        eval_config["env"]["lambda_value"] = configs["env"]["lambda_value"]
        eval_config["env"]["obs_partner_noise_speed"] = 0.0
        eval_config["env"]["obs_partner_noise_pos"] = 0.0
        eval_config["env"]["render_mode"] = 0

        self.num_maps = num_maps
        self.map_dir = eval_config["env"]["map_dir"]
        self.hr_eval_config = copy.deepcopy(eval_config)
        self.hr_eval_config["env"]["control_mode"] = "control_sdc_only"
        self.hr_eval_config["env"]["goal_behavior"] = 0
        self.sp_eval_config = copy.deepcopy(eval_config)
        self.sp_eval_config["env"]["control_mode"] = "control_agents"

    def config_for_batch(self, mode, map_ids, render=False):
        """Env config for a slice of map ids. Agent count follows those maps + mode."""
        map_ids = np.ascontiguousarray(map_ids, dtype=np.int32)
        config = copy.deepcopy(self.hr_eval_config if mode == "human_replay" else self.sp_eval_config)
        control_mode = config["env"]["control_mode"]
        config["env"]["map_id_queue"] = map_ids
        config["env"]["num_agents"] = _agents_for_ids(self.map_dir, map_ids, control_mode)
        if render:
            config["env"]["render_mode"] = 1
        return config

    def map_batches(self, mode, max_agents=2048):
        control_mode = "control_sdc_only" if mode == "human_replay" else "control_agents"
        return list(iter_map_batches(self.map_dir, self.num_maps, control_mode, max_agents=max_agents))

    def rollout(self, policy, mode="self_play"):
        env = self.hr_env if mode == "human_replay" else self.sp_env
        env_statistics = self._run_rollout(policy, env, mode, per_env_logs=True)
        env_statistics = self._attach_scenario_ids(env, env_statistics, mode)

        if mode == "self_play":
            self.self_play_stats = (self.self_play_stats or []) + list(env_statistics or [])
        elif mode == "human_replay":
            self.human_replay_stats = (self.human_replay_stats or []) + list(env_statistics or [])
        return env_statistics

    def _run_rollout(self, policy, env, mode, render_env_idx=None, per_env_logs=False, view_mode=None):
        from pufferlib.ocean.drive.drive import RenderView

        if view_mode is None:
            view_mode = RenderView.FULL_SIM_STATE

        driver = env.driver_env
        num_agents = env.observation_space.shape[0]
        device = self.configs["train"]["device"]

        obs, _info = env.reset()
        terminals = np.zeros(num_agents, dtype=bool)
        state = {}
        if self.configs["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        info_list = []
        for _time_idx in range(self.sim_steps):
            if render_env_idx is not None:
                term = bool(np.asarray(terminals).reshape(-1)[render_env_idx]) if mode == "human_replay" else False
                if not term:
                    driver.render(view_mode=view_mode, env_idx=render_env_idx)

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, _value = policy.forward_eval(ob_tensor, state)
                action, _logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(env.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            obs, _rewards, terminals, truncated, info_list = env.step(action_np, per_env_logs=per_env_logs)

            if np.asarray(truncated).all():
                break

        return info_list

    def _attach_scenario_ids(self, env, logs, mode):
        if not logs:
            return logs
        driver = env.driver_env
        scenario_ids = list(getattr(driver, "scenario_ids", []) or [])
        for i, log in enumerate(logs):
            if not log:
                continue
            log["map_id"] = int(driver.map_ids[i])
            log["scenario_id"] = scenario_ids[i] if i < len(scenario_ids) else ""
            log["env_idx"] = i
            log["mode"] = mode
            for key, value in list(log.items()):
                log[key] = _to_python(value)
        return logs

    def failure_logs(self):
        """One env log per failed scene (score below solved threshold)."""
        return [
            log
            for log in self._populated_logs()
            if float(log.get(self.success_col, log.get("score", 1.0))) < 0.999
        ]

    def load_logs(self, jsonl_path):
        hr, sp = [], []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                log = json.loads(line)
                if log.get("mode") == "human_replay":
                    hr.append(log)
                else:
                    sp.append(log)
        self.human_replay_stats = hr or None
        self.self_play_stats = sp or None

    def save_failure_gifs(self, policy, load_env_fn, result_dir, modes=None):
        """Re-roll each failed scene and write a labeled top-down matplotlib gif."""
        fails = self.failure_logs()
        if modes:
            fails = [log for log in fails if log.get("mode") in modes]
        if not fails:
            return []

        gif_root = os.path.join(result_dir, "failures")
        os.makedirs(gif_root, exist_ok=True)
        written = []
        for i, log in enumerate(fails, start=1):
            mode = log.get("mode", "self_play")
            map_id = int(log["map_id"])
            print(f"  gif {i}/{len(fails)}  {mode} map_{map_id:05d} score={log.get('score', 0):.3f}")
            path = self._render_failure_gif(policy, load_env_fn, gif_root, log)
            if path:
                written.append(path)
        return written

    def _collect_viz_frames(self, policy, env):
        driver = env.driver_env
        num_agents = env.observation_space.shape[0]
        device = self.configs["train"]["device"]
        obs, _info = env.reset()
        roads = driver.get_road_edge_polylines()
        terminals = np.zeros(num_agents, dtype=bool)
        state = {}
        if self.configs["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )
        col_sticky = np.zeros(num_agents, dtype=bool)
        off_sticky = np.zeros(num_agents, dtype=bool)
        frames = []
        info_list = []
        for _ in range(self.sim_steps):
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, _value = policy.forward_eval(ob_tensor, state)
                action, _logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(env.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            obs, rewards, terminals, truncated, info_list = env.step(action_np, per_env_logs=True)
            agents = driver.get_global_agent_state()
            n = min(num_agents, int(agents["x"].shape[0]))
            collided = np.asarray(obs[:n, OBS_COLLISION_IDX] > 0.5)
            reward = np.asarray(rewards).reshape(-1)[:n]
            offroad = (reward < -0.05) & ~collided
            col_sticky[:n] |= collided
            off_sticky[:n] |= offroad
            frames.append(
                {
                    "x": np.asarray(agents["x"][:n]).copy(),
                    "y": np.asarray(agents["y"][:n]).copy(),
                    "heading": np.asarray(agents["heading"][:n]).copy(),
                    "length": np.asarray(agents["length"][:n]).copy(),
                    "width": np.asarray(agents["width"][:n]).copy(),
                    "id": np.asarray(agents["id"][:n]).copy(),
                    "collision": col_sticky[:n].copy(),
                    "offroad": off_sticky[:n].copy(),
                }
            )
            if np.asarray(truncated).all() or np.asarray(terminals).all():
                break
        return frames, roads, info_list

    def _render_failure_gif(self, policy, load_env_fn, gif_root, log):
        mode = log.get("mode", "self_play")
        map_id = int(log["map_id"])
        scenario = _safe_name(str(log.get("scenario_id") or f"map_{map_id:05d}"))
        ckpt = _safe_name(str(log.get("checkpoint") or "ckpt"))
        out_dir = os.path.join(gif_root, mode)
        os.makedirs(out_dir, exist_ok=True)
        tags = []
        if float(log.get("collision_rate", 0) or 0) > 0:
            tags.append("col")
        if float(log.get("offroad_rate", 0) or 0) > 0:
            tags.append("off")
        tag = ("__" + "_".join(tags)) if tags else ""
        gif_path = os.path.join(
            out_dir,
            f"{ckpt}__map_{map_id:05d}__{scenario}{tag}__score_{float(log.get('score', 0)):.2f}.gif",
        )

        config = self.config_for_batch(mode, [map_id], render=False)
        env = load_env_fn(config)
        try:
            frames, roads, _info = self._collect_viz_frames(policy, env)
        finally:
            env.close()
        if not frames:
            print(f"    warning: no frames for map {map_id} ({scenario})")
            return None
        return _write_scene_gif(gif_path, frames, roads, log)

    def collect_stats(self):
        stats = {}

        if self.human_replay_stats is not None:
            populated = [log for log in self.human_replay_stats if log and log.get("n", 0) > 0]
            if populated:
                collisions_per_agent = np.array([log["collisions_per_agent"] for log in populated])
                did_collide = np.array([log["collision_rate"] for log in populated])
                stats["eval/hr_mean_collisions_per_agent"] = float(np.mean(collisions_per_agent))
                stats["eval/hr_mean_did_collide"] = float(np.mean(did_collide))
                stats["eval/hr_score"] = float(np.mean([log["score"] for log in populated]))

        if self.self_play_stats is not None:
            populated = [log for log in self.self_play_stats if log and log.get("n", 0) > 0]
            if populated:
                collisions_per_agent = np.array([log["collisions_per_agent"] for log in populated])
                did_collide = np.array([log["collision_rate"] for log in populated])
                stats["eval/sp_mean_collisions_per_agent"] = float(np.mean(collisions_per_agent))
                stats["eval/sp_mean_did_collide"] = float(np.mean(did_collide))
                stats["eval/sp_score"] = float(np.mean([log["score"] for log in populated]))
                stats["eval/sp_num_agents"] = float(np.mean([log["n"] for log in populated]))

        table = self.per_scenario_table()
        if not table.empty:
            stats["eval/num_scenarios"] = float(len(table))
            summary = self.category_summary()
            for _, row in summary.iterrows():
                cat = row["category"]
                stats[f"eval/{cat}_scenarios"] = float(row["scenarios"])
                stats[f"eval/{cat}_fail_share"] = float(row["share_of_all_failures"])
        return stats

    def _populated_logs(self):
        logs = []
        for block in (self.human_replay_stats, self.self_play_stats):
            if not block:
                continue
            logs.extend(log for log in block if log and log.get("n", 0) > 0)
        return logs

    def per_scenario_table(self):
        logs = self._populated_logs()
        if not logs:
            return pd.DataFrame()

        grouped = defaultdict(list)
        for row in logs:
            grouped[(row.get("mode", ""), int(row["map_id"]))].append(row)

        out = []
        for (mode, map_id), rows in grouped.items():
            n_sum = float(sum(float(r.get("n", 0.0) or 0.0) for r in rows))
            n_mean = n_sum / len(rows) if rows else 0.0
            scenario_id = next((str(r.get("scenario_id", "")) for r in rows if r.get("scenario_id")), "")
            agg = {
                "mode": mode,
                "map_id": map_id,
                "scenario_id": scenario_id,
                "n": n_mean,
            }
            for key in RATE_KEYS:
                weighted = [r[key] * r.get("n", 0.0) for r in rows if key in r]
                if weighted and n_sum:
                    agg[key] = float(sum(weighted) / n_sum)
            success_rate = float(agg.get(self.success_col, 0.0))
            agg["success_rate"] = success_rate
            agg["success"] = success_rate * n_mean
            agg["failures"] = n_mean - agg["success"]
            agg["category"] = majority_structure(
                classify_structure(
                    row,
                    max_localized_cars=self.max_localized_cars,
                    score_key=self.success_col,
                )
                for row in rows
            )
            out.append(agg)
        return pd.DataFrame(out).sort_values(["mode", "map_id"]).reset_index(drop=True)

    def category_summary(self):
        table = self.per_scenario_table()
        if table.empty:
            return pd.DataFrame(
                columns=["category", "scenarios", "cars", "failed_cars", "share_of_scenarios", "share_of_all_failures"]
            )
        total_fail = float(table["failures"].sum())
        rows = []
        for cat in CATEGORIES:
            part = table[table["category"] == cat]
            cars = float(part["n"].sum()) if not part.empty else 0.0
            failed = float(part["failures"].sum()) if not part.empty else 0.0
            rows.append(
                {
                    "category": cat,
                    "scenarios": int(len(part)),
                    "cars": cars,
                    "failed_cars": failed,
                    "share_of_scenarios": len(part) / len(table),
                    "share_of_all_failures": (failed / total_fail) if total_fail > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def save(self, result_dir):
        os.makedirs(result_dir, exist_ok=True)
        jsonl_path = os.path.join(result_dir, "map_log.jsonl")
        csv_path = os.path.join(result_dir, "per_scenario.csv")
        summary_path = os.path.join(result_dir, "category_summary.csv")

        with open(jsonl_path, "w") as f:
            for row in self._populated_logs():
                f.write(json.dumps(row) + "\n")

        table = self.per_scenario_table()
        summary = self.category_summary()
        if not table.empty:
            table.to_csv(csv_path, index=False)
        if not summary.empty:
            summary.to_csv(summary_path, index=False)
        return {
            "jsonl": jsonl_path,
            "csv": csv_path,
            "summary": summary_path,
            "n_rows": len(self._populated_logs()),
            "n_scenarios": int(len(table)),
        }

    def plot(self, result_dir):
        """Write PNGs next to the CSVs. Same layout as analyze_failures.py, plus mode/seed views."""
        return plot_results(
            result_dir,
            table=self.per_scenario_table(),
            summary=self.category_summary(),
            logs=self._populated_logs(),
        )


CATEGORY_COLORS = {
    "solved": "tab:green",
    "localized_failure": "tab:orange",
    "globally_difficult": "tab:cyan",
    "many_fail": "tab:red",
}


def _failure_panels(ax, table, summary, title_prefix=""):
    """analyze_failures.py 3-panel: scatter, hist, fail-share bar."""
    for cat in CATEGORIES:
        part = table[table["category"] == cat] if "category" in table.columns else table.iloc[0:0]
        ax[0].scatter(
            part["n"],
            part["success_rate"],
            s=14,
            alpha=0.7,
            c=CATEGORY_COLORS[cat],
            label=f"{cat} ({len(part)})",
        )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("cars in scenario (log)")
    ax[0].set_ylabel("scenario success rate")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title(f"{title_prefix}Each dot = one scenario".strip())
    ax[0].legend(fontsize=7, loc="lower right")

    ax[1].hist(table["success_rate"], bins=20, range=(0, 1), color="tab:blue")
    ax[1].set_xlabel("scenario success rate")
    ax[1].set_ylabel("number of scenarios")
    ax[1].set_title(f"{title_prefix}Success distribution".strip())

    fail_share = [100 * float(summary.loc[summary["category"] == cat, "share_of_all_failures"].sum()) for cat in CATEGORIES]
    ax[2].bar(range(len(CATEGORIES)), fail_share, color=[CATEGORY_COLORS[c] for c in CATEGORIES])
    ax[2].set_xticks(range(len(CATEGORIES)))
    ax[2].set_xticklabels(CATEGORIES, rotation=20, fontsize=8)
    ax[2].set_ylabel("% of all failed cars")
    ax[2].set_title(f"{title_prefix}Where failures come from".strip())
    ax[2].set_ylim(0, max(fail_share + [1]) * 1.15)


def plot_results(result_dir, table=None, summary=None, logs=None):
    """Write visualization PNGs into result_dir. Loads CSV/JSONL if data is not passed in."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(result_dir, exist_ok=True)
    if table is None:
        csv_path = os.path.join(result_dir, "per_scenario.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"no per_scenario.csv in {result_dir}")
        table = pd.read_csv(csv_path)
    if summary is None:
        summary_path = os.path.join(result_dir, "category_summary.csv")
        summary = pd.read_csv(summary_path) if os.path.isfile(summary_path) else pd.DataFrame()
    if logs is None:
        jsonl_path = os.path.join(result_dir, "map_log.jsonl")
        if os.path.isfile(jsonl_path):
            logs = [json.loads(line) for line in open(jsonl_path) if line.strip()]
        else:
            logs = []

    if table.empty:
        return []

    written = []
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    _failure_panels(ax, table, summary)
    fig.suptitle("structure: 1-car localized vs 2+ globally-difficult vs many-fail (return<0)", fontsize=10, y=1.02)
    fig.tight_layout()
    path = os.path.join(result_dir, "failure_distribution.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    modes = [m for m in ("self_play", "human_replay") if "mode" in table.columns and (table["mode"] == m).any()]
    if len(modes) > 1:
        fig, axes = plt.subplots(len(modes), 3, figsize=(17, 4.4 * len(modes)))
        for i, mode in enumerate(modes):
            part = table[table["mode"] == mode]
            part_summary = _summary_from_table(part)
            row = axes[i]
            _failure_panels(row, part, part_summary, title_prefix=f"{mode}: ")
        fig.tight_layout()
        path = os.path.join(result_dir, "failure_distribution_by_mode.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if logs:
        raw = pd.DataFrame(logs)
        if "checkpoint" in raw.columns and "score" in raw.columns and raw["checkpoint"].nunique() > 1:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=150)
            grouped = raw.groupby(["checkpoint", "mode"])["score"].mean().unstack() if "mode" in raw.columns else None
            if grouped is not None and not grouped.empty:
                grouped.plot(kind="bar", ax=ax, rot=15)
                ax.set_ylim(0.9, 1.0)
                ax.set_ylabel("mean score")
                ax.set_xlabel("checkpoint")
                ax.set_title("Score by seed")
                ax.legend(title="mode")
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                path = os.path.join(result_dir, "score_by_seed.png")
                fig.savefig(path, dpi=130, bbox_inches="tight")
                plt.close(fig)
                written.append(path)

    return written


def _summary_from_table(table):
    if table.empty:
        return pd.DataFrame(columns=["category", "share_of_all_failures"])
    total_fail = float(table["failures"].sum()) if "failures" in table.columns else 0.0
    rows = []
    for cat in CATEGORIES:
        part = table[table["category"] == cat]
        failed = float(part["failures"].sum()) if (not part.empty and "failures" in part.columns) else 0.0
        rows.append(
            {
                "category": cat,
                "share_of_all_failures": (failed / total_fail) if total_fail > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)

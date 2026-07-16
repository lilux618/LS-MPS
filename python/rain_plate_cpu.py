#!/usr/bin/env python3
"""Inclined-plate rain benchmark visual baseline.

This is a deterministic CPU particle workload used to validate injection,
neighbor statistics, wall contact, film transport, outflow accounting, and
visualization. The pairwise incompressibility term is a penalty baseline; the
project's WLS-PPE path remains separately validated in the C++ benchmark and
will replace this term in the next integration step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from scipy.spatial import cKDTree


@dataclass
class Params:
    seed: int = 7
    steps: int = 180
    dt: float = 0.0015
    save_every: int = 6
    inject_steps: int = 125
    inject_per_step: int = 12
    l0: float = 0.012
    re_ratio: float = 2.1
    gravity: float = 9.81
    plate_left_z: float = 0.22
    plate_right_z: float = 0.08
    plate_x0: float = 0.06
    plate_x1: float = 0.94
    restitution: float = 0.02
    wall_friction: float = 0.985
    pair_stiffness: float = 1250.0
    pair_damping: float = 5.0
    viscosity: float = 3.0
    max_speed: float = 6.0
    width: float = 1.0
    height: float = 0.72


def plate_z(x: np.ndarray | float, p: Params) -> np.ndarray | float:
    slope = (p.plate_right_z - p.plate_left_z) / (p.plate_x1 - p.plate_x0)
    return p.plate_left_z + slope * (np.asarray(x) - p.plate_x0)


def plate_vectors(p: Params) -> tuple[np.ndarray, np.ndarray]:
    slope = (p.plate_right_z - p.plate_left_z) / (p.plate_x1 - p.plate_x0)
    t = np.array([1.0, slope], dtype=float)
    t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0]], dtype=float)
    return t, n


def inject(pos: np.ndarray, vel: np.ndarray, ids: np.ndarray, next_id: int,
           step: int, p: Params, rng: np.random.Generator):
    if step >= p.inject_steps:
        return pos, vel, ids, next_id, 0
    n = p.inject_per_step
    # Three bands produce a visible rain curtain while keeping deterministic load.
    x = rng.uniform(0.16, 0.78, n)
    z = rng.uniform(0.58, 0.70, n)
    jitter = rng.normal(0.0, 0.0015, (n, 2))
    new_pos = np.column_stack([x, z]) + jitter
    new_vel = np.column_stack([rng.normal(0.0, 0.05, n), rng.normal(-1.8, 0.08, n)])
    new_ids = np.arange(next_id, next_id + n, dtype=np.int64)
    return (np.vstack([pos, new_pos]), np.vstack([vel, new_vel]),
            np.concatenate([ids, new_ids]), next_id + n, n)


def pair_dynamics(pos: np.ndarray, vel: np.ndarray, p: Params):
    n = len(pos)
    acc = np.zeros_like(pos)
    if n < 2:
        return acc, 0, np.zeros(n, dtype=int)
    re = p.re_ratio * p.l0
    tree = cKDTree(pos)
    pairs = np.array(list(tree.query_pairs(re)), dtype=np.int64)
    counts = np.zeros(n, dtype=int)
    if pairs.size == 0:
        return acc, 0, counts
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[j] - pos[i]
    r = np.linalg.norm(d, axis=1)
    valid = r > 1e-10
    i, j, d, r = i[valid], j[valid], d[valid], r[valid]
    counts += np.bincount(i, minlength=n)
    counts += np.bincount(j, minlength=n)
    e = d / r[:, None]

    # Penalty pressure: active inside nominal spacing, used only as a stable
    # visual baseline before integrating the WLS-PPE pressure projection.
    overlap = np.maximum(p.l0 - r, 0.0)
    reln = np.sum((vel[j] - vel[i]) * e, axis=1)
    fmag = p.pair_stiffness * overlap - p.pair_damping * np.minimum(reln, 0.0)
    force = fmag[:, None] * e
    np.add.at(acc, i, -force)
    np.add.at(acc, j, force)

    # Pairwise viscosity within the support radius.
    q = np.maximum(1.0 - r / re, 0.0)
    visc = p.viscosity * q[:, None] * (vel[j] - vel[i])
    np.add.at(acc, i, visc)
    np.add.at(acc, j, -visc)
    return acc, len(i), counts


def enforce_plate(pos: np.ndarray, vel: np.ndarray, p: Params):
    t, nvec = plate_vectors(p)
    x = pos[:, 0]
    active = (x >= p.plate_x0) & (x <= p.plate_x1)
    surface = plate_z(x, p)
    signed = pos[:, 1] - surface
    hit = active & (signed < 0.50 * p.l0)
    if np.any(hit):
        correction = 0.50 * p.l0 - signed[hit]
        pos[hit] += correction[:, None] * nvec
        vn = np.sum(vel[hit] * nvec, axis=1)
        vt = np.sum(vel[hit] * t, axis=1)
        vn_new = np.where(vn < 0.0, -p.restitution * vn, vn)
        # Tangential damping represents no-slip drag without pinning the film.
        vt_new = p.wall_friction * vt
        vel[hit] = vt_new[:, None] * t + vn_new[:, None] * nvec
    return int(np.count_nonzero(hit))


def remove_outflow(pos: np.ndarray, vel: np.ndarray, ids: np.ndarray, p: Params):
    keep = ((pos[:, 0] > -0.08) & (pos[:, 0] < 1.10) &
            (pos[:, 1] > -0.08) & (pos[:, 1] < p.height + 0.10))
    removed = int(np.count_nonzero(~keep))
    return pos[keep], vel[keep], ids[keep], removed


def film_metrics(pos: np.ndarray, p: Params):
    if len(pos) == 0:
        return 0.0, 0.0, 0
    x = pos[:, 0]
    surface = plate_z(x, p)
    dist = pos[:, 1] - surface
    mask = ((x >= p.plate_x0) & (x <= p.plate_x1) &
            (dist >= 0.0) & (dist < 5.0 * p.l0))
    if not np.any(mask):
        return 0.0, 0.0, 0
    bins = np.linspace(p.plate_x0, p.plate_x1, 31)
    thickness = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = mask & (x >= a) & (x < b)
        if np.any(m):
            thickness.append(float(np.max(dist[m]) + p.l0))
    return (float(np.mean(thickness)) if thickness else 0.0,
            float(np.max(thickness)) if thickness else 0.0,
            int(np.count_nonzero(mask)))


def render_frame(pos, vel, counts, step, stats, out_path: Path, p: Params):
    speed = np.linalg.norm(vel, axis=1) if len(vel) else np.array([])
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=120)
    xx = np.linspace(p.plate_x0, p.plate_x1, 300)
    zz = plate_z(xx, p)
    ax.fill_between(xx, -0.05, zz, color="#ddd6be")
    ax.plot(xx, zz, color="0.25", linewidth=2.0)
    if len(pos):
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=speed, s=8, cmap="turbo",
                        vmin=0.0, vmax=3.0, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.015)
        cb.set_label("Speed [m/s]")
    ax.set_xlim(0, p.width)
    ax.set_ylim(0, p.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Inclined-plate rain benchmark — step {step}")
    text = (f"particles={len(pos)}  injected={stats['injected_total']}  "
            f"outflow={stats['outflow_total']}\n"
            f"neighbor pairs={stats['pairs']}  film particles={stats['film_particles']}  "
            f"h_mean={stats['film_mean']*1000:.1f} mm")
    ax.text(0.015, 0.985, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(p: Params, out: Path):
    rng = np.random.default_rng(p.seed)
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    pos = np.empty((0, 2), dtype=float)
    vel = np.empty((0, 2), dtype=float)
    ids = np.empty((0,), dtype=np.int64)
    next_id = 0
    injected_total = 0
    outflow_total = 0
    rows = []
    frame_paths = []

    for step in range(p.steps):
        pos, vel, ids, next_id, nin = inject(pos, vel, ids, next_id, step, p, rng)
        injected_total += nin
        acc, pairs, counts = pair_dynamics(pos, vel, p)
        if len(pos):
            acc[:, 1] -= p.gravity
            vel += p.dt * acc
            speed = np.linalg.norm(vel, axis=1)
            fast = speed > p.max_speed
            if np.any(fast):
                vel[fast] *= (p.max_speed / speed[fast])[:, None]
            pos += p.dt * vel
            hits = enforce_plate(pos, vel, p)
            # Light side-wall handling on the left; right side is the outlet.
            left = pos[:, 0] < 0.02
            pos[left, 0] = 0.02
            vel[left, 0] = np.abs(vel[left, 0]) * 0.1
        else:
            hits = 0
        pos, vel, ids, nout = remove_outflow(pos, vel, ids, p)
        outflow_total += nout
        # counts are from pre-removal positions; only histogram aggregates are recorded.
        film_mean, film_max, film_particles = film_metrics(pos, p)
        vmax = float(np.max(np.linalg.norm(vel, axis=1))) if len(vel) else 0.0
        mass_error = injected_total - outflow_total - len(pos)
        row = dict(step=step, time=step*p.dt, particles=len(pos), injected_total=injected_total,
                   outflow_total=outflow_total, mass_error=mass_error, pairs=pairs,
                   neighbor_mean=float(np.mean(counts)) if len(counts) else 0.0,
                   wall_contacts=hits, film_particles=film_particles,
                   film_mean=film_mean, film_max=film_max, max_speed=vmax)
        rows.append(row)
        if step % p.save_every == 0 or step == p.steps - 1:
            fp = frames / f"frame_{step:04d}.png"
            render_frame(pos, vel, counts, step, row, fp, p)
            frame_paths.append(fp)

    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {
        "case": "inclined_plate_rain",
        "model": "CPU particle visual baseline; penalty pressure pending WLS-PPE integration",
        "steps": p.steps,
        "dt": p.dt,
        "injected": injected_total,
        "outflow": outflow_total,
        "remaining": len(pos),
        "mass_balance_error_particles": injected_total - outflow_total - len(pos),
        "final_film_mean_m": rows[-1]["film_mean"],
        "final_film_max_m": rows[-1]["film_max"],
        "max_speed_mps": max(r["max_speed"] for r in rows),
        "frames": len(frame_paths),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Build GIF from rendered PNG frames.
    from PIL import Image
    images = [Image.open(f).convert("P", palette=Image.Palette.ADAPTIVE) for f in frame_paths]
    images[0].save(out / "inclined_plate_rain.gif", save_all=True,
                   append_images=images[1:], duration=95, loop=0, optimize=False)
    for im in images:
        im.close()
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/rain_plate")
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--inject-per-step", type=int, default=12)
    args = ap.parse_args()
    p = Params(steps=args.steps, inject_per_step=args.inject_per_step)
    run(p, Path(args.output))


if __name__ == "__main__":
    main()

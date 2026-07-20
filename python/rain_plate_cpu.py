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
from typing import Any
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
    target_inflow_m3s: float = 0.013824  # 12 * 0.012^3 / 0.0015
    preserve_physical_inflow: bool = True
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
    particle_volume: float = 1.728e-6  # effective 3D volume represented by one particle


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
           step: int, p: Params, rng: np.random.Generator, volume_credit: float):
    if step >= p.inject_steps:
        return pos, vel, ids, next_id, 0, volume_credit
    if p.preserve_physical_inflow:
        volume_credit += p.target_inflow_m3s * p.dt
        n = int(volume_credit // p.particle_volume)
        volume_credit -= n * p.particle_volume
    else:
        n = p.inject_per_step
    if n <= 0:
        return pos, vel, ids, next_id, 0, volume_credit
    # Three bands produce a visible rain curtain while keeping deterministic load.
    x = rng.uniform(0.16, 0.78, n)
    z = rng.uniform(0.58, 0.70, n)
    jitter = rng.normal(0.0, 0.0015, (n, 2))
    new_pos = np.column_stack([x, z]) + jitter
    new_vel = np.column_stack([rng.normal(0.0, 0.05, n), rng.normal(-1.8, 0.08, n)])
    new_ids = np.arange(next_id, next_id + n, dtype=np.int64)
    return (np.vstack([pos, new_pos]), np.vstack([vel, new_vel]),
            np.concatenate([ids, new_ids]), next_id + n, n, volume_credit)


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
    """Remove particles and distinguish intended outlet removal from numerical loss."""
    outlet = (pos[:, 0] >= 1.10) & (pos[:, 1] > -0.08) & (pos[:, 1] < p.height + 0.10)
    invalid = ((pos[:, 0] <= -0.08) | (pos[:, 1] <= -0.08) |
               (pos[:, 1] >= p.height + 0.10))
    remove = outlet | invalid
    keep = ~remove
    return (pos[keep], vel[keep], ids[keep],
            int(np.count_nonzero(outlet)), int(np.count_nonzero(invalid)))


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



def plate_coordinates(pos: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized tangential coordinate u in [0,1] and wall-normal distance."""
    if len(pos) == 0:
        return np.empty(0), np.empty(0)
    t, nvec = plate_vectors(p)
    origin = np.array([p.plate_x0, p.plate_left_z], dtype=float)
    rel = pos - origin
    length = float(np.linalg.norm(np.array([p.plate_x1-p.plate_x0, p.plate_right_z-p.plate_left_z])))
    u = (rel @ t) / max(length, 1.0e-12)
    d = rel @ nvec
    return u, d


def load_validation_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {"sampling_windows": [], "flow_sections": [], "coverage_cells": 32,
                "wet_distance_factor": 3.0, "min_particles_per_cell": 1}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_window_metrics(pos: np.ndarray, p: Params, cfg: dict[str, Any], time: float) -> list[dict[str, Any]]:
    u, d = plate_coordinates(pos, p)
    rows: list[dict[str, Any]] = []
    wet_dist = float(cfg.get("wet_distance_factor", 3.0)) * p.l0
    default_cells = int(cfg.get("coverage_cells", 32))
    min_per_cell = int(cfg.get("min_particles_per_cell", 1))
    for w in cfg.get("sampling_windows", []):
        a, b = map(float, w["u_range"])
        cells = int(w.get("coverage_cells", default_cells))
        mask = (u >= a) & (u < b) & (d >= 0.0) & (d <= wet_dist)
        counts, _ = np.histogram(u[mask], bins=np.linspace(a, b, cells + 1))
        coverage = float(np.count_nonzero(counts >= min_per_cell) / max(cells, 1))
        thickness = []
        if np.any(mask):
            for lo, hi in zip(np.linspace(a,b,cells+1)[:-1], np.linspace(a,b,cells+1)[1:]):
                mm = mask & (u >= lo) & (u < hi)
                if np.any(mm):
                    thickness.append(float(np.max(d[mm]) + p.l0))
        rows.append({
            "time": time, "window": w["name"], "particle_count": int(np.count_nonzero(mask)),
            "fluid_volume_m3": float(np.count_nonzero(mask) * p.particle_volume),
            "coverage_ratio": coverage,
            "film_mean_m": float(np.mean(thickness)) if thickness else 0.0,
            "film_p90_m": float(np.quantile(thickness, 0.90)) if thickness else 0.0,
            "film_max_m": float(np.max(thickness)) if thickness else 0.0,
        })
    return rows


def compute_section_crossings(prev_pos: np.ndarray, prev_ids: np.ndarray, pos: np.ndarray, ids: np.ndarray,
                              p: Params, cfg: dict[str, Any], dt: float, time: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(prev_ids) == 0 or len(ids) == 0:
        for sec in cfg.get("flow_sections", []):
            rows.append({"time": time, "section": sec["name"], "crossing_particle_count": 0,
                         "instantaneous_flow_rate_m3s": 0.0, "crossing_volume_m3": 0.0})
        return rows
    common, ip, ic = np.intersect1d(prev_ids, ids, assume_unique=True, return_indices=True)
    if len(common) == 0:
        return rows
    up, _ = plate_coordinates(prev_pos[ip], p)
    uc, dc = plate_coordinates(pos[ic], p)
    for sec in cfg.get("flow_sections", []):
        s = float(sec["u"])
        max_d = float(sec.get("max_normal_distance_factor", 5.0)) * p.l0
        crossed = (up < s) & (uc >= s) & (dc >= 0.0) & (dc <= max_d)
        n = int(np.count_nonzero(crossed))
        vol = n * p.particle_volume
        rows.append({"time": time, "section": sec["name"], "crossing_particle_count": n,
                     "instantaneous_flow_rate_m3s": vol / dt, "crossing_volume_m3": vol})
    return rows

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


def run(p: Params, out: Path, validation_cfg: dict[str, Any] | None = None):
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
    invalid_removed_total = 0
    volume_credit = 0.0
    rows = []
    frame_paths = []
    validation_cfg = validation_cfg or {"sampling_windows": [], "flow_sections": []}
    window_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    cumulative_section_volume: dict[str, float] = {s["name"]: 0.0 for s in validation_cfg.get("flow_sections", [])}

    for step in range(p.steps):
        prev_pos = pos.copy()
        prev_ids = ids.copy()
        pos, vel, ids, next_id, nin, volume_credit = inject(
            pos, vel, ids, next_id, step, p, rng, volume_credit)
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
        pos, vel, ids, nout, ninvalid = remove_outflow(pos, vel, ids, p)
        outflow_total += nout
        invalid_removed_total += ninvalid
        # counts are from pre-removal positions; only histogram aggregates are recorded.
        film_mean, film_max, film_particles = film_metrics(pos, p)
        vmax = float(np.max(np.linalg.norm(vel, axis=1))) if len(vel) else 0.0
        mass_error = injected_total - outflow_total - invalid_removed_total - len(pos)
        row = dict(step=step, time=(step+1)*p.dt, particles=len(pos), injected_total=injected_total,
                   outflow_total=outflow_total, invalid_removed_total=invalid_removed_total,
                   mass_error=mass_error, pairs=pairs,
                   neighbor_mean=float(np.mean(counts)) if len(counts) else 0.0,
                   wall_contacts=hits, film_particles=film_particles,
                   film_mean=film_mean, film_max=film_max, max_speed=vmax)
        rows.append(row)
        now = (step + 1) * p.dt
        window_rows.extend(compute_window_metrics(pos, p, validation_cfg, now))
        sec_step = compute_section_crossings(prev_pos, prev_ids, pos, ids, p, validation_cfg, p.dt, now)
        for sr in sec_step:
            cumulative_section_volume[sr["section"]] = cumulative_section_volume.get(sr["section"], 0.0) + sr["crossing_volume_m3"]
            sr["cumulative_volume_m3"] = cumulative_section_volume[sr["section"]]
        section_rows.extend(sec_step)
        if step % p.save_every == 0 or step == p.steps - 1:
            fp = frames / f"frame_{step:04d}.png"
            render_frame(pos, vel, counts, step, row, fp, p)
            frame_paths.append(fp)

    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    if window_rows:
        with (out / "sampling_windows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys())); w.writeheader(); w.writerows(window_rows)
    if section_rows:
        with (out / "flow_sections.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(section_rows[0].keys())); w.writeheader(); w.writerows(section_rows)
    mass_rows = []
    for r in rows:
        injected_v = r["injected_total"] * p.particle_volume
        remaining_v = r["particles"] * p.particle_volume
        out_v = r["outflow_total"] * p.particle_volume
        invalid_v = r["invalid_removed_total"] * p.particle_volume
        denom = max(injected_v, p.particle_volume)
        balance = injected_v - remaining_v - out_v - invalid_v
        mass_rows.append({"time": r["time"], "injected_volume_m3": injected_v,
                          "remaining_volume_m3": remaining_v, "outlet_volume_m3": out_v,
                          "invalid_removed_volume_m3": invalid_v,
                          "balance_error_m3": balance,
                          "relative_mass_error": abs(balance)/denom,
                          "invalid_removed_ratio": invalid_v/denom})
    with (out / "mass_conservation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mass_rows[0].keys())); w.writeheader(); w.writerows(mass_rows)
    summary = {
        "case": "inclined_plate_rain",
        "model": "CPU particle visual baseline; penalty pressure pending WLS-PPE integration",
        "steps": p.steps,
        "dt": p.dt,
        "injected": injected_total,
        "outflow": outflow_total,
        "invalid_removed": invalid_removed_total,
        "remaining": len(pos),
        "mass_balance_error_particles": injected_total - outflow_total - invalid_removed_total - len(pos),
        "target_inflow_m3s": p.target_inflow_m3s,
        "realized_injected_volume_m3": injected_total * p.particle_volume,
        "final_film_mean_m": rows[-1]["film_mean"],
        "final_film_max_m": rows[-1]["film_max"],
        "max_speed_mps": max(r["max_speed"] for r in rows),
        "frames": len(frame_paths),
        "sampling_windows": [w["name"] for w in validation_cfg.get("sampling_windows", [])],
        "flow_sections": [s["name"] for s in validation_cfg.get("flow_sections", [])],
        "max_relative_mass_error": max(r["relative_mass_error"] for r in mass_rows),
        "max_invalid_removed_ratio": max(r["invalid_removed_ratio"] for r in mass_rows),
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
    ap.add_argument("--l0", type=float, default=0.012)
    ap.add_argument("--dt", type=float, default=0.0015)
    ap.add_argument("--validation-config", default="config/rain_plate_validation.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-inflow-m3s", type=float, default=0.013824)
    ap.add_argument("--fixed-particles-per-step", action="store_true",
                    help="Disable physical inflow preservation and inject a fixed count each step")
    args = ap.parse_args()
    p = Params(steps=args.steps, inject_per_step=args.inject_per_step, l0=args.l0, dt=args.dt,
               particle_volume=args.l0**3, seed=args.seed,
               target_inflow_m3s=args.target_inflow_m3s,
               preserve_physical_inflow=not args.fixed_particles_per_step)
    run(p, Path(args.output), load_validation_config(args.validation_config))


if __name__ == "__main__":
    main()

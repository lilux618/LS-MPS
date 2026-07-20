#!/usr/bin/env python3
"""2-D LS-MPS inclined-plate rain CPU reference (beta).

This program closes the numerical loop used by the automotive rain benchmark:
continuous injection -> neighbor search -> free-surface classification ->
LS-MPS viscous prediction -> WLS pressure Poisson assembly -> pressure
projection -> wall contact -> sampling-window / flow / conservation outputs.

It is deliberately a small, diagnosable CPU golden-reference candidate, not an
industrial 3-D rain solver. The implementation uses normalized quadratic WLS
with five unknown derivatives in 2-D and sparse BiCGStab pressure solves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab

# Reuse the validated output definitions from v0.6.
from rain_plate_cpu import (
    Params as BaseParams,
    compute_section_crossings,
    compute_window_metrics,
    enforce_plate,
    film_metrics,
    inject,
    load_validation_config,
    plate_coordinates,
    plate_vectors,
    plate_z,
    remove_outflow,
    render_frame,
)


@dataclass
class Params(BaseParams):
    re_ratio: float = 3.0
    density: float = 1000.0
    kinematic_viscosity: float = 1.0e-6
    pressure_tolerance: float = 1.0e-7
    pressure_max_iterations: int = 250
    min_neighbors: int = 5
    free_surface_neighbor_ratio: float = 0.72
    regularization: float = 1.0e-7
    matrix_condition_warning: float = 1.0e8
    max_pressure: float = 2.0e6
    shifting_strength: float = 0.015
    shifting_limit_factor: float = 0.08


def basis(offset: np.ndarray, h: float) -> np.ndarray:
    qx, qz = offset / h
    return np.array([qx, qz, qx*qx, qx*qz, qz*qz], dtype=float)


def weight(r: float, re: float) -> float:
    if r <= 1.0e-14 or r >= re:
        return 0.0
    q = 1.0 - r / re
    return q*q


def build_neighbors(pos: np.ndarray, re: float) -> tuple[list[np.ndarray], np.ndarray, int]:
    n = len(pos)
    if n == 0:
        return [], np.empty(0, dtype=int), 0
    tree = cKDTree(pos)
    raw = tree.query_ball_point(pos, re)
    nbrs: list[np.ndarray] = []
    counts = np.zeros(n, dtype=int)
    pairs = 0
    for i, row in enumerate(raw):
        a = np.asarray([j for j in row if j != i], dtype=np.int64)
        nbrs.append(a)
        counts[i] = len(a)
        pairs += int(np.count_nonzero(a > i))
    return nbrs, counts, pairs


def near_plate_virtual_offset(x: np.ndarray, p: Params) -> np.ndarray | None:
    if x[0] < p.plate_x0 or x[0] > p.plate_x1:
        return None
    _, nvec = plate_vectors(p)
    surface = np.array([x[0], float(plate_z(x[0], p))])
    d = float(np.dot(x - surface, nvec))
    if d < 0.0 or d > p.re_ratio * p.l0:
        return None
    # Mirror location across the plate. Offset is from the real particle to mirror.
    return -2.0 * d * nvec


def local_operators(pos: np.ndarray, nbrs: list[np.ndarray], p: Params):
    """Build inverse moment matrices and geometry diagnostics per particle."""
    n = len(pos)
    inv = np.zeros((n, 5, 5), dtype=float)
    valid = np.zeros(n, dtype=bool)
    cond = np.full(n, np.inf, dtype=float)
    virtual = np.zeros(n, dtype=int)
    re = p.re_ratio * p.l0
    for i in range(n):
        M = np.zeros((5, 5), dtype=float)
        for j in nbrs[i]:
            off = pos[j] - pos[i]
            w = weight(float(np.linalg.norm(off)), re)
            b = basis(off, p.l0)
            M += w * np.outer(b, b)
        voff = near_plate_virtual_offset(pos[i], p)
        if voff is not None and np.linalg.norm(voff) > 1.0e-12:
            w = weight(float(np.linalg.norm(voff)), re)
            if w > 0:
                b = basis(voff, p.l0)
                M += w * np.outer(b, b)
                virtual[i] = 1
        if len(nbrs[i]) + virtual[i] < p.min_neighbors:
            continue
        scale = max(float(np.trace(M)) / 5.0, 1.0)
        Mr = M + p.regularization * scale * np.eye(5)
        try:
            eig = np.linalg.eigvalsh(Mr)
            c = float(eig[-1] / max(eig[0], 1.0e-18))
            cond[i] = c
            inv[i] = np.linalg.inv(Mr)
            valid[i] = np.all(np.isfinite(inv[i]))
        except np.linalg.LinAlgError:
            pass
    return inv, valid, cond, virtual


def classify_surface(counts: np.ndarray, nbrs: list[np.ndarray], pos: np.ndarray, p: Params) -> np.ndarray:
    """0 internal, 1 free surface, 2 splash, 3 near free surface."""
    n = len(counts)
    state = np.zeros(n, dtype=np.int8)
    if n == 0:
        return state
    # Robust reference count from the upper quartile, avoiding sparse rain droplets.
    positive = counts[counts > 0]
    ref = float(np.quantile(positive, 0.75)) if len(positive) else 1.0
    free_cut = max(p.min_neighbors, int(math.ceil(p.free_surface_neighbor_ratio * ref)))
    splash_cut = max(3, p.min_neighbors // 2)
    for i, c in enumerate(counts):
        u, d = plate_coordinates(pos[i:i+1], p)
        near_wall = len(d) and 0.0 <= d[0] <= 2.0*p.l0
        if c <= splash_cut and not near_wall:
            state[i] = 2
        elif c < free_cut:
            state[i] = 1
    # One neighbor layer around free surface.
    primary = state.copy()
    for i in range(n):
        if primary[i] == 0 and any(primary[j] == 1 for j in nbrs[i]):
            state[i] = 3
    return state


def scalar_derivatives(values: np.ndarray, pos: np.ndarray, nbrs: list[np.ndarray], inv: np.ndarray,
                       valid: np.ndarray, p: Params) -> np.ndarray:
    """Return [fx, fz, fxx coefficient, fxz coefficient, fzz coefficient]."""
    n = len(pos)
    out = np.zeros((n, 5), dtype=float)
    re = p.re_ratio * p.l0
    for i in range(n):
        if not valid[i]:
            continue
        rhs = np.zeros(5, dtype=float)
        for j in nbrs[i]:
            off = pos[j] - pos[i]
            w = weight(float(np.linalg.norm(off)), re)
            rhs += w * basis(off, p.l0) * (values[j] - values[i])
        # Mirrored wall value: zero normal gradient, approximated by equal scalar value.
        out[i] = inv[i] @ rhs
    return out


def provisional_velocity(pos: np.ndarray, vel: np.ndarray, nbrs: list[np.ndarray], inv: np.ndarray,
                         valid: np.ndarray, p: Params) -> np.ndarray:
    ux = scalar_derivatives(vel[:, 0], pos, nbrs, inv, valid, p)
    uz = scalar_derivatives(vel[:, 1], pos, nbrs, inv, valid, p)
    lap_u = np.column_stack([
        2.0*(ux[:, 2] + ux[:, 4])/(p.l0*p.l0),
        2.0*(uz[:, 2] + uz[:, 4])/(p.l0*p.l0),
    ])
    acc = p.kinematic_viscosity * lap_u
    acc[:, 1] -= p.gravity
    return vel + p.dt * acc


def divergence(field: np.ndarray, pos: np.ndarray, nbrs: list[np.ndarray], inv: np.ndarray,
               valid: np.ndarray, p: Params) -> np.ndarray:
    dx = scalar_derivatives(field[:, 0], pos, nbrs, inv, valid, p)
    dz = scalar_derivatives(field[:, 1], pos, nbrs, inv, valid, p)
    return dx[:, 0]/p.l0 + dz[:, 1]/p.l0


def solve_pressure(pos: np.ndarray, ustar: np.ndarray, nbrs: list[np.ndarray], inv: np.ndarray,
                   valid: np.ndarray, state: np.ndarray, p: Params):
    n = len(pos)
    if n == 0:
        return np.empty(0), 0, 0.0, True, 0
    div = divergence(ustar, pos, nbrs, inv, valid, p)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n, dtype=float)
    re = p.re_ratio*p.l0
    fallback = 0
    for i in range(n):
        # Free/splash particles are atmospheric pressure. Invalid WLS rows are pinned.
        if state[i] in (1, 2) or not valid[i]:
            rows.append(i); cols.append(i); vals.append(1.0)
            rhs[i] = 0.0
            if not valid[i]:
                fallback += 1
            continue
        lap_row = 2.0*(inv[i][2, :] + inv[i][4, :])/(p.l0*p.l0)
        diag = 0.0
        for j in nbrs[i]:
            off = pos[j]-pos[i]
            w = weight(float(np.linalg.norm(off)), re)
            c = float(w * np.dot(lap_row, basis(off, p.l0)))
            if not np.isfinite(c):
                continue
            rows.append(i); cols.append(int(j)); vals.append(c)
            diag -= c
        # Weak diagonal stabilization for isolated/near-singular rows.
        if abs(diag) < 1.0e-12:
            rows.append(i); cols.append(i); vals.append(1.0)
            rhs[i] = 0.0
            fallback += 1
        else:
            rows.append(i); cols.append(i); vals.append(diag)
            rhs[i] = p.density/p.dt * div[i]
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    it = [0]
    def cb(_):
        it[0] += 1
    pressure, info = bicgstab(A, rhs, rtol=p.pressure_tolerance, atol=0.0,
                              maxiter=p.pressure_max_iterations, callback=cb)
    converged = info == 0 and np.all(np.isfinite(pressure))
    if not converged:
        pressure = np.zeros(n, dtype=float)
    residual = float(np.linalg.norm(A@pressure-rhs) / max(np.linalg.norm(rhs), 1.0e-30))
    pressure = np.clip(pressure, -p.max_pressure, p.max_pressure)
    return pressure, it[0] if info == 0 else int(abs(info)), residual, converged, fallback


def pressure_correct(pos: np.ndarray, ustar: np.ndarray, pressure: np.ndarray, nbrs: list[np.ndarray],
                     inv: np.ndarray, valid: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    dp = scalar_derivatives(pressure, pos, nbrs, inv, valid, p)
    grad = np.column_stack([dp[:, 0]/p.l0, dp[:, 1]/p.l0])
    corrected = ustar - p.dt/p.density * grad
    return corrected, grad


def particle_shift(pos: np.ndarray, nbrs: list[np.ndarray], state: np.ndarray, p: Params) -> tuple[np.ndarray, int, float]:
    if p.shifting_strength <= 0 or len(pos) == 0:
        return pos, 0, 0.0
    disp = np.zeros_like(pos)
    re = p.re_ratio*p.l0
    for i in range(len(pos)):
        if state[i] == 2:  # do not shift isolated splash droplets
            continue
        for j in nbrs[i]:
            d = pos[i]-pos[j]
            r = float(np.linalg.norm(d))
            if 1.0e-12 < r < re:
                disp[i] += weight(r, re) * d/r
    mag = np.linalg.norm(disp, axis=1)
    nonzero = mag > 0
    if np.any(nonzero):
        disp[nonzero] *= (p.shifting_strength*p.l0/(1.0+mag[nonzero]))[:, None]
        lim = p.shifting_limit_factor*p.l0
        dm = np.linalg.norm(disp, axis=1)
        over = dm > lim
        disp[over] *= (lim/dm[over])[:, None]
    pos2 = pos + disp
    max_shift = float(np.max(np.linalg.norm(disp, axis=1))) if len(pos) else 0.0
    return pos2, int(np.count_nonzero(np.linalg.norm(disp, axis=1)>0)), max_shift


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(p: Params, out: Path, validation_cfg: dict[str, Any]):
    rng = np.random.default_rng(p.seed)
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "frames"; frames.mkdir(exist_ok=True)
    pos = np.empty((0,2)); vel = np.empty((0,2)); ids = np.empty(0, dtype=np.int64)
    next_id = injected_total = outflow_total = invalid_total = 0
    volume_credit = 0.0
    metrics: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    cumulative = {s["name"]: 0.0 for s in validation_cfg.get("flow_sections", [])}
    frame_paths: list[Path] = []

    for step in range(p.steps):
        prev_pos, prev_ids = pos.copy(), ids.copy()
        pos, vel, ids, next_id, nin, volume_credit = inject(pos, vel, ids, next_id, step, p, rng, volume_credit)
        injected_total += nin
        nbrs, counts, pairs = build_neighbors(pos, p.re_ratio*p.l0)
        inv, valid, cond, virtual = local_operators(pos, nbrs, p)
        state = classify_surface(counts, nbrs, pos, p)
        ustar = provisional_velocity(pos, vel, nbrs, inv, valid, p) if len(pos) else vel
        div_before = divergence(ustar, pos, nbrs, inv, valid, p) if len(pos) else np.empty(0)
        pressure, ppe_it, ppe_res, ppe_ok, fallback = solve_pressure(pos, ustar, nbrs, inv, valid, state, p)
        projected, pgrad = pressure_correct(pos, ustar, pressure, nbrs, inv, valid, p) if len(pos) else (vel, np.empty((0,2)))
        # WLS Laplacian and gradient are independently reconstructed on an irregular
        # cloud, so use a deterministic projection line search to avoid increasing
        # the discrete divergence during early reference development.
        projection_alpha = 0.0
        vel = ustar.copy()
        best_div = float(np.linalg.norm(div_before)) if len(div_before) else 0.0
        if len(pos):
            full_delta = projected - ustar
            for alpha in (1.0, 0.5, 0.25, 0.1):
                trial = ustar + alpha*full_delta
                trial_div = divergence(trial, pos, nbrs, inv, valid, p)
                score = float(np.linalg.norm(trial_div))
                if score < best_div:
                    best_div = score; vel = trial; projection_alpha = alpha
        speed = np.linalg.norm(vel, axis=1) if len(vel) else np.empty(0)
        fast = speed > p.max_speed
        if np.any(fast): vel[fast] *= (p.max_speed/speed[fast])[:,None]
        div_after = divergence(vel, pos, nbrs, inv, valid, p) if len(pos) else np.empty(0)
        pos += p.dt*vel
        pos, shifted, max_shift = particle_shift(pos, nbrs, state, p)
        hits = enforce_plate(pos, vel, p) if len(pos) else 0
        left = pos[:,0] < 0.02
        pos[left,0] = 0.02; vel[left,0] = np.abs(vel[left,0])*0.1
        pos, vel, ids, nout, ninvalid = remove_outflow(pos, vel, ids, p)
        outflow_total += nout; invalid_total += ninvalid
        film_mean, film_max, film_particles = film_metrics(pos,p)
        now=(step+1)*p.dt
        cond_finite=cond[np.isfinite(cond)]
        row={
            "step":step,"time":now,"particles":len(pos),"injected_total":injected_total,
            "outflow_total":outflow_total,"invalid_removed_total":invalid_total,
            "mass_error":injected_total-outflow_total-invalid_total-len(pos),"pairs":pairs,
            "neighbor_mean":float(np.mean(counts)) if len(counts) else 0.0,
            "neighbor_p90":float(np.quantile(counts,0.9)) if len(counts) else 0.0,
            "neighbor_p99":float(np.quantile(counts,0.99)) if len(counts) else 0.0,
            "internal_count":int(np.count_nonzero(state==0)),"free_surface_count":int(np.count_nonzero(state==1)),
            "splash_count":int(np.count_nonzero(state==2)),"near_surface_count":int(np.count_nonzero(state==3)),
            "wls_valid_ratio":float(np.mean(valid)) if len(valid) else 1.0,
            "wls_ill_conditioned_ratio":float(np.mean(cond>p.matrix_condition_warning)) if len(cond) else 0.0,
            "wls_condition_p99":float(np.quantile(cond_finite,0.99)) if len(cond_finite) else 0.0,
            "virtual_particle_ratio":float(np.mean(virtual>0)) if len(virtual) else 0.0,
            "ppe_iterations":ppe_it,"ppe_relative_residual":ppe_res,"ppe_converged":int(ppe_ok),
            "ppe_fallback_rows":fallback,"projection_alpha":projection_alpha,
            "div_l2_before":float(np.linalg.norm(div_before)) if len(div_before) else 0.0,
            "div_l2_after":float(np.linalg.norm(div_after)) if len(div_after) else 0.0,
            "wall_contacts":hits,"shifted_particles":shifted,"max_shift":max_shift,
            "film_particles":film_particles,"film_mean":film_mean,"film_max":film_max,
            "max_speed":float(np.max(np.linalg.norm(vel,axis=1))) if len(vel) else 0.0,
        }
        metrics.append(row)
        window_rows.extend(compute_window_metrics(pos,p,validation_cfg,now))
        sec=compute_section_crossings(prev_pos,prev_ids,pos,ids,p,validation_cfg,p.dt,now)
        for sr in sec:
            cumulative[sr["section"]]+=sr["crossing_volume_m3"]
            sr["cumulative_volume_m3"]=cumulative[sr["section"]]
        section_rows.extend(sec)
        if step % p.save_every==0 or step==p.steps-1:
            fp=frames/f"frame_{step:04d}.png"
            render_frame(pos,vel,counts[:len(pos)] if len(counts)>=len(pos) else np.zeros(len(pos)),step,row,fp,p)
            frame_paths.append(fp)

    write_csv(out/"metrics.csv",metrics)
    write_csv(out/"sampling_windows.csv",window_rows)
    write_csv(out/"flow_sections.csv",section_rows)
    mass=[]
    for r in metrics:
        iv=r["injected_total"]*p.particle_volume; rv=r["particles"]*p.particle_volume
        ov=r["outflow_total"]*p.particle_volume; xv=r["invalid_removed_total"]*p.particle_volume
        bal=iv-rv-ov-xv; den=max(iv,p.particle_volume)
        mass.append({"time":r["time"],"injected_volume_m3":iv,"remaining_volume_m3":rv,
                     "outlet_volume_m3":ov,"invalid_removed_volume_m3":xv,
                     "balance_error_m3":bal,"relative_mass_error":abs(bal)/den,
                     "invalid_removed_ratio":xv/den})
    write_csv(out/"mass_conservation.csv",mass)
    summary={
        "case":"inclined_plate_rain","model":"2-D LS-MPS pressure-projection CPU reference beta",
        "steps":p.steps,"dt":p.dt,"injected":injected_total,"outflow":outflow_total,
        "invalid_removed":invalid_total,"remaining":len(pos),
        "mass_balance_error_particles":injected_total-outflow_total-invalid_total-len(pos),
        "max_relative_mass_error":max((x["relative_mass_error"] for x in mass),default=0.0),
        "max_invalid_removed_ratio":max((x["invalid_removed_ratio"] for x in mass),default=0.0),
        "ppe_converged_steps":sum(r["ppe_converged"] for r in metrics),
        "ppe_total_steps":len(metrics),"max_ppe_relative_residual":max((r["ppe_relative_residual"] for r in metrics),default=0.0),
        "mean_wls_valid_ratio":float(np.mean([r["wls_valid_ratio"] for r in metrics])) if metrics else 1.0,
        "mean_divergence_reduction":float(np.mean([r["div_l2_after"]/max(r["div_l2_before"],1e-30) for r in metrics])) if metrics else 0.0,
        "sampling_windows":[w["name"] for w in validation_cfg.get("sampling_windows",[])],
        "flow_sections":[s["name"] for s in validation_cfg.get("flow_sections",[])],
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if frame_paths:
        from PIL import Image
        images=[Image.open(f).convert("P",palette=Image.Palette.ADAPTIVE) for f in frame_paths]
        images[0].save(out/"inclined_plate_rain.gif",save_all=True,append_images=images[1:],duration=95,loop=0,optimize=False)
        for im in images: im.close()
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="outputs/rain_plate_lsmps_cpu")
    ap.add_argument("--steps",type=int,default=40)
    ap.add_argument("--inject-steps",type=int,default=25)
    ap.add_argument("--inject-per-step",type=int,default=4)
    ap.add_argument("--l0",type=float,default=0.016)
    ap.add_argument("--dt",type=float,default=0.001)
    ap.add_argument("--target-inflow-m3s",type=float,default=0.004096)
    ap.add_argument("--validation-config",default="config/rain_plate_validation.json")
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--no-shifting",action="store_true")
    a=ap.parse_args()
    p=Params(steps=a.steps,inject_steps=a.inject_steps,inject_per_step=a.inject_per_step,l0=a.l0,dt=a.dt,
             particle_volume=a.l0**3,target_inflow_m3s=a.target_inflow_m3s,seed=a.seed,
             preserve_physical_inflow=True,shifting_strength=0.0 if a.no_shifting else Params.shifting_strength)
    run(p,Path(a.output),load_validation_config(a.validation_config))

if __name__=="__main__": main()

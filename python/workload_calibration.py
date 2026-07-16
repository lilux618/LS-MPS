#!/usr/bin/env python3
"""Industrial workload calibration for LS-MPS proxy benchmarks.

The tool converts a customer/profile summary into two artifacts:
1. a calibrated parameterized proxy specification for a plate-rain case;
2. a synthetic CSR/type/timeline workload that reproduces the GPU-relevant
   statistics without requiring the customer's STL geometry.

It does not claim physical equivalence. It targets computational equivalence.
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


METRICS = [
    "particles",
    "neighbor_mean",
    "neighbor_p90",
    "neighbor_p99",
    "near_wall_ratio",
    "free_surface_ratio",
    "splash_ratio",
    "virtual_particle_ratio",
    "wls_ill_conditioned_ratio",
    "ppe_iterations",
    "cnl_rebuild_interval",
    "injected_per_step",
    "deleted_per_step",
]


@dataclass
class Knobs:
    particle_scale: float = 1.0
    support_scale: float = 1.0
    wall_complexity: float = 1.0
    neighbor_tail90: float = 1.0
    neighbor_tail99: float = 1.0
    surface_intensity: float = 1.0
    splash_intensity: float = 1.0
    virtual_scale: float = 1.0
    wls_difficulty: float = 1.0
    ppe_difficulty: float = 1.0
    injection_scale: float = 1.0
    deletion_scale: float = 1.0
    cnl_scale: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clip_ratio(x: float) -> float:
    return float(min(max(x, 0.0), 0.95))


def predict(base: dict[str, float], k: Knobs) -> dict[str, float]:
    # Support-radius changes approximately scale 3-D neighbor count by r^3.
    neighbor_factor = k.support_scale ** 3
    pred = dict(base)
    pred["particles"] = max(1.0, base["particles"] * k.particle_scale)
    pred["neighbor_mean"] = max(1.0, base["neighbor_mean"] * neighbor_factor)
    pred["neighbor_p90"] = max(pred["neighbor_mean"], base["neighbor_p90"] * neighbor_factor * k.neighbor_tail90)
    pred["neighbor_p99"] = max(pred["neighbor_p90"], base["neighbor_p99"] * neighbor_factor * k.neighbor_tail99)
    pred["near_wall_ratio"] = clip_ratio(base["near_wall_ratio"] * k.wall_complexity)
    pred["free_surface_ratio"] = clip_ratio(base["free_surface_ratio"] * k.surface_intensity)
    pred["splash_ratio"] = clip_ratio(base["splash_ratio"] * k.splash_intensity)
    pred["virtual_particle_ratio"] = clip_ratio(base["virtual_particle_ratio"] * k.virtual_scale)
    pred["wls_ill_conditioned_ratio"] = clip_ratio(base["wls_ill_conditioned_ratio"] * k.wls_difficulty)
    pred["ppe_iterations"] = max(1.0, base["ppe_iterations"] * k.ppe_difficulty)
    pred["cnl_rebuild_interval"] = max(1.0, base["cnl_rebuild_interval"] * k.cnl_scale)
    pred["injected_per_step"] = max(0.0, base["injected_per_step"] * k.injection_scale)
    pred["deleted_per_step"] = max(0.0, base["deleted_per_step"] * k.deletion_scale)
    return pred


def score(pred: dict[str, float], target: dict[str, float], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    errors: dict[str, float] = {}
    total = 0.0
    wsum = 0.0
    for m in METRICS:
        if m not in target or m not in pred:
            continue
        t = float(target[m])
        p = float(pred[m])
        scale = max(abs(t), 1.0 if m in {"particles", "ppe_iterations", "cnl_rebuild_interval", "injected_per_step", "deleted_per_step"} else 1e-3)
        e = abs(p - t) / scale
        w = float(weights.get(m, 1.0))
        errors[m] = e
        total += w * e * e
        wsum += w
    return math.sqrt(total / max(wsum, 1e-12)), errors


def calibrate(base: dict[str, float], target: dict[str, float], weights: dict[str, float], seed: int, iterations: int) -> tuple[Knobs, dict[str, float], float, dict[str, float]]:
    rng = np.random.default_rng(seed)
    # Start from a deterministic inverse mapping for mostly independent metrics.
    support = (max(target.get("neighbor_mean", base["neighbor_mean"]), 1e-9) / max(base["neighbor_mean"], 1e-9)) ** (1.0/3.0)
    init = Knobs(
        particle_scale=target.get("particles", base["particles"]) / max(base["particles"], 1e-9),
        support_scale=support,
        wall_complexity=target.get("near_wall_ratio", base["near_wall_ratio"]) / max(base["near_wall_ratio"], 1e-9),
        neighbor_tail90=target.get("neighbor_p90", base["neighbor_p90"]) / max(base["neighbor_p90"] * support**3, 1e-9),
        neighbor_tail99=target.get("neighbor_p99", base["neighbor_p99"]) / max(base["neighbor_p99"] * support**3, 1e-9),
        surface_intensity=target.get("free_surface_ratio", base["free_surface_ratio"]) / max(base["free_surface_ratio"], 1e-9),
        splash_intensity=target.get("splash_ratio", base["splash_ratio"]) / max(base["splash_ratio"], 1e-9),
        virtual_scale=target.get("virtual_particle_ratio", base["virtual_particle_ratio"]) / max(base["virtual_particle_ratio"], 1e-9),
        wls_difficulty=target.get("wls_ill_conditioned_ratio", base["wls_ill_conditioned_ratio"]) / max(base["wls_ill_conditioned_ratio"], 1e-9),
        ppe_difficulty=target.get("ppe_iterations", base["ppe_iterations"]) / max(base["ppe_iterations"], 1e-9),
        injection_scale=target.get("injected_per_step", base["injected_per_step"]) / max(base["injected_per_step"], 1e-9),
        deletion_scale=target.get("deleted_per_step", base["deleted_per_step"]) / max(base["deleted_per_step"], 1e-9),
        cnl_scale=target.get("cnl_rebuild_interval", base["cnl_rebuild_interval"]) / max(base["cnl_rebuild_interval"], 1e-9),
    )
    best = init
    best_pred = predict(base, best)
    best_score, best_errors = score(best_pred, target, weights)

    # Broad random search followed by local multiplicative refinement.
    bounds = {
        "particle_scale": (0.1, 100.0),
        "support_scale": (0.65, 1.55),
        "wall_complexity": (0.25, 6.0),
        "neighbor_tail90": (0.5, 3.0),
        "neighbor_tail99": (0.5, 4.0),
        "surface_intensity": (0.25, 20.0),
        "splash_intensity": (0.15, 8.0),
        "virtual_scale": (0.1, 8.0),
        "wls_difficulty": (0.1, 10.0),
        "ppe_difficulty": (0.1, 8.0),
        "injection_scale": (0.05, 30.0),
        "deletion_scale": (0.05, 30.0),
        "cnl_scale": (0.1, 10.0),
    }
    names = list(bounds)
    for _ in range(iterations):
        vals = {}
        for name in names:
            lo, hi = bounds[name]
            vals[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        candidate = Knobs(**vals)
        pr = predict(base, candidate)
        sc, er = score(pr, target, weights)
        if sc < best_score:
            best, best_pred, best_score, best_errors = candidate, pr, sc, er

    for radius in (0.35, 0.18, 0.08, 0.035):
        current = best.as_dict()
        for _ in range(max(200, iterations // 20)):
            vals = {}
            for name in names:
                lo, hi = bounds[name]
                vals[name] = float(np.clip(current[name] * math.exp(rng.normal(0.0, radius)), lo, hi))
            candidate = Knobs(**vals)
            pr = predict(base, candidate)
            sc, er = score(pr, target, weights)
            if sc < best_score:
                best, best_pred, best_score, best_errors = candidate, pr, sc, er
                current = best.as_dict()
    return best, best_pred, best_score, best_errors


def piecewise_row_lengths(n: int, mean: float, p90: float, p99: float, rng: np.random.Generator) -> np.ndarray:
    """Generate row lengths with controlled mean/P90/P99.

    A four-band mixture makes quantile matching deterministic and transparent.
    A final integer adjustment matches the requested mean while preserving order.
    """
    u = rng.random(n)
    low = max(1.0, 0.55 * mean)
    mid = max(low, min(p90, 1.06 * mean))
    high = max(mid, p90)
    tail = max(high, p99)
    x = np.empty(n, dtype=np.float64)
    m0 = u < 0.50
    m1 = (u >= 0.50) & (u < 0.90)
    m2 = (u >= 0.90) & (u < 0.99)
    m3 = u >= 0.99
    x[m0] = rng.uniform(0.75*low, 1.05*low, m0.sum())
    x[m1] = rng.uniform(0.90*mid, 1.02*high, m1.sum())
    x[m2] = rng.uniform(high, max(high+1.0, 0.98*tail), m2.sum())
    x[m3] = rng.uniform(tail, 1.08*tail, m3.sum())
    x = np.maximum(1, np.rint(x)).astype(np.int32)
    desired = int(round(mean*n))
    delta = desired - int(x.sum())
    if delta != 0:
        order = rng.permutation(n)
        sign = 1 if delta > 0 else -1
        remaining = abs(delta)
        k = 0
        while remaining > 0 and k < n*100:
            idx = order[k % n]
            if sign > 0 or x[idx] > 1:
                x[idx] += sign
                remaining -= 1
            k += 1
    return x


def assign_types(n: int, target: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    # 0 interior, 1 near wall, 2 free surface, 3 splash, 4 wall/other
    ratios = [
        clip_ratio(target.get("near_wall_ratio", 0.0)),
        clip_ratio(target.get("free_surface_ratio", 0.0)),
        clip_ratio(target.get("splash_ratio", 0.0)),
    ]
    s = sum(ratios)
    if s > 0.95:
        ratios = [r*0.95/s for r in ratios]
    counts = [int(round(n*r)) for r in ratios]
    interior = max(0, n - sum(counts))
    arr = np.concatenate([
        np.zeros(interior, dtype=np.uint8),
        np.full(counts[0], 1, dtype=np.uint8),
        np.full(counts[1], 2, dtype=np.uint8),
        np.full(counts[2], 3, dtype=np.uint8),
    ])
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n-len(arr), dtype=np.uint8)])
    rng.shuffle(arr)
    return arr[:n]


def generate_synthetic(out: Path, profile: dict[str, float], seed: int, materialize_csr: bool) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(round(profile["particles"]))
    row_len = piecewise_row_lengths(n, profile["neighbor_mean"], profile["neighbor_p90"], profile["neighbor_p99"], rng)
    ptype = assign_types(n, profile, rng)
    virtual_counts = rng.binomial(np.maximum(row_len, 1), min(profile.get("virtual_particle_ratio", 0.0), 0.8)).astype(np.int16)
    wls_bad = rng.random(n) < profile.get("wls_ill_conditioned_ratio", 0.0)
    row_ptr = np.empty(n+1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_len, out=row_ptr[1:])
    total_edges = int(row_ptr[-1])

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "particle_type.npy", ptype)
    np.save(out / "neighbor_row_length.npy", row_len)
    np.save(out / "csr_row_ptr.npy", row_ptr)
    np.save(out / "virtual_neighbor_count.npy", virtual_counts)
    np.save(out / "wls_ill_conditioned.npy", wls_bad)

    if materialize_csr:
        # Locality-biased synthetic columns: most neighbors are near the row id.
        col_idx = np.empty(total_edges, dtype=np.int32)
        cursor = 0
        for i, degree in enumerate(row_len):
            offsets = np.rint(rng.normal(0.0, max(8.0, math.sqrt(n)), int(degree))).astype(np.int64)
            cols = (i + offsets) % n
            col_idx[cursor:cursor+degree] = cols.astype(np.int32)
            cursor += degree
        np.save(out / "csr_col_idx.npy", col_idx)

    steps = int(profile.get("timeline_steps", 200))
    injected = int(round(profile.get("injected_per_step", 0)))
    deleted = int(round(profile.get("deleted_per_step", 0)))
    rebuild = max(1, int(round(profile.get("cnl_rebuild_interval", 10))))
    with (out / "timeline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "injected", "deleted", "cnl_rebuild", "ppe_iterations"])
        writer.writeheader()
        for step in range(steps):
            writer.writerow({
                "step": step,
                "injected": max(0, int(rng.poisson(injected))) if injected else 0,
                "deleted": max(0, int(rng.poisson(deleted))) if deleted else 0,
                "cnl_rebuild": int(step % rebuild == 0),
                "ppe_iterations": max(1, int(round(rng.normal(profile.get("ppe_iterations", 1), max(1.0, 0.05*profile.get("ppe_iterations", 1))))))
            })

    realized = {
        "particles": n,
        "neighbor_mean": float(np.mean(row_len)),
        "neighbor_p90": float(np.quantile(row_len, 0.90, method="higher")),
        "neighbor_p99": float(np.quantile(row_len, 0.99, method="higher")),
        "near_wall_ratio": float(np.mean(ptype == 1)),
        "free_surface_ratio": float(np.mean(ptype == 2)),
        "splash_ratio": float(np.mean(ptype == 3)),
        "virtual_particle_ratio": float(np.sum(virtual_counts) / max(np.sum(row_len), 1)),
        "wls_ill_conditioned_ratio": float(np.mean(wls_bad)),
        "ppe_iterations": float(profile.get("ppe_iterations", 1)),
        "cnl_rebuild_interval": float(rebuild),
        "injected_per_step": float(injected),
        "deleted_per_step": float(deleted),
        "csr_nnz": total_edges,
        "materialized_col_idx": bool(materialize_csr),
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(realized, f, ensure_ascii=False, indent=2)
    return realized


def write_report(path: Path, target: dict[str, float], baseline: dict[str, float], calibrated: dict[str, float], realized: dict[str, Any], knobs: Knobs, final_score: float, errors: dict[str, float]) -> None:
    lines = [
        "# Industrial Workload Calibration Report",
        "",
        "> 目标是计算负载等价，不是几何或物理结果等价。",
        "",
        f"- Calibration score (normalized RMS): **{final_score:.4f}**",
        f"- Synthetic particles: **{realized['particles']:,}**",
        f"- Synthetic CSR NNZ: **{realized['csr_nnz']:,}**",
        "",
        "## Metric comparison",
        "",
        "| Metric | Customer target | Plate baseline | Calibrated proxy | Synthetic realized | Proxy error | Synthetic error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METRICS:
        if m not in target:
            continue
        rv = realized.get(m, float('nan'))
        se = abs(rv-target[m]) / max(abs(target[m]), 1.0 if m in {"particles", "ppe_iterations", "cnl_rebuild_interval", "injected_per_step", "deleted_per_step"} else 1e-3)
        lines.append(f"| {m} | {target[m]:.6g} | {baseline.get(m, float('nan')):.6g} | {calibrated.get(m, float('nan')):.6g} | {rv:.6g} | {errors.get(m, float('nan')):.2%} | {se:.2%} |")
    lines += ["", "## Calibrated plate knobs", "", "```json", json.dumps(knobs.as_dict(), ensure_ascii=False, indent=2), "```", "", "## Interpretation", "", "- 参数化平板代理用于后续真实时间推进和物理验证。", "- 统计合成代理直接用于邻居循环、WLS、CSR/SpMV、分支和动态管理 kernel 的性能复现。", "- 当客户 profile 更新时，重新执行本工具即可生成新 workload，而不需要共享完整 STL。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--materialize-csr", action="store_true", help="write csr_col_idx.npy; can be large")
    args = ap.parse_args()

    target_doc = load_json(args.target)
    baseline_doc = load_json(args.baseline)
    target = target_doc["metrics"]
    baseline = baseline_doc["metrics"]
    weights = target_doc.get("weights", {})
    args.out.mkdir(parents=True, exist_ok=True)

    knobs, calibrated, final_score, errors = calibrate(baseline, target, weights, args.seed, args.iterations)
    synthetic_dir = args.out / "synthetic_workload"
    realized = generate_synthetic(synthetic_dir, target, args.seed + 1, args.materialize_csr)

    result = {
        "target": target,
        "baseline": baseline,
        "calibrated_proxy": calibrated,
        "knobs": knobs.as_dict(),
        "calibration_score": final_score,
        "relative_errors": errors,
        "synthetic_realized": realized,
    }
    (args.out / "calibrated_workload.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out / "calibration_report.md", target, baseline, calibrated, realized, knobs, final_score, errors)

    with (args.out / "metric_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "target", "baseline", "calibrated", "synthetic_realized", "relative_error"])
        writer.writeheader()
        for m in METRICS:
            if m in target:
                writer.writerow({"metric": m, "target": target[m], "baseline": baseline.get(m), "calibrated": calibrated.get(m), "synthetic_realized": realized.get(m), "relative_error": errors.get(m)})

    print(f"Calibration score: {final_score:.6f}")
    print(f"Wrote: {args.out / 'calibrated_workload.json'}")
    print(f"Wrote: {args.out / 'calibration_report.md'}")
    print(f"Synthetic workload: {synthetic_dir}")


if __name__ == "__main__":
    main()

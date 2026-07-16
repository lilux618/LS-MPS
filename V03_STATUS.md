# v0.3 Status — Inclined Plate Rain

## Implemented

- Deterministic continuous rain injection
- CPU neighbor search with `scipy.spatial.cKDTree`
- Pair-count and average-neighbor workload metrics
- Gravity, pairwise viscosity and short-range incompressibility penalty
- Inclined wall projection and tangential film transport
- Particle removal/outflow accounting
- Film particle count and thickness estimates
- Speed-coloured PNG frame sequence and animated GIF
- Per-step CSV metrics and JSON summary

## Validation achieved

- Exact particle-number balance: injected = remaining + outflow
- Deterministic result for a fixed seed
- No NaN/Inf during the supplied smoke run
- Monotonic injection accounting
- Dynamic transition from falling droplets to near-wall film particles

## Not yet claimed

- The dynamic loop is not yet driven by the C++ WLS-PPE projection path.
- Film thickness is a particle-envelope metric, not a calibrated free-surface reconstruction.
- Surface tension, contact angle, air coupling, STL geometry and industrial wall laws are absent.
- The result is suitable for workload visualization and pipeline validation,
  not yet for comparison with PREONLAB engineering predictions.

## Next integration step

Replace the penalty pressure stage with:

1. WLS divergence of predicted velocity
2. WLS/constraint-based PPE assembly
3. Jacobi-preconditioned BiCGStab
4. WLS pressure-gradient correction
5. Free-surface Dirichlet and near-wall Neumann constraints

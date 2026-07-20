# v0.7 Status — CPU LS-MPS Integration Beta

## Implemented

- One automotive-oriented inclined-plate rain case only.
- Continuous deterministic rain-particle injection.
- Dynamic cell-neighbor search through SciPy `cKDTree`.
- Normalized quadratic 2-D LS-MPS/WLS moment matrices.
- Matrix regularization, condition-number and valid/fallback diagnostics.
- Internal / free-surface / splash / near-free-surface classification.
- WLS velocity Laplacian for viscous provisional velocity.
- WLS divergence source term.
- Sparse WLS pressure-Poisson assembly and BiCGStab solve.
- WLS pressure-gradient correction with deterministic divergence-reduction line search.
- Analytic inclined-wall contact and optional particle shifting.
- Existing sampling-window, section-flow and mass-conservation outputs retained.
- CPU numerical-health validation report.

## What this version proves

It closes the CPU numerical path and makes the performance/engineering sampling
interface operate on an LS-MPS pressure-projection calculation rather than only
the earlier penalty-particle visualization model.

## Remaining limitations

- It is a 2-D reference, while the industrial automotive application is 3-D.
- Wall Neumann treatment is represented by mirrored zero-normal-gradient virtual
  support in local WLS; it is not yet the full Type-A pressure-Neumann basis used
  in the reviewed `lsmps3D_CPU` repository.
- Free-surface classification is a neighbor-density reference implementation,
  not the full cubed-sphere algorithm.
- The projection line search is a stabilization mechanism while the independently
  reconstructed WLS Laplacian and gradient are being aligned.
- No experimental or commercial-software validation has been performed.
- No GPU implementation is included yet.

Therefore v0.7 is a **CPU golden-reference candidate**, not an accepted physical
truth for automotive rain.

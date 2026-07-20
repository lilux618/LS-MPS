# v0.6 Test Results

Execution date: 2026-07-16

## Checks executed

```bash
python -m py_compile python/*.py
make acceptance
python3 python/resolution_convergence.py --duration 0.06 --out outputs/resolution_smoke --spacings 0.014,0.012,0.010
```

## Results

- Python syntax checks: PASS
- 180-step validation run: PASS
- Output schema and invariant validation: PASS
- Particle accounting identity: PASS
- Maximum relative mass error: 0
- Invalid-removal ratio: 0
- Deterministic two-run check: PASS; all canonical CSV/JSON SHA-256 hashes matched
- Three-resolution smoke run: PASS
- Physical inflow volume over the short resolution smoke test remained close across spacings:
  - h=0.014: 8.20456e-4 m^3
  - h=0.012: 8.27712e-4 m^3
  - h=0.010: 8.29000e-4 m^3

## Interpretation boundary

These tests validate execution, determinism, accounting, output definitions, and resolution-test setup. They do not validate production LS-MPS physics because the current dynamic baseline still uses penalty pressure and pair viscosity.

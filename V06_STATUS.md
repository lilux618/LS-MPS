# v0.6 Status

v0.6 focuses on making the existing rain-plate baseline independently checkable before the user adopts it.

## Improvements

- Preserves the same physical inlet volume flow across particle spacings.
- Uses end-of-step timestamps instead of labeling the first updated state as time zero.
- Separates intended outlet removal from invalid domain loss.
- Adds explicit invalid-removal metrics and thresholds.
- Adds structural/schema, finite-value, monotonicity, range and conservation checks.
- Adds deterministic two-run reproducibility verification.
- Adds a one-command acceptance target.

## Acceptance command

```bash
make acceptance
```

## Important limitation

The supplied dynamic baseline still uses pairwise penalty pressure and viscosity. It validates the sampling, accounting and comparison workflow, not a production LS-MPS rain solver. The WLS-PPE integration remains the next physics milestone.

# v0.7 CPU LS-MPS Smoke Test

Command:

```bash
make lsmps-cpu-smoke
```

Observed result:

- 20/20 PPE solves converged.
- Maximum reported relative PPE residual: `9.991248645761072e-08`.
- Warm-stage mean WLS valid ratio: `0.8426818411928136`.
- Projection increased the measured WLS divergence on `0` steps.
- Relative mass-balance error: `0`.
- Invalid-particle removal ratio: `0`.

This is a numerical-health smoke test only. It does not establish experimental
accuracy for automotive rain.

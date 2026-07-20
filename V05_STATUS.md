# v0.5 Status

## Scope

Only one automotive application proxy is retained: continuous rain on an inclined plate.

## Added

- Configurable surface sampling windows;
- wet-cell coverage ratio;
- window particle count and effective fluid volume;
- mean/P90/max film thickness;
- particle-ID based section crossing and flow rate;
- cumulative section volume;
- mass-conservation time series;
- CPU-reference comparison tool with interpolated time-series metrics;
- three-resolution convergence runner;
- explicit `lsmps3D_CPU` export contract.

## Model boundary

The validation infrastructure is complete, but the bundled dynamic visual baseline still uses pairwise penalty pressure. It is not yet a physical LS-MPS golden solution. The next solver step is to replace the dynamic pressure/viscosity path with regular/pressure-Neumann WLS matrices, WLS divergence, WLS PPE and pressure correction while preserving the v0.5 output schema.

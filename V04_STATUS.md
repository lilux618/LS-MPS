# v0.4 Status

## Added

- Customer profile JSON contract;
- weighted calibration score;
- parameterized plate proxy knob fitting;
- synthetic particle-type generation;
- neighbor row-length generation matching mean/P90/P99;
- CSR row pointer generation and optional column materialization;
- virtual-particle and WLS ill-conditioned masks;
- CNL/injection/deletion/PPE time-series generation;
- Markdown and CSV comparison reports.

## Current boundary

The generated workload is a computational proxy. It can be used to compare GPU
kernels and architectures without customer STL data, but it does not reproduce
the customer's physical result. Physical validation and workload calibration
are maintained as two separate acceptance tracks.

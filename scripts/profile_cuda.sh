#!/usr/bin/env bash
set -euo pipefail
# CUDA kernels are provided as workload units. Integrate them with your device CSR input,
# then use these commands after building a GPU driver executable.
# nsys profile --trace=cuda,nvtx -o results/lsmps_workload ./bin/lsmps-bench-gpu config/wall_film.cfg
# ncu --set full --kernel-name regex:classify_surface_splash -o results/classify ./bin/lsmps-bench-gpu config/rain_injection.cfg
# ncu --set full --kernel-name regex:assemble_wls_upper45 -o results/wls ./bin/lsmps-bench-gpu config/narrow_gap.cfg
printf '%s\n' "CUDA profiling commands documented; GPU driver is the next implementation stage."

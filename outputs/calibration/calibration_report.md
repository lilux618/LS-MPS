# Industrial Workload Calibration Report

> 目标是计算负载等价，不是几何或物理结果等价。

- Calibration score (normalized RMS): **0.0000**
- Synthetic particles: **3,000,000**
- Synthetic CSR NNZ: **138,000,000**

## Metric comparison

| Metric | Customer target | Plate baseline | Calibrated proxy | Synthetic realized | Proxy error | Synthetic error |
|---|---:|---:|---:|---:|---:|---:|
| particles | 3e+06 | 100000 | 3e+06 | 3e+06 | 0.00% | 0.00% |
| neighbor_mean | 46 | 22.6 | 46 | 46 | 0.00% | 0.00% |
| neighbor_p90 | 68 | 32 | 68 | 73 | 0.00% | 7.35% |
| neighbor_p99 | 92 | 34 | 92 | 95 | 0.00% | 3.26% |
| near_wall_ratio | 0.23 | 0.075 | 0.23 | 0.23 | 0.00% | 0.00% |
| free_surface_ratio | 0.27 | 0.023 | 0.27 | 0.27 | 0.00% | 0.00% |
| splash_ratio | 0.14 | 0.217 | 0.14 | 0.14 | 0.00% | 0.00% |
| virtual_particle_ratio | 0.11 | 0.035 | 0.11 | 0.109998 | 0.00% | 0.00% |
| wls_ill_conditioned_ratio | 0.075 | 0.012 | 0.075 | 0.0748717 | 0.00% | 0.17% |
| ppe_iterations | 38 | 12 | 38 | 38 | 0.00% | 0.00% |
| cnl_rebuild_interval | 8 | 10 | 8 | 8 | 0.00% | 0.00% |
| injected_per_step | 24000 | 1200 | 24000 | 24000 | 0.00% | 0.00% |
| deleted_per_step | 21500 | 900 | 21500 | 21500 | 0.00% | 0.00% |

## Calibrated plate knobs

```json
{
  "particle_scale": 30.0,
  "support_scale": 1.2673107851462142,
  "wall_complexity": 3.066666666666667,
  "neighbor_tail90": 1.0440217391304347,
  "neighbor_tail99": 1.3294117647058823,
  "surface_intensity": 11.73913043478261,
  "splash_intensity": 0.6451612903225807,
  "virtual_scale": 3.142857142857143,
  "wls_difficulty": 6.25,
  "ppe_difficulty": 3.1666666666666665,
  "injection_scale": 20.0,
  "deletion_scale": 23.88888888888889,
  "cnl_scale": 0.8
}
```

## Interpretation

- 参数化平板代理用于后续真实时间推进和物理验证。
- 统计合成代理直接用于邻居循环、WLS、CSR/SpMV、分支和动态管理 kernel 的性能复现。
- 当客户 profile 更新时，重新执行本工具即可生成新 workload，而不需要共享完整 STL。

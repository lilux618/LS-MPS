# v0.5 汽车淋雨单场景验证规范

v0.5 只保留倾斜平板连续淋雨场景。验证分为三层：

1. **实现一致性**：同一输入下，将 GPU/候选实现与 CPU LS-MPS Reference 比较。
2. **数值可信性**：检查质量守恒、压力投影后散度下降、时间步敏感性和粒子分辨率收敛。
3. **工程有效性**：后续与平板淋雨试验、客户商软或汽车实测窗口数据比较。

当前仓库已经实现第 1 层所需的数据接口，以及第 2 层中的质量守恒和分辨率收敛框架。当前动态演示仍使用 penalty pressure，因此只能用于验证统计与比较管线；在接入真实 WLS-PPE 后，同一接口无需变化。

## 统一采样定义

配置文件：`config/rain_plate_validation.json`

采样窗口使用平板局部坐标 `u`，`u=0` 为平板上游，`u=1` 为下游。窗口输出：

- `particle_count`
- `fluid_volume_m3`
- `coverage_ratio`
- `film_mean_m`
- `film_p90_m`
- `film_max_m`

覆盖率通过窗口内等距表面单元计算。若单元内、规定壁面法向距离以内至少包含指定数量粒子，该单元判定为湿润。

截面输出：

- `crossing_particle_count`
- `instantaneous_flow_rate_m3s`
- `crossing_volume_m3`
- `cumulative_volume_m3`

流量通过粒子 ID 的跨步轨迹检测，不使用“截面附近粒子数”代替穿越事件。

## 输出文件

运行：

```bash
make validate-rain
```

输出目录 `outputs/rain_plate_v05/`：

- `metrics.csv`：全局粒子、邻居和水膜统计；
- `sampling_windows.csv`：各采样窗口时间曲线；
- `flow_sections.csv`：各流量截面时间曲线；
- `mass_conservation.csv`：注入、域内、删除水量及守恒误差；
- `summary.json`：运行摘要；
- 动画及帧文件。

## CPU Reference 数据契约

`lsmps3D_CPU` 可以承担 CPU 算法参考，但需要接入相同的平板、注入、删除和采样定义，并导出同名的：

- `sampling_windows.csv`
- `flow_sections.csv`
- `mass_conservation.csv`

长期动态粒子轨迹不要求逐点相同。前 1–10 步应额外比较邻居、WLS 矩阵、PPE RHS、压力和修正速度；长期阶段比较窗口统计时间曲线。

比较命令：

```bash
make compare-reference REFERENCE=/path/to/lsmps3D_CPU/output
```

默认阈值定义在 `rain_plate_validation.json`：

- 覆盖率曲线 RMSE ≤ 0.08；
- 窗口水体积相对 L2 ≤ 10%；
- 累计流量相对误差 ≤ 10%；
- 平均膜厚相对 L2 ≤ 15%；
- 最大相对质量误差 ≤ 1%。

这些是 Benchmark 初始工程阈值，不是汽车淋雨最终验收标准。获得客户试验或商软数据后应重新标定。

## 分辨率收敛

```bash
make resolution-check
```

默认运行 `l0=0.014/0.012/0.010 m`，输出 `outputs/resolution_convergence/resolution_summary.csv`。重点检查：

- 各窗口覆盖率；
- 水体积；
- 平均膜厚；
- 累计截面流量。

## 当前边界

CPU Reference 仓库能验证 GPU 是否忠实复现 LS-MPS 算法，但不能单独证明汽车淋雨物理真实。最终仍需平板试验、客户商软或实测数据完成工程校准。

# LS-MPS Industrial Workload Benchmark v0.1

这个项目不是完整工业求解器，而是把《LS-MPS算法原理及优化》中的核心理解量化为可运行 benchmark：

1. 均匀网格邻居搜索，输出真实邻居分布；
2. 14 方向覆盖度 + 邻居阈值的自由面/飞溅分类；
3. 二阶 3D WLS 9×9 矩矩阵；
4. Cholesky + 正则化，统计病态与失败粒子；
5. 二阶多项式制造解，验证梯度和 Laplacian 重构；
6. `bulk_uniform`、`wall_film`、`rain_injection`、`narrow_gap` 四种负载；
7. 输出 JSON/CSV，可与舜云 profile 和真实场景统计对齐；
8. 提供 CUDA 的分类和 WLS 矩阵装配 kernel，用于后续 GPU 驱动集成。

## 当前定位

这是第一阶段“算法理解 + 负载统计”版本。CPU 路径是可执行参考实现；CUDA 文件包含工业热点 kernel，但尚未接入完整 GPU 驱动、PPE 和 BiCGStab。

## 编译和运行

```bash
make
make smoke
make suite
```

结果写入 `results/*.json` 和 `results/*.csv`。

## 场景含义

- `bulk_uniform`：规则内部粒子，测 WLS 理想吞吐和数学正确性。
- `wall_film`：薄水膜 + 底部壁面，制造单侧邻居和近壁病态负载。
- `rain_injection`：水膜 + 空中孤立粒子，制造 splash、动态注入和 CNL 重建压力。
- `narrow_gap`：上下壁面之间的少层粒子，模拟齿轮箱窄间隙。
- `industrial_3m_template`：300 万粒子 GPU 目标模板，CPU 参考不建议直接运行。

## 输出指标

- 各粒子类型数量；
- 邻居均值、P50/P90/P99；
- 邻居容量截断数量；
- NeighborSearch、Classification、WLS 耗时；
- WLS 正常、正则化、失败数量；
- 梯度与 Laplacian 制造解误差。

## 与工业实现的对应

| PPT/工业模块 | Benchmark 对应 |
|---|---|
| CNL + 粒子 CSR | CPU uniform-grid neighbor list；GPU 驱动待接入 CSR |
| 自由面/飞溅识别 | `classify_particle` / `classify_surface_splash` |
| 9×9 矩矩阵 | `run_wls` / `assemble_wls_upper45` |
| Cholesky + 寄存器压缩 | CPU Cholesky；CUDA 上三角 45 元素存储 |
| 近壁病态 | `wall_film`、`narrow_gap` |
| 注入导致 CNL 重建 | `rain_injection` 配置中的注入和重建参数 |
| PPE/BiCGStab | 下一阶段加入，先用制造解锁定 WLS 正确性 |

## 下一阶段

1. GPU CSR 构建和端到端 driver；
2. Neumann WLS 约束与虚拟粒子；
3. PPE 组装与纯散度源项；
4. Jacobi + BiCGStab；
5. 粒子注入、删除和 compaction；
6. 输出 Mparticles/s、Mpairs/s、迭代数和算子占比。

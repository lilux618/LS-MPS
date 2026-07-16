# LS-MPS Industrial Workload Benchmark v0.4

本项目围绕 LS-MPS 算法的工业落地展开，当前包含四个相互独立的验证组件：

1. **WLS 算子正确性**：验证 3D 二阶矩矩阵、Cholesky 多右端求解、梯度和 Laplacian 多项式再现。
2. **PPE 稀疏求解负载**：构造与粒子邻居图一致的稀疏压力矩阵，用离散制造解验证 BiCGStab 和统计工业负载。
3. **动态雨滴可视化**：倾斜板连续注入 → 下落 → 碰壁 → 水膜输运 → 流出，输出粒子动画和质量守恒指标。
4. **工业 Workload 标定**：从客户粒子 profile（JSON）反向生成计算负载等价的代理场景和合成数据结构。

各组件先独立验证，再在后续版本将 WLS 权重接入 PPE 并驱动动态循环，避免错误相互掩盖。

## 已实现

**C++ static benchmark:**
- 均匀网格邻居搜索与邻居分布统计；
- 14 方向覆盖分类；
- Interior / FreeSurface / Splash / NearWall / Wall；
- 3D 二阶 9×9 WLS；
- Cholesky 分解、regularization 统计；
- 二阶多项式梯度和 Laplacian 制造解；
- 近壁 Neumann 与虚拟点的负载计数接口；
- CSR PPE 稀疏矩阵；
- 离散制造压力解；
- CPU BiCGStab；
- PPE unknown、NNZ、迭代数、残差和误差统计；
- bulk、wall film、rain injection、narrow gap 四类静态负载。

**Python 动态可视化 (v0.3):**
- scipy.spatial.cKDTree 邻居搜索；
- 确定性连续雨滴注入、倾斜板投影、水膜输运、流出统计；
- 速度着色粒子动画（PNG 帧序列 + GIF 合成）；
- 逐步质量守恒追踪与 CSV 指标导出。

**Workload 标定 (v0.4):**
- 客户粒子 profile JSON → 参数化平板代理 knob 拟合；
- 合成粒子类型、CSR 行长度（匹配 mean/P90/P99）；
- 虚拟粒子和 WLS 病态 mask 生成；
- CNL/注入/删除/PPE 时间序列生成；
- Markdown + CSV 标定报告。

## 当前 PPE 的边界

PPE 当前使用稳定的 graph-Laplacian 系数来验证稀疏矩阵规模、邻居间接访存、BiCGStab 迭代负载以及不同工业粒子分布对 NNZ 和收敛的影响。它还不是最终 LS-MPS PPE 离散——下一版将使用已验证的 WLS 局部分解生成压力 Laplacian 行权重，并加入真正的自由面 Dirichlet、近壁 Neumann 和虚拟粒子贡献。

## 构建与运行

C++ workload benchmark:

```bash
make              # 编译 CPU benchmark
make suite        # 运行全部四个静态负载场景
make smoke        # 快速冒烟测试 (4000 粒子)
```

单场景：

```bash
./bin/lsmps-bench config/wall_film.cfg results
```

Python 动态可视化：

```bash
make rain-demo    # 倾斜板雨滴动画，输出到 outputs/rain_plate/
```

Workload 标定：

```bash
make calibrate    # 客户 profile → 代理 workload，输出到 outputs/calibration/
```

## v0.4 CPU 测试结果摘要

| Case | Particles | PPE unknowns | NNZ | BiCGStab iterations | PPE relative L2 |
|---|---:|---:|---:|---:|---:|
| bulk_uniform | 50,000 | 6,000 | 160,953 | 265 | 3.72e-5 |
| wall_film | 80,000 | 6,000 | 76,247 | 394 | 1.75e-5 |
| rain_injection | 100,000 | 6,000 | 76,157 | 351 | 1.31e-5 |
| narrow_gap | 80,000 | 6,000 | 76,251 | 359 | 2.08e-5 |

WLS 二阶多项式再现的梯度 RMS 误差约为 1e-13 至 1e-11，Laplacian RMS 误差约为 1e-11 至 1e-9。

## 下一步

1. 将 WLS Laplacian 行权重接入 PPE；
2. 实现自由面压力 Dirichlet；
3. 实现近壁 Neumann 约束矩阵；
4. 实现虚拟粒子的真实几何补点；
5. 加入 Jacobi 预条件器；
6. CUDA 化 CSR 邻居、WLS Cholesky、PPE assembly、SpMV 与 reduction；
7. 将 WLS-PPE 投影路径接入动态循环（替换 penalty pressure）。

## 各版本特性速览

### v0.3 — 动态雨滴可视化

倾斜板连续雨滴注入 → 下落 → 碰壁 → 水膜输运 → 流出全流程。当前使用 pairwise penalty pressure 作为动态视觉基线，**尚未接入 C++ WLS 和 PPE 模块**。动画属于 benchmark-development 产物。详见 [V03_STATUS.md](V03_STATUS.md)。

### v0.4 — 工业 Workload 标定

从客户粒子 profile 反向生成计算负载等价的代理场景。不再需要完整汽车 STL 即可构建 GPU 性能代理。标定目标是计算等价性，非物理结果等价。详见 [WORKLOAD_CALIBRATION.md](WORKLOAD_CALIBRATION.md) 和 [V04_STATUS.md](V04_STATUS.md)。

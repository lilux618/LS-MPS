# LS-MPS Industrial Workload Benchmark v0.3

本项目围绕 LS-MPS 算法的工业落地展开，当前包含三个相互独立的验证组件：

1. **WLS 算子正确性**：验证 3D 二阶矩矩阵、Cholesky 多右端求解、梯度和 Laplacian 多项式再现。
2. **PPE 稀疏求解负载**：构造与粒子邻居图一致的稀疏压力矩阵，用离散制造解验证 BiCGStab 和统计工业负载。
3. **动态雨滴可视化**：倾斜板连续注入 → 下落 → 碰壁 → 水膜输运 → 流出，输出粒子动画和质量守恒指标。

前两条是 C++ benchmark（独立验证局部微分算子和全局稀疏求解器），第三条是 Python 可视化流水线。三者先分开验证，再在下一版将 WLS 权重接入 PPE 并驱动动态循环，避免错误相互掩盖。

## 已实现

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
- bulk、wall film、rain injection、narrow gap、inclined plate rain 五类负载；
- Python 动态可视化流水线（scipy.spatial.cKDTree 邻居搜索）；
- 确定性连续雨滴注入、倾斜板投影、水膜输运、流出统计；
- 速度着色粒子动画（PNG 帧序列 + GIF 合成）；
- 逐步质量守恒追踪与 CSV 指标导出。

## 当前 PPE 的边界

v0.3 的 PPE 使用稳定的 graph-Laplacian 系数来验证：

- 稀疏矩阵规模；
- 邻居间接访存；
- BiCGStab 迭代负载；
- 不同工业粒子分布对 NNZ 和收敛的影响。

它还不是最终 LS-MPS PPE 离散。下一版将使用已验证的 WLS 局部分解生成压力 Laplacian 行权重，并加入真正的自由面 Dirichlet、近壁 Neumann 和虚拟粒子贡献。

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

## v0.3 CPU 测试结果摘要

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
7. 增加真实注入/删除和 CNL rebuild 时间序列。

## 动态可视化模型边界

v0.3 rain demo 验证了确定性注入、邻居负载增长、倾斜壁面接触、水膜输运、流出统计和可视化流水线。它当前使用 pairwise penalty pressure 作为动态视觉基线，**尚未接入已独立测试的 C++ WLS 和 PPE 模块**。因此动画属于 benchmark-development 产物，并非经过验证的工业 LS-MPS 预测。接入 WLS-PPE 投影路径是下一版本的首要任务（见 [V03_STATUS.md](V03_STATUS.md)）。

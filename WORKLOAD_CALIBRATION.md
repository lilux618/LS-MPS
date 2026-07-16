# Industrial Workload Calibration

## 目标

完整汽车 STL 与倾斜平板在几何和物理结果上不同，但 GPU 主要执行相同的算法拓扑。该模块不复制客户几何，而是根据客户 profile 反向生成计算负载等价的代理场景。

标定指标包括：

- 粒子总数；
- 邻居均值、P90、P99；
- 近壁、自由面、splash 粒子比例；
- 虚拟粒子比例；
- WLS 病态率；
- PPE/BiCGStab 迭代数；
- CNL 重建周期；
- 每步注入和删除粒子数。

## 两种代理

### 参数化平板代理

输出 `calibrated_workload.json` 中的 `knobs`，用于调整粒子规模、支持半径、壁面复杂度、自由面和飞溅强度、WLS/PPE 难度及动态注入行为。该代理最终应接入真实 LS-MPS 时间推进。

### 统计合成代理

直接生成：

- `particle_type.npy`；
- `neighbor_row_length.npy`；
- `csr_row_ptr.npy`；
- 可选 `csr_col_idx.npy`；
- `virtual_neighbor_count.npy`；
- `wls_ill_conditioned.npy`；
- `timeline.csv`。

它用于无需 STL 的 GPU kernel 性能测试，重点复现循环长度、warp 发散、间接访存、SpMV、fallback 和动态管理负载。

## 使用

```bash
make calibrate
```

或：

```bash
python3 python/workload_calibration.py \
  --target config/customer_car_rain_profile.example.json \
  --baseline config/plate_rain_baseline_profile.json \
  --out outputs/calibration
```

默认不会生成完整 `csr_col_idx.npy`，避免 300 万粒子测试产生数百 MB 文件。需要完整 CSR 时增加：

```bash
--materialize-csr
```

## 客户侧需要导出的最小 profile

客户不需要共享完整模型或 STL，只需输出 JSON 中定义的统计量。更理想的版本还应增加：

- 不同粒子类型各自的邻居直方图；
- CNL cell occupancy 分布；
- WLS regularized/fallback 分类型计数；
- CSR 行长直方图；
- BiCGStab 每步迭代时间序列和残差曲线；
- 各 kernel 时间占比；
- 多卡 halo 粒子比例和通信量。

## 边界

标定分数表示统计接近程度，不代表物理结果一致。物理正确性仍需通过标准算例、实验或商软结果独立验证。

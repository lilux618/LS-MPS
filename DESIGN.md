# Design Notes v0.2

## 验证分层

### A. 局部算子层

输入邻居点云，构造 9×9 WLS 矩矩阵；通过 Cholesky 分解求解多个右端，验证二阶多项式的梯度和 Laplacian 再现。

### B. 全局求解层

从相同邻居图构建 CSR 压力矩阵，并使用离散制造解 `b=A*p_exact` 验证 BiCGStab。v0.2 使用 graph Laplacian 作为稳定基线，不宣称它已经是最终 LS-MPS PPE。

### C. 工业 workload 层

通过 wall film、rain injection、narrow gap 等生成器控制：

- 粒子类型比例；
- 邻居数分布；
- 表面和近壁比例；
- PPE unknown 与 NNZ；
- 迭代次数和间接访存规模。

## 为什么不直接把 WLS 权重塞入 PPE

WLS 微分权重、边界条件和全局矩阵的符号/缩放任何一处错误，都可能表现为 BiCGStab 不收敛。先用可控的 graph Laplacian 校验全局求解链，再替换局部行权重，能快速区分：

- WLS 离散错误；
- 边界装配错误；
- 稀疏矩阵错误；
- 求解器错误。

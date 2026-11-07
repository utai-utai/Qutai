# 量子计算与PQC

建议顺序：

- [bell.ipynb](bell.ipynb)：Bell态、概率、采样、测量与Bloch球。
- [quantum_training.ipynb](quantum_training.ipynb)、[quantum_probabilities.ipynb](quantum_probabilities.ipynb)：从一个角度开始，分别训练期望值与概率。
- [vqe.ipynb](vqe.ipynb)、[qaoa.ipynb](qaoa.ipynb)：变分能量和四节点MaxCut优化。
- [hybrid.ipynb](hybrid.ipynb)：经典层与共享参数PQC端到端混合训练。
- [barren_plateau.ipynb](barren_plateau.ipynb)：固定PQC结构，比较局部/全局cost的梯度分布、宽/小角度初始化，以及有解析答案的无纠缠对照。

所有示例仅用标准库与Qutai；默认CUDA/float32参数、complex64量子态。无CUDA可改设备为CPU。状态空间仍按`2**qubits`增长。

Barren plateau实验默认2/4/6/8比特、深度1/4，每组24个参数seed；每个点实际前向与反向，提供均值/RMS/样本方差、bootstrap区间和原始梯度图。它不强制单调曲线，也不把小规模有限模拟当普适定理。脚本 [barren_plateau.py](barren_plateau.py) 可独立运行，输出文件采用新建模式，保留已有结果。

```text
python examples/quantum/barren_plateau.py --device cuda --samples 24 --output runs/bp.json
python examples/quantum/adjoint_benchmark.py
```

后一个脚本在相同电路/参数下比较伴随法与参数移位，双方预热后交替计时，不执行参数更新。
